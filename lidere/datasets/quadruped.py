import json
import torch
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from lidere import files

from torchvision.transforms import v2

from torchvision.ops import box_convert
from lidere.functions.keypoints import random_scale_and_crop_keypoints, rotate_img_keypoints, center_crop_keypoints, pad_to_square

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


QUADRUPED_SIGMAS = [0.026, 0.067, 0.067, 0.067, 0.067, 0.025, 0.067, 0.067, 0.067, 0.067, 0.025, 0.067, 0.067, 0.067, 0.067, 0.035, 0.067, 0.035, 0.067, 0.067, 
                    0.035, 0.067, 0.035, 0.067, 0.079, 0.072, 0.062, 0.079, 0.072, 0.062, 0.089, 0.107, 0.107, 0.087, 0.087, 0.089, 0.067, 0.067, 0.067]

def build_heatmap(kps, img_size):

    grid = torch.stack(
        torch.meshgrid(torch.linspace(0, 1, img_size[0]), torch.linspace(0, 1, img_size[1]), indexing='ij')
    ).permute(1,2,0)


    kps_heatmap = kps[:,[1,0]] / img_size
    dists = (grid[:,:,None,:] - kps_heatmap).pow(2).sum(-1).sqrt()

    sigma =  5/img_size[0]
    heatmaps = torch.exp(-0.5 * (dists / sigma) ** 2) / (sigma * 2.506)
    heatmaps /= (0.001+heatmaps.max())
    return heatmaps



@njit(parallel=False, fastmath=True)
def build_heatmap_numba(kps, valid, H, W):
    """
    kps: float32 array of shape (K, 2) in pixel coords (x, y)
    H, W: image height and width
    returns: heatmaps of shape (H, W, K)
    """

    K = kps.shape[0]
    heatmaps = np.zeros((H, W, K), dtype=np.float32)

    k_range = np.argwhere(valid)[:,0]

    # Gaussian parameters (in normalized coordinates)
    sigma = 5.0 / H
    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)
    norm = 1.0 / (2.0 * np.pi * sigma * sigma)

    # Gaussian cutoff (4σ)
    radius2 = (4.0 * sigma) * (4.0 * sigma)

    # normalized pixel coordinates
    ys = np.linspace(0.0, 1.0, H)
    xs = np.linspace(0.0, 1.0, W)

    # normalize keypoints (x,y) → (y,x)
    kps_norm = np.empty_like(kps)
    for k in k_range:
        kps_norm[k, 0] = kps[k, 1] / H
        kps_norm[k, 1] = kps[k, 0] / W

    # per-keypoint max (thread-safe)
    max_vals = np.zeros(K, dtype=np.float32)

    # parallel over keypoints
    for k in k_range:
        ky = kps_norm[k, 0]
        kx = kps_norm[k, 1]
        local_max = 0.0

        for i in range(H):
            dy = ys[i] - ky
            dy2 = dy * dy

            for j in range(W):
                dx = xs[j] - kx
                d2 = dy2 + dx * dx

                if d2 < radius2:
                    val = np.exp(-d2 * inv_2sigma2) * norm
                    heatmaps[i, j, k] = val
                    if val > local_max:
                        local_max = val

        max_vals[k] = local_max

    # global normalization
    max_val = np.max(max_vals)
    heatmaps /= (max_val + 1e-3)

    return heatmaps



class QuadrupedDataset(Dataset):
    
    def __init__(self, split, dataset=None, img_size=(256, 256), aug_scale=(0.88,1.8), aug_fac=2, intermediate_size=1200, 
                 aug_rot=15, aug_min_kps=3, aug=None, visibility=2, subsample=None):

        assert split in {'train', 'test'}

        self.aug = aug
        self.aug_scale = aug_scale
        self.aug_rot = aug_rot
        self.visibility = visibility
        self.intermediate_size = intermediate_size
        self.img_size = img_size

        self.no_heatmap = False


        self.min_kps = aug_min_kps

        ann_file = files.get_dataset_path('Quadruped80K', 'annotations', f'{split}.json')
        self.annotations = json.load(open(ann_file))

        # there is no vis=1
        vis = torch.cat([torch.tensor(a['keypoints']).view(39,3)[:,2] for a in self.annotations['annotations']])
        assert set(vis.tolist()) == set([-1,0,2])

        self.dataset_info = json.load(open(files.get_dataset_path('Quadruped80K', f'superquadruped_dataset.json')))

        self.categories = [self.dataset_info['dataset_info']['keypoint_info'][str(i)]['name'] for i in range(39)]
        self.lr_permute = [0,1,2,4,3,10,11,12,13,14,5,6,7,8,9,15,16,17,18,19,20,21,22,23,27,28,29,24,25,26,35,32,31,34,33,30,36,38,37]

        self.img_path = files.get_dataset_path('Quadruped80K', 'images')
        
        self.img_id2ann = {x['image_id']: x for x in self.annotations['annotations']}
        self.img_id2img_info = {x['id']: x for x in self.annotations['images']}

        print(set([x['source_dataset'] for x in self.annotations['images']]))

        dataset = [dataset] if isinstance(dataset, str) else dataset
        if dataset is not None:
            filter_fun = lambda x: x['source_dataset'] in dataset
        else:
            filter_fun = lambda x: True

        self.img_ids = [(x['id'], x) for x in self.annotations['images'] if filter_fun(x)]
        
        missing_in_anns = [x for x in self.img_ids if x[0] not in self.img_id2ann]
        if len(missing_in_anns) > 0:
            print('these image ids are missing in the annotations', missing_in_anns)

        self.img_ids = [x for x in self.img_ids if x[0] in self.img_id2ann]

        self.image_size_pre_crop = None
        
        rs = []


        if self.aug is not None:
            assert self.img_size is not None
            af = aug_fac / 2.0 
            rs = [
                v2.RandomApply([v2.Grayscale(num_output_channels=3)], p=0.25),
                v2.RandomApply([v2.ColorJitter(
                    brightness=(1 - 0.5*af, 1 + 1.0*af),
                    contrast=(1 - 0.3*af, 1 + 0.5*af), 
                    saturation=(1 - 0.3*af, 1 + 0.5*af), 
                    hue=0.05)
                ], p=0.5),
                v2.RandomApply([v2.GaussianBlur(kernel_size=21, sigma=(1,1+2*af))], p=0.2)
            ]

        if self.img_size is not None:
            self.meshgrid = torch.stack(
                torch.meshgrid(torch.linspace(0, 1, self.img_size[0]), torch.linspace(0, 1, self.img_size[1]), indexing='ij')
            ).permute(1,2,0)

        # elif self.img_size is not None:
        #     rs = []
        # else:
        #     rs = []     

        if subsample is not None:
            self.img_ids = self.img_ids[::subsample]   

        self.transform = transforms.Compose(rs + [transforms.ToTensor()])

        # self.gaussian = torch.compile(gaussian)

    def __len__(self):
        return len(self.img_ids)
    
    def show(self, idx=0):
        from matplotlib import pyplot as plt
        sample = self[idx]

        print(sample['visible'])
        print(sample['keypoints'].shape, )

        plt.imshow(sample['image'][:,0].permute(1,2,0))
        plt.imshow(sample['heatmaps'].sum(0), alpha=0.5)

        plt.scatter(*sample['keypoints'].T, c=sample['visible'])

    def __getitem__(self, idx):
        img_id, img_info = self.img_ids[idx]

        file_name = img_info['file_name']
        ann = self.img_id2ann[img_id]

        img = Image.open(os.path.join(self.img_path, file_name)).convert('RGB')
        img_size = img.size

        img_info = self.img_id2img_info[img_id]

        assert img_info['width'] == img_size[0] and img_info['height'] == img_size[1]

        bb = torch.tensor(ann['bbox'])[None]
        bb = box_convert(bb, 'xywh', 'xyxy').view(2,2)

        kps = torch.tensor(ann['keypoints']).view(39, 3)
        valid = kps[:,2] >= 2
        valid2= kps[:,2]
        vis = kps[:,2]
        kps = kps[:,:2]

        img = self.transform(img)

        if self.intermediate_size is not None:
            scale_fac = (torch.tensor(img_size)/ self.intermediate_size).max()
        else:
            scale_fac = 1

        target_res = (torch.tensor(img_size)/scale_fac).long().flip(0).tolist()
        
        img = torch.nn.functional.interpolate(img[None], target_res, mode='bicubic')[0]
        kps = kps / scale_fac
        bb = bb / scale_fac

        if self.aug is not None:
            img, kps, valid_, _ = rotate_img_keypoints(img, kps, torch.distributions.Uniform(-self.aug_rot, self.aug_rot).sample((1,)).item())
            valid = valid & valid_
            img, kps, valid_, _ = random_scale_and_crop_keypoints(img, kps, scale_range=self.aug_scale, crop_size=self.img_size, min_keypoints=self.min_kps)
            valid = valid & valid_
        else:
            img, kps[valid], bb = pad_to_square(img, kps[valid], bb)

        kps[~valid] = 0

        if not self.no_heatmap:
            heatmaps = torch.from_numpy(build_heatmap_numba(kps[:,].numpy(), valid.numpy(), img.shape[1], img.shape[2]))
            heatmaps = heatmaps.permute(2,0,1)
        else:
            heatmaps = torch.zeros(1)

        return {
            'id': file_name,
            'image': img[:, None], #.clip(0,1),
            'bbox': bb.flatten().long(),
            # 'annotations': ann,
            'keypoints': kps,
            'visible': vis,
            'valid': valid2,
            'img_size': torch.tensor([img_info['height'], img_info['width']]),
            'area': ann['area'],
            'heatmaps': heatmaps.half(),
            # 'img_info': img_info,
        }