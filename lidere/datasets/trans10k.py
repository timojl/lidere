
import torch
import os

from PIL import Image
from os.path import join

from torchvision.transforms.functional import to_tensor
from lidere.datasets.functions import SegmentationTransforms
from lidere import files



class Trans10K(object):

    def __init__(self, split, version='800px', subsets=None, aug=None, p_pad=0, pad_value=0, img_size=224, max_samples=None):
         
        self.aug = aug
        self.img_size = img_size

        self.transform, self.target_transform = None, None

        self.transform = SegmentationTransforms(aug=self.aug, p_pad=p_pad, pad_value=pad_value, crop_size=(img_size, img_size), up_fac=1.5)
                
        print('init trans10k', split, img_size)
        self.split = split
        
        subsets = subsets if subsets is not None else ['easy', ]
    
        if version == 'original':
            if self.split in ['train', 'val']:
                self.base_path = join(files.get_path('DATA_ROOT'), 'trans10k_trainval')
            else:
                self.base_path = join(files.get_path('DATA_ROOT'), 'trans10k_test')

        elif version=='800px':
            self.base_path = join(files.get_path('DATA_ROOT'), 'trans10k_all_800px')

        if split == 'train':
            images = os.listdir(os.path.join(self.base_path, 'train', 'images'))
            images = list(zip(['']*len(images), images))

        elif split == 'val':
            images1 = os.listdir(os.path.join(self.base_path, 'validation', 'easy', 'images')) if 'easy' in subsets else []
            images2 = os.listdir(os.path.join(self.base_path, 'validation', 'hard', 'images')) if 'hard' in subsets else []
            images = list(zip(['easy']*len(images1), images1)) + list(zip(['hard']*len(images2), images2))
            
        elif split == 'test':
            images1 = os.listdir(os.path.join(self.base_path, 'test', 'easy', 'images')) if 'easy' in subsets else []
            images2 = os.listdir(os.path.join(self.base_path, 'test', 'hard', 'images')) if 'hard' in subsets else []
            images = list(zip(['easy']*len(images1), images1)) + list(zip(['hard']*len(images2), images2))  

        self.images = sorted(images, key=lambda x: x[1])
        
        if max_samples is not None:
            self.images = self.images[:max_samples]

    def sample(self, bs=4, shuffle=False):
        from torch.utils.data import DataLoader
        return next(iter(DataLoader(self, batch_size=bs, shuffle=shuffle)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        subset, image = self.images[idx]

        split = self.split if self.split != 'val' else 'validation'

        img = to_tensor(Image.open(os.path.join(self.base_path, split, subset, 'images', image)).convert('RGB'))
        mask = to_tensor(Image.open(os.path.join(self.base_path, split, subset, 'masks', f'{image[:-4]}_mask.png'))).mul(255).byte()

        seg = torch.zeros(mask.shape[1:], dtype=torch.uint8)
        seg[(mask[0]==255) & (mask[1]==0)] = 1  # red
        seg[(mask[0]==255) & (mask[1]==255)] = 2  # white

        # if self.pad:
        #     img = pad_to_square(img)
        #     mask_ = pad_to_square(mask)

        img, sem = self.transform(img, seg)

        # # else:
        # if self.aug:
        #     img, seg_ = joint_image_mask_augment(img, seg, self.img_size, upscale_fac=(1,1.5))
        #     seg_ = seg_[None]
        # else:
        #     img = self.transform(img)
        #     seg_ = self.target_transform(seg[None])

        return dict(
            image=img[:,None],
            sem=sem[None]  # self.target_transform(mask)[0][None].long()
        )
        


def create_800px_version():
    for split in ['val', 'train', 'test']:
        large_images = []
        print(split)
        d = Trans10K(split, version='original', img_size=512)
        if split == 'val':
            split = 'validation'
        for subset, image in d.images:
            size = Image.open(os.path.join(d.base_path, split, subset, 'images', image)).size
            if size[0] > 2000:
                large_images += [(subset, image)]


        for subset, image in large_images:
            mask_filename =  f'{image[:-4]}_mask.png'
            img = Image.open(os.path.join(d.base_path, split, subset, 'images', image))
            mask = Image.open(os.path.join(d.base_path, split, subset, 'masks', mask_filename))
            img.thumbnail((800, 800))
            mask.thumbnail((800, 800), resample=Image.Resampling.NEAREST)
            
            new_base = os.path.join('/scratch/datasets/trans10k_800px', split, subset)
            os.makedirs(os.path.join(new_base, '..'), exist_ok=True)
            os.makedirs(os.path.join(new_base), exist_ok=True)

            os.makedirs(os.path.join(new_base, 'images'), exist_ok=True)
            os.makedirs(os.path.join(new_base, 'masks'), exist_ok=True)
            img.save(os.path.join(new_base, 'images', image))
            mask.save(os.path.join(new_base, 'masks', mask_filename))


if __name__ == "__main__":
    create_800px_version()            