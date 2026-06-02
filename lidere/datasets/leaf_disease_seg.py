import numpy as np
from PIL import Image
from os.path import join, expanduser
import os
from torchvision.transforms.functional import to_tensor

from lidere.datasets.functions import joint_image_mask_augment, SegmentationTransforms
from lidere import files


class LeafDiseaseSeg(object):

    def __init__(self, split, img_size=224, p_pad=0, pad_value=0, aug=None):
        
        self.aug = aug
        self.img_size = img_size

        self.transform = SegmentationTransforms(aug=self.aug, p_pad=p_pad, pad_value=pad_value, p_blur=0.2, crop_size=img_size, up_fac=2)

        print('init leaf', split, img_size, p_pad, aug)
        self.split = split

        self.base_path = join(files.get_path('DATA_ROOT'), 'leaf_disease_segmentation')

        if not os.path.isdir(self.base_path):
            print('Dataset not found. Download from here:\nhttps://automl-mm-bench.s3.amazonaws.com/semantic_segmentation/leaf_disease_segmentation.zip')

        self.samples = np.genfromtxt(os.path.join(self.base_path, f'{split}.csv'), delimiter=',', dtype=None, encoding=None, skip_header=1)


    def sample(self, bs=4, shuffle=False):
        from torch.utils.data import DataLoader
        return next(iter(DataLoader(self, batch_size=bs, shuffle=shuffle)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        num, image, mask = self.samples[idx]

        img = to_tensor(Image.open(os.path.join(self.base_path, image)).convert('RGB'))
        mask = to_tensor(Image.open(os.path.join(self.base_path, mask)))[0].byte()

        # image = self.transform(img)[:,None]
        # mask = self.target_transform(mask)[0][None].byte()

        image, sem = self.transform(img, mask)

        # if self.aug:
        #     image, sem = joint_image_mask_augment(img, mask, self.img_size, upscale_fac=(1,1.5))
        #     sem = sem[None].float()
        # else:
        #     image = self.transform(img)
        #     sem = self.target_transform(mask[None])# [0][None].byte()


        return dict(
            image=image[:,None],
            sem=sem[None],
            id=str(idx)
        )
        
