
from PIL import Image
from os.path import join
import os
from torchvision.transforms.functional import to_tensor
from lidere import files
from lidere.datasets.functions import SegmentationTransforms


class CamouflagedObjectSegmentation(object):
    """
    for an explanation see Appendix C for ConvLora
    """

    def __init__(self, split, test_version='CAMO', aug=None, img_size=224,):
        
        self.aug = aug
        self.img_size = img_size

        self.transform = SegmentationTransforms(aug=self.aug, crop_size=(img_size, img_size), up_fac=1.5)


        print('init camo', split, img_size)
        self.split = split
        self.test_version = test_version

        if split == 'train':
            self.base_path = join(files.get_path('DATA_ROOT'), 'camo_sem_seg', 'TrainDataset')
        if split == 'val':
            self.base_path = join(files.get_path('DATA_ROOT'), 'camo_sem_seg', 'ValDataset')
        if split == 'test':
            self.base_path = join(files.get_path('DATA_ROOT'), 'camo_sem_seg', 'TestDataset', test_version)
        
        self.samples = [x[:-4] for x in os.listdir(os.path.join(self.base_path, 'GT'))]

    def sample(self, bs=4, shuffle=False):
        from torch.utils.data import DataLoader
        return next(iter(DataLoader(self, batch_size=bs, shuffle=shuffle)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prefix = self.samples[idx]

        img = to_tensor(Image.open(os.path.join(self.base_path, 'Imgs', prefix + '.jpg')).convert('RGB'))
        mask = to_tensor(Image.open(os.path.join(self.base_path, 'GT', prefix + '.png')))[0].byte()

        image, sem = self.transform(img, mask)

        # if self.aug:
        #     image, sem = joint_image_mask_augment(img, mask, self.img_size, upscale_fac=(1,1.5))
        #     sem = sem[None]
        # else:
        #     image = self.transform(img)
        #     sem = self.target_transform(mask[None]).byte()# [0][None].byte()

        return dict(
            image=image[:,None],
            sem=sem[None],
            id=str(idx)
        )
        