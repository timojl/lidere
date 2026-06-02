import os
import torch
from torch.utils.data import Dataset, DataLoader
from lidere import files
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor
from lidere.datasets.functions import joint_image_mask_random_crop, pad_to_square, pad_to_square_pil, SegmentationTransforms


class BSDS500(Dataset):
    def __init__(self, split, img_size=(224, 224), avg_contour=False, aug=None):

        self.split = split
        self.aug = aug
        self.img_size = img_size
        self.avg_contour = avg_contour


        self.root_dir = os.path.join(files.get_dataset_path('BSR_bsds500'), 'BSR', 'BSDS500', 'data')
        self.gt_dir = os.path.join(self.root_dir, 'groundTruth', split)
        self.files = [
            os.path.join(self.gt_dir, f)
            for f in os.listdir(self.gt_dir)
            if f.endswith('.mat')
        ]

        if img_size is not None:
            self.transform = SegmentationTransforms(aug=self.aug, crop_size=img_size, up_fac=1.5)
        else:
            self.transform = None
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        import scipy.io as sio

        mat_path = self.files[idx]

        num = os.path.basename(mat_path).split('.')[0]

        img = os.path.join(self.root_dir, 'images', self.split, str(num) + '.jpg')
        img = to_tensor(Image.open(img).convert('RGB'))

        mat_data = sio.loadmat(mat_path)
        
        # sem = torch.from_numpy(np.stack([mat_data['groundTruth'][0][i][0][0][0] for i in range(5)]))
        con = torch.from_numpy(np.stack([mat_data['groundTruth'][0][i][0][0][1] for i in range(len(mat_data['groundTruth'][0]))]))

        if self.avg_contour:
            con = con.float().mean(0)[None]
        
        if self.transform is not None:
            img, con = self.transform(img, con)

        return dict(
            contour=con,
            # sem=sem,
            image=img[:,None]

        )
