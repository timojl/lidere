import os
import warnings
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

class ImageFolderDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, size=None):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.size = size  # (H, W) or None
        img_ids = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.jpg')}
        lbl_ids = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith('.png')}
        no_lbl = img_ids - lbl_ids
        no_img = lbl_ids - img_ids
        if no_lbl or no_img:
            warnings.warn(f'{len(no_lbl)} images without labels, {len(no_img)} labels without images')
        self.ids = sorted(img_ids & lbl_ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        img = Image.open(os.path.join(self.img_dir, id_ + '.jpg')).convert('RGB')
        lbl = Image.open(os.path.join(self.lbl_dir, id_ + '.png'))
        if self.size is not None:
            wh = (self.size[1], self.size[0])  # PIL takes (W, H)
            img = img.resize(wh, Image.BILINEAR)
            lbl = lbl.resize(wh, Image.NEAREST)  # NEAREST preserves class ids
        return {
            'id': id_,
            'image': pil_to_tensor(img).float().mul(1/255)[:,None],            # uint8, (3, H, W)
            'sem':   pil_to_tensor(lbl)[0].byte()[None],  # int64, (H, W)
        }
    

class ImageListDataset(Dataset):
    def __init__(self, pairs, size=None):
        # pairs: list of (img, lbl) PIL.Image tuples
        self.pairs = pairs
        self.size = size  # (H, W) or None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img, lbl = self.pairs[i]
        img = img.convert('RGB')
        if self.size is not None:
            wh = (self.size[1], self.size[0])  # PIL takes (W, H)
            img = img.resize(wh, Image.BILINEAR)
            lbl = lbl.resize(wh, Image.NEAREST)  # NEAREST preserves class ids
        return {
            'id': str(i),
            'image': pil_to_tensor(img).float().mul(1/255)[:,None],  # float32, (3, 1, H, W)
            'sem':   pil_to_tensor(lbl)[0].byte()[None],             # uint8, (1, H, W)
        }