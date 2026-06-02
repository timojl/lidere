import torch
import random 
import numpy as np

from torchvision.transforms import v2
from torchvision.transforms.functional import resize, crop, InterpolationMode
from torchvision import transforms
from torchvision.transforms import functional as F

from PIL import Image

def gaussian(d, sigma):
    return torch.exp(-0.5 * (d / sigma) ** 2) / (sigma * torch.sqrt(torch.tensor(2.0 * torch.pi)))


def pad_to_square(img, value=0):

    if isinstance(img, Image.Image):
        return pad_to_square_pil(img, value=value)

    assert img.ndim in {2,3}
    # assert img.shape[0] in {1,3}
    img_size = torch.tensor(img.shape[1:]) if img.ndim == 3 else torch.tensor(img.shape)

    pad = img_size.max() - img_size
    pad = torch.cat([pad // 2, pad - pad // 2])[[1,3,0,2]]
    out = torch.nn.functional.pad(img, pad.tolist(), mode='constant', value=value)

    return out

def pad_to_square_pil(img, value=0):

    from PIL import ImageOps
    max_side = max(img.size)
    return ImageOps.pad(
        img,
        size=(max_side, max_side),
        color=value,
        centering=(0.5, 0.5), 
    )


def joint_image_mask_random_crop(image, mask, target_size=224, upscale_fac=1.5, min_sum=0.3, trials=20, mask_interpolation=InterpolationMode.NEAREST):


    assert image.ndim == 3 
    assert mask.ndim in {2,3}

    if isinstance(upscale_fac, (tuple, list)):
        upscale_fac = torch.distributions.Uniform(upscale_fac[0], upscale_fac[1]).sample((1,)).item()

    image = resize(image, int(target_size*upscale_fac))

    if mask.ndim == 2:
        m = resize(mask[None], int(target_size*upscale_fac), interpolation=mask_interpolation)[0]
    elif mask.ndim == 3:
        m = resize(mask, int(target_size*upscale_fac), interpolation=mask_interpolation)

    m_sum = m.sum()
    for _ in range(trials):
        i, j, h, w = transforms.RandomCrop.get_params(image, (target_size, target_size))
        m2 = crop(m, i ,j,h, w)
        if min_sum is not None and m2.sum() > min_sum*m_sum:
            break
    

    image = crop(image, i ,j,h, w)
    return image, m2


def joint_image_mask_augment(image, mask, img_size, p_gray=0.3, p_hflip=0.5, p_col=0.5, brightness=(0.3, 1.5), contrast=(0.3, 1.5), saturation=(0.3, 1.5), upscale_fac=(1,1.5), mask_interpolation=InterpolationMode.NEAREST):
    
    if torch.rand(1).item() < p_gray:
        image = v2.functional.rgb_to_grayscale(image, num_output_channels=3)

    if torch.rand(1).item() < p_col:
        image = v2.functional.adjust_brightness(image, torch.distributions.Uniform(*brightness).sample().item())
        image = v2.functional.adjust_contrast(image, torch.distributions.Uniform(*contrast).sample().item())
        image = v2.functional.adjust_saturation(image, torch.distributions.Uniform(*saturation).sample().item())

    image, mask = joint_image_mask_random_crop(image, mask, target_size=img_size, upscale_fac=upscale_fac, mask_interpolation=mask_interpolation)

    if torch.rand(1).item() < p_hflip:
        image = image.flip(-1)
        mask = mask.flip(-1)

    return image, mask


class SegmentationTransforms(object):

    def __init__(self, crop_size=(224, 224), p_pad=0, pad_value=0, p_blur=0, up_fac=1.25, aug=None):

        assert isinstance(aug, (type(None), str))

        self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
        self.aug = aug
        self.up_fac = up_fac
        self.p_pad = p_pad
        self.pad_value = pad_value

        self.aug_transforms = transforms.Compose([
            v2.RandomApply([v2.Grayscale(num_output_channels=3)], p=0.3),
            # v2.RandomApply([v2.ColorJitter(brightness=(0.5, 1.5), contrast=(0.7, 1.3), saturation=(0.7, 1.3), hue=None)], p=0.3)
            v2.RandomApply([v2.ColorJitter(brightness=(0.5, 2), contrast=(0.5, 2), saturation=(0.5, 2), hue=None)], p=0.5),
            v2.RandomApply([v2.GaussianBlur(kernel_size=21, sigma=(1,3))], p=p_blur)  # NEW
        ])

    def resize_mask(self, mask):
        if mask.ndim == 2:
            return F.resize(mask[None], self.crop_size, interpolation=F.InterpolationMode.NEAREST)[0]
        elif mask.ndim == 3:
            return F.resize(mask, self.crop_size, interpolation=F.InterpolationMode.NEAREST)
        else:
            raise ValueError('unsupported mask size')


    def __call__(self, image, mask):
        if self.aug == 'standard' and random.random() < 0.8:

            mask = torch.from_numpy(np.array(mask)) if isinstance(mask, Image.Image) else mask

            image = torch.from_numpy(np.array(image)).permute(2,0,1).mul(1/255) if isinstance(image, Image.Image) else image

            if random.random() < 0.5 and self.crop_size is not None:  # crop probability
                assert self.crop_size[0] == self.crop_size[1]
                image, mask = joint_image_mask_random_crop(image, mask, target_size=self.crop_size[0], upscale_fac=(1, self.up_fac))

            image = self.aug_transforms(image).float()

        if self.aug in {'standard', 'hflip'} and random.random() < 0.5:
            image = F.hflip(image)
            mask  = F.hflip(mask)

        if random.random() < self.p_pad:
            image = pad_to_square(image, value=self.pad_value)
            mask = pad_to_square(mask, value=self.pad_value)

        if self.crop_size is not None:
            image = F.resize(image, self.crop_size, interpolation=F.InterpolationMode.BILINEAR)
            mask = self.resize_mask(mask)
        # otherwise, do nothing

        if isinstance(image, Image.Image):
            image = F.to_tensor(image)

        if isinstance(mask, Image.Image):
            mask =  torch.from_numpy(np.array(mask))
    
        return image, mask



def object_centers_to_center_map(centers, labels, n_classes, scales, img):
    from lidere.utilities import rect_meshgrid
    """ take a centers tensor and generate a CenterNet-like heatmap  """

    grid = rect_meshgrid(img.shape[1:]).permute(1,0,2)*torch.tensor(img.shape[1:])
    
    # centers = bboxes.view(-1,2,2).float().mean(1)[:,[1,0]]
    if labels is None:
        labels = [0 for c in centers]

    center_heatmap = torch.zeros(n_classes, *img.shape[1:])
    for s, c, l in zip(scales, centers, labels):
        dist = (grid - c).pow(2).sum(-1).sqrt()
        center_heatmap[l] += torch.exp(-dist/(2*s.pow(2)))

    return center_heatmap
