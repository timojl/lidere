import torch
import random
from torchvision.transforms import functional as tvf

def rotate_img_keypoints(img, keypoints, angle, mask=None):
    import cv2

    if isinstance(img, torch.Tensor):
        img = img.numpy()

    h, w = img.shape[1], img.shape[2]
    M = cv2.getRotationMatrix2D([w//2, h//2], angle, 1).astype('float32')
    rotated_img = cv2.warpAffine(img.transpose([1,2,0]), M, (w, h))  
    rotated_img = torch.from_numpy(rotated_img).permute(2,0,1)

    keypoints = torch.cat([keypoints, torch.ones(len(keypoints))[:,None]], dim=1)

    rotated_kps = keypoints @ torch.from_numpy(M).T
    visible = (
        (rotated_kps[:, 0] >= 0) & (rotated_kps[:, 0] < w) &
        (rotated_kps[:, 1] >= 0) & (rotated_kps[:, 1] < h)
    )   

    if mask is not None:
        mask = torch.from_numpy(cv2.warpAffine(mask.numpy(), M, (w, h))  )

    return rotated_img, rotated_kps, visible, mask


def random_scale_and_crop_keypoints(image, keypoints, scale_range=(0.8, 1.2),
                                    crop_size=(256, 256), min_keypoints=5, mask=None):

    assert image.ndim == 3 and image.shape[0] == 3

    crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size

    h, w = image.shape[1:]         
    th, tw = crop_size      

    min_keypoints = min(min_keypoints, len(keypoints))

    for count in range(300):
        # scale = random.uniform(*scale_range)
        scale = torch.exp(torch.empty(()).uniform_(*torch.tensor(scale_range).log().tolist()))

        new_h = int(h * scale)
        new_w = int(w * scale)

        if new_h < th or new_w < tw:
            continue

        keypoints_ = keypoints * scale

        top = random.randint(0, new_h - th)
        left = random.randint(0, new_w - tw)

        keypoints_ = keypoints_ - torch.tensor([left, top], device=keypoints.device)

        visible = (
            (keypoints_[:, 0] >= 0) & (keypoints_[:, 0] < tw) &
            (keypoints_[:, 1] >= 0) & (keypoints_[:, 1] < th)
        )

        if visible.sum().item() >= min_keypoints:
            break
    else:
        # fallback: no valid crop found, use scale=1 with a centered crop
        new_h, new_w = h, w
        scale = 1.0
        keypoints_ = keypoints.clone()
        top = max(0, (new_h - th) // 2)
        left = max(0, (new_w - tw) // 2)
        keypoints_ = keypoints_ - torch.tensor([left, top], device=keypoints.device)

        visible = (
            (keypoints_[:, 0] >= 0) & (keypoints_[:, 0] < tw) &
            (keypoints_[:, 1] >= 0) & (keypoints_[:, 1] < th)
        )        

    # make sure the keypoints remain in the image
    # keypoints_[:, 0].clamp_(0, tw - 1)
    # keypoints_[:, 1].clamp_(0, th - 1)

    image = tvf.resize(image, [new_h, new_w])
    image = tvf.crop(image, top, left, th, tw)

    if mask is not None:
        mask = tvf.resize(mask[None], [new_h, new_w])
        mask = tvf.crop(mask, top, left, th, tw)[0]

    return image, keypoints_, visible, mask


def center_crop_keypoints(image, keypoints, crop_size=(256, 256)):
    assert image.ndim == 3 and image.shape[0] == 3
    crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
    h, w = image.shape[1:]
    th, tw = crop_size
    # Resize so the crop fits with equal padding on all sides
    scale = max(th / h, tw / w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    image = tvf.resize(image, [new_h, new_w])
    keypoints_ = keypoints * scale
    top = (new_h - th) // 2
    left = (new_w - tw) // 2
    keypoints_ = keypoints_ - torch.tensor([left, top], device=keypoints.device)
    image = tvf.crop(image, top, left, th, tw)
    return image, keypoints_


def square_crop_by_bbox(image, bbox, keypoints, padding=1.0):
    H, W = image.shape[1:3]

    x, y, w, h = bbox

    cy, cx = y + h / 2, x + w / 2
    side = max(h, w) / 2
    side = int(side * padding)

    y1 = max(0, int(cy - side))
    y2 = min(H, int(cy + side))
    x1 = max(0, int(cx - side))
    x2 = min(W, int(cx + side))

    cropped = image[:, y1:y2, x1:x2]



    shift = torch.tensor([x1, y1], device=keypoints.device)
    kp_shifted = keypoints - shift

    return cropped, kp_shifted, shift 


def pad_to_square(image, keypoints, bbox):
    assert image.ndim == 3 and image.shape[0] == 3
    h, w = image.shape[1:]
    size = max(h, w)
    pad_top = (size - h) // 2
    pad_left = (size - w) // 2
    pad_bottom = size - h - pad_top
    pad_right = size - w - pad_left
    image = tvf.pad(image, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
    keypoints_ = keypoints + torch.tensor([pad_left, pad_top], device=keypoints.device)
    bbox_ = bbox + torch.tensor([pad_left, pad_top], device=bbox.device)
    return image, keypoints_, bbox_


def square_crop_by_keypoints(images, keypoints, padding=1.1, single_kp_size=None, resize=None):
    
    all_crops, all_kps = [], []

    for img, kps_ in zip(images, keypoints):

        H, W = img.shape[-2:]

        valid = kps_.sum(1) > 0
        kps = kps_[valid]

        # bbox from keypoints
        x_min, y_min = kps.min(0).values
        x_max, y_max = kps.max(0).values

        # square side with padding
        side = max(x_max - x_min, y_max - y_min)
        side = side * padding
        side = int(torch.ceil(side))
        side = max(side, 1)

        # center
        cx = ((x_min + x_max) / 2).int()
        cy = ((y_min + y_max) / 2).int()

        if len(kps) == 1 and single_kp_size is not None:
            # fixed square size around single keypoint
            side = int(torch.ceil(torch.tensor(200 * padding)))
            side = max(side, 1)

            # make sure side fits inside the image
            side = min(side, W, H)

            # center on the keypoint
            x1 = cx - side // 2
            y1 = cy - side // 2

            # clamp to image boundaries (keeps square)
            x1 = int(torch.clamp(x1, 0, W - side))
            y1 = int(torch.clamp(y1, 0, H - side))

            x2 = x1 + side
            y2 = y1 + side
        else:

            # square crop coords
            x1 = cx - side // 2
            y1 = cy - side // 2
            x2 = x1 + side
            y2 = y1 + side

            # shift to stay inside image (keeps square)
            dx = torch.clamp(-x1, min=0) - torch.clamp(x2 - W, min=0)
            dy = torch.clamp(-y1, min=0) - torch.clamp(y2 - H, min=0)

            x1 += dx; x2 += dx
            y1 += dy; y2 += dy

            x1 = int(torch.clamp(x1, 0, W - side))
            y1 = int(torch.clamp(y1, 0, H - side))
            x2 = x1 + side
            y2 = y1 + side

            if y1 == y2:
                y1, y2 = y1 - int(H*padding), y2 + int(H*padding)
            
            if x1 == x2:
                x1, x2 = x1 - int(H*padding), x2 + int(H*padding)

            y1, x1 = max(y1, 0), max(x1, 0)

        crop = img[:, y1:y2, x1:x2]
        kps_crop = kps_ - torch.tensor([x1, y1])
        kps_crop[~valid] = 0

        if resize is not None:
            new_h, new_w = resize
            scale_x = new_w / side
            scale_y = new_h / side

            crop = torch.nn.functional.interpolate(crop[None], size=resize, mode='bilinear', align_corners=False, antialias=True)[0]
            kps_crop = kps_crop * torch.tensor([scale_x, scale_y], device=kps.device)
            
        # crop image
        all_crops += [crop]    # (C, side, side)

        # correct keypoints
        all_kps += [kps_crop]

    return torch.stack(all_crops), torch.stack(all_kps)


def scale_square_image(img, kps, target_size):

    img_shape = img.shape
    img = torch.nn.functional.interpolate(img[None], target_size, mode='bilinear')[0]
    kps = kps / torch.tensor([img_shape[2]/target_size[1], img_shape[1]/target_size[0]])
    return img, kps