
import torchvision
import torch
import os
from PIL import Image

from lidere.datasets.functions import SegmentationTransforms

from lidere import files
import xml.etree.ElementTree as ET
from lidere.datasets.functions import object_centers_to_center_map
from lidere import utilities as ut


CLASS_NAMES = ['Blueberry leaf','Tomato leaf yellow virus', 'Raspberry leaf', 'Peach leaf', 'Corn leaf blight', 'Potato leaf early blight', 'Tomato Septoria leaf spot', 'Tomato leaf', 'Strawberry leaf', 'Bell_pepper leaf', 'Tomato leaf bacterial spot', 'Bell_pepper leaf spot', 'Potato leaf late blight', 'Squash Powdery mildew leaf', 'Tomato mold leaf', 'Cherry leaf', 'grape leaf', 'Apple leaf', 'Apple rust leaf', 'Tomato Early blight leaf', 'Apple Scab Leaf', 'Tomato leaf late blight', 'Soyabean leaf', 'grape leaf black rot', 'Tomato leaf mosaic virus', 'Corn rust leaf','Corn Gray leaf spot']

VAL_INDICES = [1040, 2127,  664, 1614, 1957, 1518, 1945, 2194, 1227, 1580,  255,  746,
         201,  547,  709, 2324,  402, 2284,  588, 1665, 1924,  585, 1651, 2328,
        2226,  927, 1684, 1917,  353, 1748, 1513, 1500, 1795,  434, 1611,  710,
        1626, 1240, 1553, 1983, 1656,  999,  798,  104, 1271, 1818, 1441,  166,
        1211,   86,  640,  609,   15, 1829,  407,  225, 1146, 2223, 1808, 2300,
        2276, 2333, 1119, 2248, 2109,  559, 1838,  331,  715,   88, 1467, 1811,
        1203, 1991,  223,  678, 1247,  787, 2238, 1116,  729, 1063,  497, 1578,
        1683,  724,  894, 1834,  932, 1489,  499,  377,  851, 2344, 1718, 1015,
        2252,  540,  435,  505, 1587, 1604,  433, 1816,  706, 2031,  944,  586,
        2012,  671,  815,   53,   29,   65, 1612,  141, 1226, 1550, 1088, 1619,
         704,  116,  258, 2272,  931, 2256, 1018, 1512,  660,  806, 2273,  733,
         153,  471, 2102, 2329, 1941, 1316, 1787, 1187, 1907,  770, 2350,  656,
        2274, 1915, 1507, 1650,  239, 2091,  878, 1745,  838, 1050, 2055, 1914,
        1692, 1633,  994, 2025,  600, 1430,  383,  351, 2202, 1451,  888, 1802,
         226, 1628, 1581,   56,  525,  449,  567,  622, 1851,  750, 2171, 1862,
        1622,   68, 1735,  252, 2219,  300, 1598,  939,  840, 1559,  193,  809,
           4, 1870,   55,   71,  135, 2138, 1134, 1413]


class PlantDoc(object):

    def __init__(self, split, aug=None, img_size=224, p_pad=0, up_fac=1.25, max_samples=None):

        self.aug = aug
        self.root_path = os.path.join(files.get_dataset_path('plantdoc'),  'PlantDoc-Object-Detection-Dataset-master')

        self.subset, indices = {
            'train': ('TRAIN', slice(0, 2000)),
            'val': ('TRAIN', slice(2000, None)),
            'train+': ('TRAIN', slice(0, None)),
            'train2': ('TRAIN', [i for i in range(2352) if i not in VAL_INDICES]),
            'val2': ('TRAIN', VAL_INDICES),
            'test': ('TEST', slice(0, None))
        }[split]

        file_list = os.listdir(os.path.join(self.root_path, self.subset))
        self.prefixes = sorted(list(set(os.path.splitext(f)[0] for f in file_list)))
        print('pre filter', len(self.prefixes))
        self.prefixes = [p for p in self.prefixes if os.path.isfile(f'{self.root_path}/{self.subset}/{p}.jpg')]
        self.prefixes = [p for p in self.prefixes if os.path.isfile(f'{self.root_path}/{self.subset}/{p}.xml')]
        # self.prefixes = [p for p in self.prefixes if p not in {'Hydrangea+%2527Claudie%2527%252C+Powdery+Mildew.JPG'}]  # empty
        if isinstance(indices, slice):
            self.prefixes = self.prefixes[indices]
        elif isinstance(indices, list):
            self.prefixes = [self.prefixes[i] for i in indices]
        else:
            raise ValueError

        if max_samples is not None:
            self.prefixes = self.prefixes[:max_samples]

        print('post filter', len(self.prefixes))
        self.aug_transform = SegmentationTransforms(aug=aug, crop_size=img_size, p_pad=p_pad, up_fac=up_fac)
        

    def parse(self, prefix):
        tree = ET.parse(f"{self.root_path}/{prefix}.xml")
        root = tree.getroot()

        objects = []
        for obj in root.findall("object"):
            bbox_elem = obj.find("bndbox")
            bbox = (
                int(bbox_elem.find("xmin").text),
                int(bbox_elem.find("ymin").text),
                int(bbox_elem.find("xmax").text),
                int(bbox_elem.find("ymax").text)
            )
            label = obj.find("name").text
            difficulty = int(obj.find("difficult").text)
            truncated = int(obj.find("truncated").text)
            objects.append((bbox, label, difficulty, truncated))
        return objects

    def __len__(self):
        return len(self.prefixes)

    def class_name(self, k):
        return CLASS_NAMES[k]

    def get_bboxes_by_sample_id(self, sample_id):
        idx = int(sample_id.split('-')[1])
        prefix = self.prefixes[idx]    
        boxes = self.parse(self.subset + '/' + prefix)
        labels = [CLASS_NAMES.index(b[1]) if b[1] in CLASS_NAMES else None for b in boxes]
        # boxes = [b[0] for b in boxes]

        if len(labels) > 0:
            labels, boxes = zip(*[(l, b[0]) for l, b in zip(labels, boxes) if l is not None])
        else:
            labels, boxes = [], []

        return torch.tensor(boxes).float(), torch.tensor(labels)

    def return_empty(self, img, sample_id, orig_img_size):
        img2, _ = self.aug_transform(img, torch.zeros(100, 100))
        # print('empty')
        return dict(
            id=sample_id,
            original_img_size=torch.tensor(orig_img_size),
            image=img2[:,None],
            centers_gauss=torch.zeros((1,) + img2.shape[1:]),
            centers_size=torch.zeros((2, 1) + img2.shape[1:]),
            label_map=torch.zeros((1,) + img2.shape[1:], dtype=torch.uint8),
            centers_binary=torch.zeros(img2.shape[1:], dtype=bool),
        )        

    def __getitem__(self, idx):
        prefix = self.prefixes[idx]    
        sample_id = f'{self.subset}-{idx}'

        img = Image.open(f'{self.root_path}/{self.subset}/{prefix}.jpg').convert('RGB')
        img = torchvision.transforms.functional.to_tensor(img)

        orig_img_size = img.shape[1:]
    
        boxes = self.parse(self.subset + '/' + prefix)
        labels = [CLASS_NAMES.index(b[1]) if b[1] in CLASS_NAMES else None for b in boxes]

        boxes = torch.tensor([b[0] for b in boxes])

        if len(boxes) == 0:
            return self.return_empty(img, sample_id, orig_img_size)

        # render masks
        _, H, W = img.shape
        masks = torch.zeros((len(boxes), H, W), dtype=torch.bool)

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            masks[i, y1:y2, x1:x2] = True

        img2, masks2 = self.aug_transform(img, masks)

        scales = masks2.flatten(1).sum(1).sqrt()
        nonzeros = [m.nonzero() for m in masks2]

        valids = [s > 0 and l is not None for s, l in zip(scales, labels)]

        labels = [l for l, valid in zip(labels, valids) if valid]

        if len(labels) == 0:
            return self.return_empty(img, sample_id, orig_img_size)

        centers = torch.stack([nonzero.float().mean(0) for nonzero, valid in zip(nonzeros, valids) if valid])

        sizes = torch.stack([nonzero.max(0).values - nonzero.min(0).values for nonzero, valid in zip(nonzeros, valids) if valid])
        sizes = sizes / torch.tensor(img2.shape[1:])
        sizes = sizes[:,[1,0]]

        import math
        scales =  5*math.sqrt(img2.shape[1]) + 0.25*scales    

        centers_binary = torch.zeros(img2.shape[1:], dtype=bool)
        centers_binary[centers[:,0].long(), centers[:,1].long()] = True

        heatmap = object_centers_to_center_map(centers, [0 for _ in labels], 1, scales*0.02, img2)        

        grid = ut.rect_meshgrid(img2.shape[1:]).permute(1,0,2) 
        box_map = torch.zeros(*img2.shape[1:], 2)
        label_map = torch.zeros(*img2.shape[1:], dtype=torch.uint8)
        best_dist = 99999*torch.ones(*img2.shape[1:])
        for s, size, center, label in zip(scales, sizes, centers/torch.tensor(img2.shape[1:]), labels):
            if s > 0:
                d = (grid - center).pow(2).sum(-1).sqrt()
                m = (d<0.1) & (d<best_dist)

                label_map[m] = label+1
                box_map[m] = size.float()
                best_dist[m] = d[m]


        return dict(
            # bboxes=boxes,
            # masks=masks2,
            original_img_size=torch.tensor(orig_img_size),
            id=sample_id,
            image=img2[:,None],
            centers_gauss=heatmap,
            label_map=label_map[None],
            centers_binary=centers_binary,
            centers_size=box_map.permute(2,0,1)[:,None]
        )

 