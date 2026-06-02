import torch
from lidere.tasks.base import OnlyLossEvaluation
from torch.utils.data import DataLoader
from torch import nn
import numpy as np

from lidere.functions.keypoints import square_crop_by_keypoints, square_crop_by_bbox


def heatmaps_to_kps(heatmaps):

    assert heatmaps.ndim == 3

    shape = heatmaps.shape[1:]
    hm = heatmaps.flatten(1).max(1)
    
    kps = torch.stack(torch.unravel_index(hm.indices, shape)).T

    return torch.cat([kps, hm.values[:,None]], dim=1)


class PoseEstimationTask(OnlyLossEvaluation):

    def __init__(self, rmse_norm_indices=None, k_falloff=(0.1,), eval_size=None, crop=None, loss='bce', flip_eval=False, extra_output=False):
        self.key_name = 'heatmaps'
        self.loss = loss
        self.flip_eval = flip_eval
        self.rmse_norm_indices = rmse_norm_indices
        self.eval_size = tuple(eval_size) if eval_size is not None else None
        self.crop = crop
        self.extra_output = extra_output

        print(f'flip: {flip_eval}, eval_size: {eval_size} {loss}')

        assert isinstance(k_falloff, float) or isinstance(k_falloff, (list, tuple))
        self.k_falloff = torch.tensor(k_falloff)

    def loss_function(self, outputs, labels, iteration=None):

        out_scaled = torch.nn.functional.interpolate(
                outputs['heatmaps'][:,:,0], 
                labels['heatmaps'].shape[2:],
                mode='bilinear',
        )

        if self.loss == 'ce':

            gt_heatmaps = labels['heatmaps']
            gt_heatmaps_ = torch.cat([gt_heatmaps, (1-gt_heatmaps.sum(1))[:, None]], dim=1)

            loss = torch.nn.functional.cross_entropy(
                out_scaled.permute(1,0,2,3).flatten(1).T, 
                gt_heatmaps_.permute(1,0,2,3).flatten(1).T.cuda()
            )
        elif self.loss == 'bce':
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                out_scaled.flatten(1, 3), 
                labels['heatmaps'].cuda().flatten(1, 3), 
                reduction='mean'
            )
        elif self.loss == 'mse':
            loss = torch.nn.functional.mse_loss(out_scaled, labels['heatmaps'].cuda(), reduction='sum')
        else:
            raise ValueError('invalid loss function')
        
        return loss, dict()

    def heatmaps_to_kps(self, heatmaps):
        return torch.stack([heatmaps_to_kps(heatmaps[i]) for i in range(len(heatmaps))]).cpu()

    def evaluate(self, model, dataset, bs=8, n_workers=1, max_iterations=None):

        model.eval()

        if bs != 1:
            import warnings
            warnings.warn('batch size is set to 1')

        bs = 1

        # TODO: if shuffled mAP scores change, not clear why.
        loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=n_workers)

        losses = []
        areas, scores, preds, gts, img_sizes, offsets, sample_ids, bboxes = [], [], [], [], [], [], [], []
        # sample_ids = []
        scales = []

        for i_iter, sample in enumerate(loader):

            
            assert sample['keypoints'].shape[0] == 1

            model.eval()

            if sample['keypoints'].sum() == 0:
                print('ignore sample without keypoints')
                continue

            pre_crop_size = sample['image'].shape[3:]

            bbox = torch.tensor(sample['bbox'])
            bbox = bbox[0]

            if self.crop is not None:
                
                crop_mode, crop_pad = self.crop

                if crop_mode == 'kps':
                    crops, kps_crop = square_crop_by_keypoints(sample['image'][:, :, 0], sample['keypoints'], padding=crop_pad, resize=self.eval_size)
                elif crop_mode == 'bbox':
                    crops, kps_crop, offset = square_crop_by_bbox(sample['image'][0,:,0], bbox, sample['keypoints'][0], padding=crop_pad)
                    crops, kps_crop = crops[None], kps_crop[None]
                    bbox[:2] -= offset
                    offsets += [offset]
                else:
                    raise ValueError('invalid mode')

                img = crops[:, :, None]
                print(img.shape)
                print()
                sample['keypoints'] = kps_crop
                sample['keypoints'][0, sample['valid'][0]!=2] = 0
            else:
                img = sample['image']

            with torch.no_grad():
                pred0 = model(sample)

            assert pred0['heatmaps'].shape[0] == 1, 'batch size must be 1'
            heatmaps = pred0['heatmaps'][0]
            losses += [self.loss_function(pred0, sample)[0].detach()]
            
            assert self.eval_size is None or sample['image'].shape[3:] == self.eval_size, f'{sample["image"].shape[3:]} vs. {self.eval_size}'

            if self.flip_eval:
                with torch.no_grad():
                    img_f = img.flip(-1)
                    heatmaps_hflip = model(dict(image=img_f))['heatmaps'][0]

                # flip back and left-right permute maps
                if self.loss == 'bce':
                    heatmaps_hflip = heatmaps_hflip.flip(-1)[dataset.lr_permute]
                    heatmaps = (heatmaps + heatmaps_hflip).mul(0.5)
                elif self.loss == 'ce':
                    heatmaps_hflip = heatmaps_hflip.flip(-1)
                    heatmaps_hflip[:-1] = heatmaps_hflip[:-1][dataset.lr_permute]
                    heatmaps = (heatmaps + heatmaps_hflip).mul(0.5)

            # scale back to the crop size (i.e. original scale if self.eval_size is None)

            heatmaps = nn.functional.interpolate(heatmaps, img.shape[3:], mode='bicubic', align_corners=False)[:,0]

            if self.loss == 'bce':
                heatmaps = heatmaps.sigmoid()
            elif self.loss == 'ce':
                heatmaps = heatmaps.softmax(dim=0)
                heatmaps = heatmaps[:-1]
                    
            # return heatmaps

            kps = heatmaps_to_kps(heatmaps).cpu()
            kps = kps[:, [1,0,2]]

            # scale by ratio of image size before cropping and original image size
            fac = sample['img_size'] / torch.tensor(pre_crop_size)
        
            # print(sample['image'].shape, fac, sample['img_size'], pre_crop_size)

            preds += [kps[:,:2]*fac[None,:]]
            gts += [sample['keypoints']*fac[:,None,:]]
            scores += [kps[:,2]]
            areas += sample['area']
            #bboxes += [torch.tensor(sample['bbox'])]
            bboxes += [bbox]
            sample_ids += [sample['id'][0]]
            # sample_ids += sample['img_info']['id'].tolist()

            img_sizes += [sample['img_size']]
            if 'scale' in sample:
                scales += sample['scale'].tolist()

            if max_iterations is not None and i_iter >= max_iterations:
                break

        
        preds, gts = torch.cat(preds, dim=0), torch.cat(gts, dim=0)

        valid_min2_kps = (gts.sum(2) > 0).sum(1) > 1
        valid = gts.sum(2) > 0
        valid = valid & valid_min2_kps[:, None]

        areas, bboxes, img_sizes = torch.tensor(areas), torch.stack(bboxes), torch.cat(img_sizes, dim=0)
        per_sample_scores = torch.tensor([s[valid[i]].mean() for i, s in enumerate(torch.stack(scores))])        

        img_sizes = img_sizes[:,[1,0]]
        dists = (preds - gts).pow(2).sum(-1).sqrt()

        k = (2*self.k_falloff)**2
        oks_ = torch.exp(
            -dists.pow(2) / (2 * areas[:,None] * k[None,:])
        )

        oks_ = torch.tensor([oks_[i, valid[i]].mean() for i in range(len(oks_))])
        coco_map = self.coco_eval(gts, preds, per_sample_scores, sample_ids, valid, areas, img_sizes, bboxes)[0]
        
        valid_samples = (valid.sum(1) > 0)

        ne_metric = dict()

        extra = dict()
        if self.extra_output:
            extra = dict(
                oks=oks_.tolist(),
                preds=preds,
                gts=gts,
                bboxes=bboxes,
                sample_ids=sample_ids,
                scores=scores,
                areas=areas,
                img_sizes=img_sizes,
                offsets=offsets,
                valid=valid,
            )

        return dict(
            rmse=dists[valid].mean().item(),
            # map=np.asarray(precisions).mean(),
            map=coco_map,
            mean_oks=oks_[valid_samples].mean().item(),
            loss=torch.stack(losses).mean().cpu().item(),
            **ne_metric,
            **extra
        )
    

    def coco_eval(self, gts, preds, scores, sample_ids, valid, areas, img_sizes, boxes):
        # evaluate using COCO tools

        # print(gts.sum(), preds.sum(), sample_ids[:5], valid.sum(), areas.sum(), img_sizes.sum())

        valid_kps = (valid.sum(0) > 0)
        # valid_kps = (torch.ones(39) > 0)

        dists = (preds - gts).pow(2).sum(-1).sqrt()
        print('RMSE:', dists[valid].mean().item())
        print('bad indices', [(i, d[v].mean()) for i, (d, v) in enumerate(zip(dists, valid)) if d[v].mean() > 100])



        import re
        to_int = lambda x: int(re.sub(r'[^0-9]', '', x))

        annotations = [
            dict(
                image_id=to_int(iid.split('_')[-1][:-4]),
                id=i,
                keypoints=torch.cat([kps, v[:,None].float()], dim=1)[valid_kps].flatten().tolist(),
                bbox=bb.tolist(), # [10,10,100,1000], # not needed
                area=a.item(),
                iscrowd=0,
                category_id=1,
                num_keypoints=int(v.sum().item())
            ) 
            for i, (iid, kps, v, a, bb) in enumerate(zip(sample_ids, gts, valid, areas, boxes))
            #if v.float().sum() > 0 and iid in sample_ids
        ]
        images = [dict(id=to_int(iid.split('_')[-1][:-4]), height=img_size[1], width=img_size[0]) for iid, img_size in zip(sample_ids, img_sizes.tolist())]


        coco_categories = [dict(id=1, name="person", supercategory="person", keypoints=[f"kp{i}" for i in range(valid_kps.sum())], skeleton=[])]
        coco_gt = dict(images=images, annotations=annotations, categories=coco_categories)


        coco_preds = [
            dict(
                image_id=to_int(iid.split('_')[-1][:-4]),
                category_id=1,
                keypoints=torch.cat([kps, v[:,None].float()], dim=1)[valid_kps].flatten().tolist(),
                score=1, # use 1 to be consistent with mmpose


            ) for iid, kps, v, s in zip(sample_ids, preds, valid, scores)
            #if v.sum() > 0  and iid in sample_ids
        ]
    
        print('number of valid predictions', len(coco_preds))

        from pycocotools.coco import COCO
        import json
        import tempfile
        from pycocotools.cocoeval import COCOeval


        gt_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(coco_gt, gt_file)
        gt_file.close()

        coco_gt = COCO(gt_file.name)
        coco_dt = coco_gt.loadRes(coco_preds)

        coco_eval = COCOeval(coco_gt, coco_dt, iouType="keypoints")
        coco_eval.params.kpt_oks_sigmas = np.array(self.k_falloff)[valid_kps]

        coco_eval.evaluate()    # compute OKS matches
        coco_eval.accumulate()  # compute precision–recall and AP
        coco_eval.summarize()   # print metrics

        return coco_eval.stats


    def show(self, sample, model, apply_bb=False):
        from lidere import utilities as ut
        from matplotlib import pyplot as plt
        from torchvision.ops import box_convert

        assert sample['image'].shape[0] == 1

        if apply_bb:
            bb = torch.tensor(sample['bbox'])
            bb = box_convert(bb, 'xywh', 'xyxy')
            sample['image'] = sample['image'][:,:,:, bb[1]:bb[3], bb[0]:bb[2]]
            sample['keypoints'] = sample['keypoints'] - bb[:2]

        pred = ut.predict(model, sample)

        mask = (sample['visible'][0] == 2)

        hm = pred['heatmaps'][0,:,0]

        hm = torch.nn.functional.interpolate(hm[None], sample['image'].shape[3:])[0]
        
        kps = heatmaps_to_kps(hm).cpu()
        kps, scores = kps[:,[1,0]], kps[:,2]

        hm_ = hm.sigmoid().sum(0)

        
        dist = (sample['keypoints'][0, mask] - kps[mask]).pow(2).sum(-1).sqrt()
        print('RMSE', dist.mean())

        plt.imshow(sample['image'][0,:,0].permute(1,2,0))
        plt.imshow(hm_.cpu().numpy(), alpha=0.6)


        print(kps.shape)
        for i in range(len(kps)):
            if mask[i]:
                # print(i, kps[i])
                plt.scatter(*kps[i], c=plt.cm.tab20(i % 20))
                plt.scatter(*sample['keypoints'][0, i], marker='+', c=plt.cm.tab20(i % 20))


        return hm
