
import torch
import tqdm
import os
from lidere.tasks.base import OnlyLossEvaluation
from torchvision.ops import focal_loss


class CenterNetTask(OnlyLossEvaluation):

    def __init__(self, mode='focal', loss_term='all', with_classification=False):
        self.mode = mode
        self.key_name = 'centers_gauss'
        self.nms_threshold = 0.5
        self.loss_term = loss_term
        self.with_classification = with_classification


    def maybe_rename_outputs(self, outputs):
        if 'centers_and_sizes' in outputs:
            outputs = dict(
                centers_gauss=outputs['centers_and_sizes'],
                centers_size=outputs['centers_and_sizes'][:, 1:],
            )
        return outputs

    def inspect_dataset(self, dataset, shuffle=True):
        """ to validate that the dataset produces correct samples. """
        from matplotlib import pyplot as plt
        from torchvision.utils import draw_bounding_boxes
        from torch.utils.data import DataLoader

        sample = next(iter(DataLoader(dataset, batch_size=1, shuffle=shuffle)))

        boxes2, _, labels = self.compute_bboxes(
            sample['centers_gauss'][0,0], 
            sample['centers_size'][0,:,0],
            torch.randn(1,*sample['centers_gauss'][0,0].shape),
            lambda x: x
        )
        boxes2 = boxes2 * torch.tensor(sample['image'].shape[3:])[[1,0]].repeat(2)
        plt.imshow(draw_bounding_boxes(sample['image'][0,:,0], boxes2, [dataset.class_name(l) for l in labels]).permute(1,2,0))        

    def show_prediction(self, model, sample):
        from matplotlib import pyplot as plt
        from torchvision.utils import draw_bounding_boxes
        from torch.utils.data import DataLoader

        with torch.no_grad():
            pred = model(sample)
        pred = self.maybe_rename_outputs(pred)

        boxes2, scores, labels = self.compute_bboxes(
            pred['centers_gauss'][0,0,0].cpu().sigmoid(), 
            pred['centers_size'][0,:,0].cpu(),
            pred['class_embeddings'][0,:,0].cpu() if 'class_embeddings' in pred else None,
            model.embed_proj if hasattr(model, 'embed_proj') else None
        )

        boxes2 = boxes2 * torch.tensor(sample['image'].shape[3:])[[1,0]].repeat(2)
        boxes2 = boxes2[scores > 0.2]
        
        plt.imshow(draw_bounding_boxes(sample['image'][0,:,0], boxes2).permute(1,2,0))      

    def compute_bboxes(self, pred_centers, pred_sizes, pred_embed, embed_to_class, min_score=0.02):
        import scipy.ndimage as ndi
        import numpy as np
        from torchvision.ops import nms

        assert pred_centers.ndim == 2
        
        img_size = torch.tensor(pred_centers.shape)

        peaks = pred_centers == torch.nn.functional.max_pool2d(pred_centers[None, None], kernel_size=5, stride=1, padding=2)[0,0]
        peaks = peaks & (pred_centers > min_score)
        peaks = torch.argwhere(peaks).cpu()

        # labels = peaks[:,0]
        scores = pred_centers[peaks[:,0], peaks[:,1]].cpu()
        sizes = (pred_sizes[:, peaks[:,0], peaks[:,1]].T)
        
        labels = torch.ones(len(scores), 1, dtype=torch.long)

        if self.with_classification:
            
            labels = pred_embed[:, peaks[:,0], peaks[:,1]]

            with torch.no_grad():
                # labels = embed_to_class(labels.T.cuda()).argmax(1).cpu() - 1
                labels = embed_to_class(labels.T.cuda()).cpu().softmax(1)

        # 
        peaks = peaks[:,[1,0]] / img_size[[1,0]]

        if len(peaks) > 0:
            boxes = torch.cat([peaks - 0.5* sizes, peaks + 0.5* sizes], dim=1)
            boxes = boxes.clamp(0, 1)

            valid_indices = nms(boxes, scores, self.nms_threshold)[:200]

            all_detections = []
            for _box, _score, _labels in zip(boxes[valid_indices], scores[valid_indices], labels[valid_indices]):
                cls_scores = list(filter(lambda x: x[1] > 0.05, list(zip(range(len(_labels)), _labels.tolist()))))

                
                for idx, cls_score in sorted(cls_scores, key=lambda x:-x[1]):
                    all_detections += [(_box, _score*cls_score, idx-1)]
                
            boxes, scores, labels = zip(*all_detections)
            
            return torch.stack(boxes), torch.stack(scores), torch.tensor(labels)
        else:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

    def predict_boxes(self, pred):
        results = []
        for i in range(len(pred['centers_gauss'])):
            pred_centers = pred['centers_gauss'][i,0,0].sigmoid()
            pred_sizes = pred['centers_size'][i,:,0].cpu()
            pred_embed = pred['class_embeddings'][i,:,0].cpu()
            results.append(self.compute_bboxes(pred_centers, pred_sizes, pred_embed, self.model.embed_proj, min_score=0.01))
        return results
           

    def evaluate(self, model, dataset_val, bs=4, model_cls=None, n_workers=2, min_score=0.01):

        from torchmetrics.detection import MeanAveragePrecision

        self.model = model
        model_cls_ref = model if model_cls is None else model_cls
        model.eval()
        ap = MeanAveragePrecision(iou_type="bbox")
        losses_val = []

        n_workers = min(os.cpu_count(), n_workers)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=bs, shuffle=False, num_workers=n_workers)

        with torch.no_grad():
                
            for sample_val in tqdm.tqdm(loader_val):
                
                pred = model(sample_val)
                if model_cls is not None:
                    pred2 = model_cls(sample_val)
                    pred['class_embeddings'] = pred2['class_embeddings']

                pred = self.maybe_rename_outputs(pred)
                
                img_size = torch.tensor(pred['centers_gauss'].shape[3:])
                assert img_size[0] == img_size[1]

                for i in range(len(pred['centers_gauss'])):
                    pred_centers = pred['centers_gauss'][i,0,0].sigmoid()
                    pred_sizes = pred['centers_size'][i,:,0].cpu()
                    if self.with_classification:
                        pred_embed = pred['class_embeddings'][i,:,0].cpu()
                        embed_proj = model_cls_ref.embed_proj
                    else:
                        pred_embed, embed_proj = None, None

                    boxes, scores, labels = self.compute_bboxes(pred_centers, pred_sizes, pred_embed, embed_proj, min_score=min_score)
                    boxes = boxes * sample_val['original_img_size'][i][[1,0]].repeat(2)

                    # The ground truth boxes are in the original image coordinates. 
                    # Therefore, we have to transform the predicted bounding boxes
                    gt_boxes, gt_labels = dataset_val.get_bboxes_by_sample_id(sample_val['id'][i])

                    if not self.with_classification:
                        labels = labels.fill_(0)
                        gt_labels = gt_labels.fill_(0)            

                    ap.update(
                        [dict(boxes=boxes, scores=scores, labels=labels)], 
                        [dict(boxes=gt_boxes, labels=gt_labels)]
                    )

                
                
                losses_val += [self.loss_function(pred, sample_val)]

        loss_val = torch.tensor([x[0] for x in losses_val]).mean()
        ap_scores = ap.compute()
        out = dict(loss=loss_val.item(), 
                   ap=ap_scores['map'].item(), 
                   map50=ap_scores['map_50'].item(), 
                   ap_s=ap_scores['map_small'].item(),
                   ap_m=ap_scores['map_medium'].item(),
                   ap_lg=ap_scores['map_large'].item(),
                   #predictions=predictions
        )

        out.update({
            f'loss_{k}': torch.tensor([x[1][k] for x in losses_val]).mean().item()
            for k in losses_val[0][1].keys()
        })

        return out    


    def loss_function(self, outputs, labels, iteration=None): 

        # little hack to enable predictions in a single tensor
        outputs = self.maybe_rename_outputs(outputs)

        pred_size = outputs['centers_gauss'].shape[3:]

        centers_gauss = outputs['centers_gauss'][:,:,0]
        device = centers_gauss.device

        gt_centers_gauss = torch.nn.functional.interpolate(labels['centers_gauss'], pred_size, mode='bilinear')
        gt_centers_size = torch.nn.functional.interpolate(labels['centers_size'], (1,) + pred_size, mode='trilinear')
    
        if self.mode == 'focal':
            loss = focal_loss.sigmoid_focal_loss(
                centers_gauss, 
                gt_centers_gauss.to(device).float(),
                reduction='mean'
            )
        else:
            raise ValueError('invalid mode')

        # size loss
        gt_size = gt_centers_size.to(device)
        valid = gt_size > 0
        loss_l1 = torch.nn.functional.l1_loss(
            gt_size, 
            outputs['centers_size'], 
            reduction='none'
        )
        loss_l1 = loss_l1[valid].mean()

        # classification loss
        
        loss_classification = 0
        if self.with_classification:
            gt_label_map = torch.nn.functional.interpolate(labels['label_map'], pred_size, mode='nearest')[:,0]
            m = gt_label_map != 0
            pred_embed_m = outputs['class_embeddings'][:,:,0].permute(1,0,2,3)[:,m]
            loss_classification = torch.nn.functional.cross_entropy(
                self.model.embed_proj(pred_embed_m.T),
                gt_label_map[m].cuda()
            )

        # loss_complete = loss
        
        if self.loss_term == 'all':
            loss_complete = 50*loss + loss_l1 + 0.2*loss_classification
        elif self.loss_term == 'only_classification':
            loss_complete = loss_classification
        else:
            raise ValueError('invalid loss term')

        return loss_complete, dict(size=float(loss_l1), center=float(loss), classification=float(loss_classification))
