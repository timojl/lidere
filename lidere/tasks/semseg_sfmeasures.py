import torch
from lidere.tasks.semseg import SemanticSegmentationTask

class SemanticSegmentationTaskSFMeasures(SemanticSegmentationTask):
    """ This class is required for the evaluating on the camouflaged dataset. """

    def evaluate(self, model, dataset_val, bs=8, n_workers=1):
        import sys

        from lidere.third_party import conv_lora_metrics

        sys.path.append('lidere/third_party/')
        smeasure = conv_lora_metrics.Smeasure()
        emeasure = conv_lora_metrics.Emeasure()
        fmeasure = conv_lora_metrics.WeightedFmeasure()

        from torchmetrics.classification import MulticlassJaccardIndex
        
        metric_jaccard = MulticlassJaccardIndex(num_classes=self.n_classes, ignore_index=self.ignore_index, average=None if self.with_classes else 'macro')
        losses_val = []

        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=bs, shuffle=False, num_workers=n_workers)

        with torch.no_grad():
                
            for sample_val in loader_val:

                outputs = model(sample_val)

                preds = torch.nn.functional.interpolate(
                    outputs[self.key_name].cpu(), 
                    sample_val[self.key_name].shape[-3:],
                    mode='trilinear',
                )

                for i in range(preds.shape[0]):
                    p = preds[i].argmax(0)[0].mul(255).numpy()
                    g = sample_val[self.key_name][i,0].mul(255).numpy()
                    smeasure.step(p,g)
                    emeasure.step(p,g)
                    fmeasure.step(p,g)
                
                metric_jaccard.update(preds.argmax(1).flatten(), sample_val[self.key_name].flatten())
                losses_val += [self.loss_function_output_scaled(outputs, sample_val)]

        scores_jaccard = metric_jaccard.compute()
        out = dict(
            loss = torch.tensor(losses_val).mean().item(),
            s=smeasure.get_results()['sm'],
            e=emeasure.get_results()['em']['adp'],
            f=fmeasure.get_results()['wfm']
        )

        if self.with_classes:
            out.update(iou=scores_jaccard.mean().item(), iou_cls=scores_jaccard.tolist())
        else:
            if not torch.isnan(scores_jaccard):
                out.update(iou=scores_jaccard.item())

        return out
