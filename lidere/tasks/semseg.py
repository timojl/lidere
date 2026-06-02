import torch


class SemanticSegmentationTask(object):

    def __init__(self, ignore_index=None, n_classes=125, with_classes=False, dice=False, binary=False, key_name='sem'):
        self.key_name = key_name
        self.n_classes = n_classes
        self.dice = dice
        self.with_classes = with_classes
        self.ignore_index = ignore_index
        self.binary = binary

    def evaluate(self, model, dataset_val, bs=8, n_workers=1, max_iterations=None):

        print(f'evaluate on {len(dataset_val)} samples')

        from torchmetrics.classification import MulticlassJaccardIndex, Dice, Accuracy, BinaryJaccardIndex
        
        if self.binary:
            metric_jaccard = BinaryJaccardIndex()
        else:
            metric_jaccard = MulticlassJaccardIndex(num_classes=self.n_classes, ignore_index=self.ignore_index, average=None if self.with_classes else 'macro')
        
        if self.dice:
            ignore_index = 0 if self.binary else self.ignore_index
            metric_dice = Dice(num_classes=self.n_classes, ignore_index=ignore_index)
        
        metric_acc = Accuracy(num_classes=self.n_classes, task="multiclass", ignore_index=self.ignore_index)
        losses_val = []

        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=bs, shuffle=False, num_workers=n_workers)

        with torch.no_grad():
                
            iter_count = 0

            for sample_val in loader_val:

                outputs = model(sample_val)

                preds = torch.nn.functional.interpolate(
                    outputs[self.key_name].cpu(), 
                    sample_val[self.key_name].shape[-3:],
                    mode='trilinear',
                    align_corners=False,
                )
                
                metric_jaccard.update(preds.argmax(1).flatten(), sample_val[self.key_name].flatten())
                if self.dice:
                    metric_dice.update(preds.argmax(1).flatten(), sample_val[self.key_name].flatten())
                metric_acc.update(preds.argmax(1).flatten(), sample_val[self.key_name].flatten())
                losses_val += [self.loss_function_output_scaled(outputs, sample_val)]

                iter_count += 1
                if max_iterations is not None and iter_count > max_iterations:
                    break

        scores_jaccard = metric_jaccard.compute()
        out = dict(loss = torch.tensor(losses_val).mean().item())

        if self.with_classes:
            out.update(iou=scores_jaccard.mean().item(), iou_cls=scores_jaccard.tolist())
        else:
            if not torch.isnan(scores_jaccard):
                out.update(iou=scores_jaccard.item())

        if self.dice:
            out.update(dice=metric_dice.compute().item())

        out.update(acc=metric_acc.compute().item())

        return out

    def loss_function_output_scaled(self, outputs, labels, iteration=None):
        
        labels_sem = labels[self.key_name] # .permute(0, 4, 1, 2, 3)

        out_scaled = torch.nn.functional.interpolate(outputs[self.key_name], labels_sem.shape[1:4], mode='trilinear')
        
        out_scaled = out_scaled.permute(1,0,2,3,4).flatten(1)
        device = out_scaled.device

        return torch.nn.functional.cross_entropy(
            out_scaled.T, 
            labels_sem.flatten().to(device).long(),
            ignore_index=-100 if self.ignore_index is None else self.ignore_index
        )
        

    def loss_function(self, outputs, labels, iteration=None):

        labels_sem = labels[self.key_name] # .permute(0, 4, 1, 2, 3)


        device = outputs[self.key_name].device
        labels_scaled = torch.nn.functional.interpolate(
            labels_sem[None].to(device), 
            outputs[self.key_name].shape[2:], 
            mode='nearest'
        )
        
        # This can be a bottleneck for (spatially) large tensors.
        loss = torch.nn.functional.cross_entropy(
            outputs[self.key_name].permute(1,0,2,3,4).flatten(1).T, 
            labels_scaled.flatten().to(device).long(),
            ignore_index=-100 if self.ignore_index is None else self.ignore_index
        )

        return loss, dict()


    def plot(self, sample, out=None):

        from matplotlib import pyplot as plt

        n_cols = 2 if out is None else 3
        fig, ax = plt.subplots(2,n_cols, figsize=(2*n_cols, 3))
        
        for i in range(2):

            ax[i, 0].imshow(sample['image'][i,:,0].permute(1,2,0))

            ax[i, 1].imshow(sample[self.key_name][i,0],
                cmap=plt.cm.rainbow, interpolation='nearest', vmin=0, vmax=120)

            if out is not None:
                ax[i, 2].imshow(out[self.key_name][i,:,0].detach().cpu().argmax(0),
                    cmap=plt.cm.rainbow, interpolation='nearest', vmin=0, vmax=120)

            [a.axis('off') for a in ax.flatten()]

        fig.tight_layout()
        plt.close(fig)
        return fig
    