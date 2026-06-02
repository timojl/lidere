import torch


class OnlyLossEvaluation(object):

    def evaluate(self, model, dataset_val, bs=8, n_workers=1):

        model.eval()
        # metric_jaccard = JaccardIndex("binary", threshold=0.2)
        losses_val = []

        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=bs, shuffle=False, num_workers=n_workers)

        with torch.no_grad():
                
            for sample_val in loader_val:

                outputs = model(sample_val)
                # metric_jaccard.update(preds.squeeze(1).flatten(), sample_val[self.key_name].flatten())
                losses_val += [self.loss_function(outputs, sample_val)[0]]

        out = dict(loss = torch.tensor(losses_val).mean().item())

        return out    
