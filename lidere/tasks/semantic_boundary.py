import torch
import numpy as np



def predict_contour_multi(model, x, S=100, k=0, tol=0.07):
    def f(x):
        with torch.no_grad():
            out = model(dict(image=x[None,:,None].cuda()))['contour'][0,0,0].sigmoid()

        out = torch.nn.functional.interpolate(out[None,None], x.shape[1:], mode='bilinear')[0,0].cpu()
        return out
    
    pred = process_tensor(x, S, f, k=k)

    from skimage import morphology, filters
    from skimage import io

    pred[pred<0.01] = 0

    if tol is None:
        return pred
    else:
        return torch.from_numpy(nms_edges(pred.numpy(), tol=tol))



def predict_contour(model, x, tol=0.05):
    assert x.ndim == 3
    with torch.no_grad():
        pred = model(dict(image=x[None,:,None]))
    pred = torch.nn.functional.interpolate(pred['contour'][0].sigmoid().detach(), x.shape[1:], mode='bilinear')[0,0]    

    if tol is None:
        return pred
    else:    
        return torch.from_numpy(nms_edges(pred.cpu().numpy(), tol=tol))    


def process_tensor(x, S, f, k=0):
    """ 
    divide image into square cells of size S and run f on each
    """

    C, H, W = x.shape
    pad_h, pad_w = (-H) % S, (-W) % S
    x_pad = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))

    step = S - k
    patches = x_pad.unfold(1, S, step).unfold(2, S, step)  # (C, nH, nW, S, S)
    nH, nW = patches.shape[1:3]

    out = torch.zeros_like(x_pad[0])
    count = torch.zeros_like(out)

    for i in range(nH):
        for j in range(nW):
            patch = patches[:, i, j]               # shape (C, S, S)
            processed = f(patch)                   # expects output shape (S, S)
            y0 = i * step
            x0 = j * step
            out[y0:y0+S, x0:x0+S] += processed
            count[y0:y0+S, x0:x0+S] += 1

    return (out / count)[:H, :W]



def nms_edges(edge_prob, tol=0.05):
    """
    Non-maximum suppression for edge probability maps.
    Vectorized version with interpolation along gradient directions.
    """

    from scipy import ndimage

    # Compute gradients
    gx = ndimage.sobel(edge_prob, axis=1, mode='reflect')
    gy = ndimage.sobel(edge_prob, axis=0, mode='reflect')
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx)  # radians [-pi, pi]

    # Prepare for interpolation
    rows, cols = edge_prob.shape
    sin_a = np.sin(ang)
    cos_a = np.cos(ang)

    # Coordinates in the direction of the gradient
    r0 = np.arange(rows)[:, None]
    c0 = np.arange(cols)[None, :]

    # Shift coordinates forward and backward along gradient
    r1 = r0 + sin_a
    c1 = c0 + cos_a
    r2 = r0 - sin_a
    c2 = c0 - cos_a

    # Interpolate values from the edge_prob map
    def interp(r, c):
        return ndimage.map_coordinates(edge_prob, [r.flatten(), c.flatten()], order=1).reshape(rows, cols)

    val1 = interp(r1, c1)
    val2 = interp(r2, c2)

    # Keep only local maxima along gradient direction
    keep = (edge_prob >= val1 - tol) & (edge_prob >= val2 - tol)
    nms = np.where(keep, edge_prob, 0)

    return nms


def eval_boundaries(model, dataset, n_iterations=None, S=200, multi=True, thin=True, tol=0.07):
    import random, string
    import os
    import shutil
    from PIL import Image
    tmp_dir = 'boundary-output/' + ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    if os.path.exists(tmp_dir + '-eval'):
        shutil.rmtree(tmp_dir + '-eval')    

    os.makedirs(tmp_dir, exist_ok=True)

    n_iterations = len(dataset) if n_iterations is None else n_iterations

    for i in range(n_iterations):
        if multi:
            pred = predict_contour_multi(model, dataset[i]['image'][:,0], S=S, k=int(0.2*S), tol=tol)
        else:
            pred = predict_contour(model, dataset[i]['image'][:,0], tol=tol)
        idx = os.path.basename(dataset.files[i]).split('.')[0]
        shutil.copy(dataset.files[i], os.path.join(tmp_dir, f'{idx}.mat'))
        Image.fromarray(pred.mul(255).byte().cpu().numpy()).save(f'{tmp_dir}/{idx}.png')

    import sys
    sys.path.append('third_party/edge_eval_python/')

    print(S, multi, thin, tol)

    from impl.edges_eval_dir import edges_eval_dir
    res = edges_eval_dir(tmp_dir, tmp_dir, workers=8, thin=thin)
    shutil.rmtree(tmp_dir)
    print(res)
    return res


class BoundaryPredictionTask(object):

    def __init__(self, threshold=0.5, key_name='boundaries', no_jaccard=False, no_fscores=False):
        self.label_types = ['boundaries']
        self.key_name = key_name
        self.threshold = threshold
        self.no_jaccard = no_jaccard
        self.no_fscores = no_fscores

    def evaluate(self, model, dataset_val, bs=8, n_workers=1):

        from torchmetrics.classification import JaccardIndex
        
        model.eval()

        metric_jaccard = JaccardIndex("binary", threshold=self.threshold) if not self.no_jaccard else None
        losses_val = []

        thresholds = torch.linspace(0.025, 0.5, 20)
        i_acc, u_acc = [], []

        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=bs, shuffle=False, num_workers=n_workers)

        with torch.no_grad():
                
            for sample_val in loader_val:

                outputs = model(sample_val)

                # scale model output to label size
                preds = torch.nn.functional.interpolate(
                    outputs[self.key_name].cpu(), 
                    sample_val[self.key_name].shape[-3:],
                    mode='trilinear'
                )

                preds = preds.sigmoid()
                
                if not self.no_jaccard:
                    metric_jaccard.update(preds.squeeze(1).flatten(), sample_val[self.key_name].flatten())

                i_acc += [torch.stack([torch.logical_and(preds.flatten(0,2).flatten(1) > t, sample_val[self.key_name].flatten(0,1).flatten(1)).sum(-1) 
                                      for t in thresholds]).T]
                u_acc += [torch.stack([torch.logical_or(preds.flatten(0,2).flatten(1) > t, sample_val[self.key_name].flatten(0,1).flatten(1)).sum(-1)
                                      for t in thresholds]).T]

                losses_val += [self.loss_function(outputs, sample_val)[0]]

        dataset_val.avg_contour = False
        dataset_val.aug = False
        dataset_val.transform = None

        i_acc, u_acc = torch.cat(i_acc), torch.cat(u_acc)
        print('IoUs', (i_acc.sum(1) / (u_acc.sum(1)+1)))

        out = dict(loss = torch.tensor(losses_val).mean().item())

        out.update(iou_opt=(i_acc.sum(1) / (u_acc.sum(1)+1)).max().item())

        if not self.no_fscores:
            res = eval_boundaries(model, dataset_val, n_iterations=30)
            out.update(dict(
                # p=res['ois_p'],   TODO: figure our why this is not provided
                # r=res['ois_r'], 
                ois=res['ois_f'], ods=res['ods_f']))

        score_jaccard = metric_jaccard.compute() if not self.no_jaccard else None
        if score_jaccard is not None and not torch.isnan(score_jaccard):
            out.update(iou=score_jaccard.item())

        return out

    def loss_function(self, outputs, labels, iteration=None): 

        out_scaled = torch.nn.functional.interpolate(
            outputs[self.key_name], 
            labels[self.key_name].shape[1:4],
            mode='trilinear'
        )

        out_scaled = out_scaled[:, 0]  # there is only a single map
        device = out_scaled.device

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out_scaled.flatten(), 
            labels[self.key_name].flatten().to(device).float()
        )
        return loss, dict()

