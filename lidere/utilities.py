import torch
import math
import time


def get_sample(dataset, bs=1, shuffle=False, idx=None):
    from torch.utils.data import DataLoader, default_collate

    if idx is None:
        loader = DataLoader(dataset, batch_size=bs, shuffle=shuffle)
        return next(iter(loader))
    else:
        return default_collate([dataset[idx]])

def predict(model, sample):
    with torch.inference_mode():
        return model(sample)


def square_meshgrid(size, device='cpu'):
    return torch.dstack(
        torch.meshgrid(torch.linspace(0, 1, size, device=device), 
                       torch.linspace(0, 1, size, device=device), indexing='xy')
    )

def rect_meshgrid(sizes, device='cpu'):
    return torch.dstack(
        torch.meshgrid(torch.linspace(0, 1, sizes[0], device=device), 
                       torch.linspace(0, 1, sizes[1], device=device), indexing='xy')
    )


def random_base64_string(length):
    import random
    import string
    import base64

    random_bytes = bytes(''.join(random.choices(string.ascii_letters + string.digits, k=length)), 'utf-8')
    return base64.b64encode(random_bytes).decode('utf-8')[:length]



class TimeBlock(object):

    def __init__(self, name):
        self.name = name
        self.duration = None

    def __enter__(self):
        print(f'enter block {self.name}...')
        self.t_start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.duration = time.time() - self.t_start
        print(f'block {self.name} took {self.duration:.1f}s')



def pos_enc_sincos(x, dim=10, flatten=False):
    """ 
    sinosoidal positional encoding 
    important: value range of x must be considered. This configuration works best in [-100, 100].
    """

    old_shape = x.shape
    x = x.flatten()
    # div_vec = 1 / (10000 ** (2 * torch.arange(dim).float() / dim))
    div_vec = torch.exp(torch.arange(0, dim, 2).to(x.device) * (-math.log(10000.0) / dim))

    pe2 = torch.cat([
        torch.sin(x[:,None] * div_vec[None, :]),
        torch.cos(x[:,None] * div_vec[None, :]),
    ], dim=1)

    pe2 = pe2.view(old_shape + (dim,))

    if flatten:
        pe2 = pe2.flatten(-2, -1)
        
    return pe2




def count_parameters(model, only_trainable=False):
    """ Count the number of parameters of a torch model. """
    import numpy as np
    return sum([np.prod(p.size()) for p in model.parameters()
                if (only_trainable and p.requires_grad) or not only_trainable])


def show_types(sample):

    if isinstance(sample, dict):
        x = list(sample.items())
    elif isinstance(sample, (tuple, list)):
        print(f'list of {len(sample)} elements')
        x = list(enumerate(sample[:20]))
    else:
        return

    for k, v in x:
        if isinstance(v, torch.Tensor):
            print(f'{k:<12}{str(v.dtype):<20}{str(v.shape)}')
        else:
            if hasattr(v, '__len__'):
                print(f'{k:<12}{str(type(v).__name__):<20}length: {str(len(v))}')


def tensor_info(t):

    s = f'{str(list(t.shape))} / {t.dtype}\nrange: {t.min():.5f}-{t.max():.5f}\nmean: {t.float().mean():.5f}'
    return s


def show(*imgs, size=1, info=True):
    from matplotlib import pyplot as plt
    import numpy as np

    def process_img(img):

        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)


        if img.ndim == 2:
            if isinstance(img, (torch.FloatTensor)):
                return dict(X=img)
            else:    
                return dict(X=img, cmap=plt.cm.tab20, vmin=0, interpolation='nearest')

        # quick hack for offsets
        if img.shape[0] == 2:
            img = torch.cat([torch.zeros(1, *img.shape[1:]), img]).permute(1,2,0)   

        assert img.ndim == 3, f'wrong image shape: {img.shape}'

        if img.shape[0] == 3:
            img = img.permute(1,2,0)


        return dict(X=img)

    _ , ax = plt.subplots(1, len(imgs), figsize=(len(imgs)*4*size, 5*size))

    
    if len(imgs) == 1:
        ax.axis('off')
        ax.imshow(**process_img(imgs[0]))
        if info:
            ax.text(0,0,tensor_info(imgs[0]), va='top')
    else:
        [a.axis('off') for a in ax.flatten()]
        import matplotlib.patheffects as path_effects
        for i, img in enumerate(imgs):
            ax[i].imshow(**process_img(img))
            if info:
                t = ax[i].text(0,0,tensor_info(img), c='white', va='top')
                t.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])


def show_pca(*feats, axes=None):
    """ assumes feats to have format [feats,h,w] """
    from sklearn.decomposition import PCA
    from matplotlib import pyplot as plt

    if axes is None:
        if len(feats) == 1:
            ax = [plt.subplots(1, len(feats), figsize=(len(feats)*4, 5))[1]]
        else:
            _ , ax = plt.subplots(1, len(feats), figsize=(len(feats)*4, 5))
    else:
        ax = axes

    for i, feat in enumerate(feats):
        assert feat.ndim == 3
        img_dim = feat.shape[-2:]
        n = 4
        pca = PCA(n)
        dim3 = pca.fit_transform(feat.flatten(1).cpu().numpy().T).reshape(*img_dim,n)
        dim3 = dim3[...,:3]
        print('variance explained', pca.explained_variance_ratio_)
        print('sum', sum(pca.explained_variance_ratio_))
        dim3 = torch.from_numpy(dim3).sigmoid()
        ax[i].imshow(dim3)


def summarize_vars(
    vars_dict,
    *,
    max_str_len=80,
    max_list_len=10,
    max_dict_keys=10,
    max_dict_depth=2,
    max_preview_elems=10,      # for 1D
    max_preview_rows =5,        # for 2D
    small_dim =8,               # second dimension threshold    
    indent_step=2,
    ignore_private=True,
    ignore_modules=True,
    ignore_callables=True,
    ignore_ipython=True,
    only_changed=False,
):
    """
    Print compact, relevant context about variables (notebook-friendly).
    """

    import sys
    import numpy as np
    from torch import nn
    import inspect

    # -------------------- filtering helpers --------------------

    IPYTHON_PREFIXES = (
        "In", "Out", "exit", "quit", "get_ipython",
        "display", "HTML", "JSON", "Math"
    )

    # store last seen object ids (for only_changed)
    if not hasattr(summarize_vars, "_last_seen"):
        summarize_vars._last_seen = {}

    def get_signature(obj):
        try:
            sig = inspect.signature(obj)
            return str(sig)
        except (ValueError, TypeError):
            return "(...)"

    def is_relevant(name, value):
        if ignore_private and name.startswith("_"):
            return False

        if ignore_ipython and name.startswith(IPYTHON_PREFIXES):
            return False

        if ignore_modules and isinstance(value, type(sys)):
            return False

        if ignore_callables and callable(value):
            return False

        if inspect.isclass(value):
            return False            

        if only_changed:
            prev_id = summarize_vars._last_seen.get(name)
            if prev_id == id(value):
                return False
            summarize_vars._last_seen[name] = id(value)

        return True

    # -------------------- summarization helpers --------------------

    def truncate(s, max_len):
        s = str(s)
        return s if len(s) <= max_len else s[: max_len - 3] + "..."

    def summarize_dict(d, depth, indent=0):
        if depth <= 0:
            return "{...}"

        items = list(d.items())
        shown = items[:max_dict_keys]
        more = len(items) - len(shown)

        indent_str = " " * indent
        child_indent = indent + indent_step
        child_indent_str = " " * child_indent

        lines = ["{"]

        for k, v in shown:
            try:
                val_summary = summarize_value(v, depth=depth - 1, indent=child_indent)
            except Exception:
                val_summary = "<?>"

            lines.append(f"{child_indent_str}{k}: {val_summary}")

        if more > 0:
            lines.append(f"{child_indent_str}... (+{more} keys)")

        lines.append(f"{indent_str}}}")

        return "\n".join(lines)

    def summarize_value(v, depth, indent=0):
        # None
        if v is None:
            return "None"

        # Basic scalars
        if isinstance(v, (int, float, bool)):
            return f"{v} ({type(v).__name__})"

        # String
        if isinstance(v, str):
            return f'"{truncate(v, max_str_len)}" (len={len(v)})'

        # List / Tuple
        if isinstance(v, (list, tuple)):
            preview = ", ".join(truncate(x, 20) for x in v[:max_list_len])
            suffix = ", ..." if len(v) > max_list_len else ""
            return f"{type(v).__name__}(len={len(v)}) [{preview}{suffix}]"

        # NumPy array
        if isinstance(v, np.ndarray):
            ndim = v.ndim

            # 1D → always show head
            if ndim == 1:
                head = v[:max_preview_elems].tolist()
                suffix = "..." if v.shape[0] > max_preview_elems else ""
                return f"np.ndarray(len={v.shape[0]}, dtype={v.dtype}) {head}{suffix}"

            # 2D with small second dim → show first rows
            if ndim == 2 and v.shape[1] <= small_dim:
                rows = v[:max_preview_rows].tolist()
                suffix = "..." if v.shape[0] > max_preview_rows else ""
                return (
                    f"np.ndarray(shape={v.shape}, dtype={v.dtype}) "
                    f"{rows}{suffix}"
                )

            # fallback summary
            info = f"np.ndarray shape={v.shape}, dtype={v.dtype}"
            if v.size and np.issubdtype(v.dtype, np.number):
                info += f", min={v.min()}, max={v.max()}"
            return info

        if isinstance(v, torch.Tensor):
            t = v.detach().cpu()
            ndim = t.ndim

            # 1D → always show head
            if ndim == 1:
                head = t[:max_preview_elems].tolist()
                suffix = "..." if t.shape[0] > max_preview_elems else ""
                return (
                    f"torch.Tensor(len={t.shape[0]}, dtype={t.dtype}) "
                    f"{head}{suffix}"
                )

            # 2D with small second dim → show first rows
            if ndim == 2 and t.shape[1] <= small_dim:
                rows = t[:max_preview_rows].tolist()
                suffix = "..." if t.shape[0] > max_preview_rows else ""
                return (
                    f"torch.Tensor(shape={tuple(t.shape)}, dtype={t.dtype}) "
                    f"{rows}{suffix}"
                )

            # fallback summary
            info = (
                f"torch.Tensor shape={tuple(t.shape)}, "
                f"dtype={t.dtype}, device={t.device}"
            )

            if t.numel() and t.dtype != torch.bool:
                info += f", min={t.min().item()}, max={t.max().item()}"

            return info

        # ✅ Torch nn.Module (callable model)
        if isinstance(v, nn.Module):
            info = f"nn.Module({v.__class__.__name__})"
            n_params = sum(p.numel() for p in v.parameters())
            info += f", params={n_params:,}"
            return info

        # Dictionary (must come BEFORE generic container)
        if isinstance(v, dict):
            return f"dict(len={len(v)}) {summarize_dict(v, depth, indent)}"

        # ✅ Generic container-like object (__len__ + __getitem__)
        has_len = hasattr(v, "__len__")
        has_getitem = hasattr(v, "__getitem__")

        if has_len and has_getitem:
            try:
                length = len(v)
                return f"{type(v).__name__}(len={length})"
            except Exception:
                pass

        # ✅ Function / method with signature
        if inspect.isfunction(v) or inspect.ismethod(v):
            sig = get_signature(v)
            name = v.__name__
            return f"function {name}{sig}"

        # Callable but not function/class (e.g. functor)
        if callable(v):
            return f"{type(v).__name__} (callable)"

        # Fallback
        return 'instance of ' + type(v).__name__

    # -------------------- main --------------------

    for name, value in vars_dict.items():
        if not is_relevant(name, value):
            continue

        try:
            summary = summarize_value(value, depth=max_dict_depth, indent=0)
        except Exception as e:
            summary = f"<error: {e}>"

        print(f"{name}: {summary}")