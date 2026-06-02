import torch
import math
from torch import nn

def pos_enc_sincos(x, dim=10, flatten=False, temperature=10000.0):
    """ 
    sinosoidal positional encoding 
    important: value range of x must be considered. Works best for integers, i.e. torch.arange(N).
    """

    old_shape = x.shape
    x = x.flatten()
    # div_vec = 1 / (10000 ** (2 * torch.arange(dim).float() / dim))
    div_vec = torch.exp(torch.arange(0, dim, 2).to(x.device) * (-math.log(temperature) / dim))

    pe2 = torch.cat([
        torch.sin(x[:,None] * div_vec[None, :]),
        torch.cos(x[:,None] * div_vec[None, :]),
    ], dim=1)

    pe2 = pe2.view(old_shape + (dim,))

    if flatten:
        pe2 = pe2.flatten(-2, -1)
        
    return pe2

def rect_meshgrid(sizes, device='cpu'):
    return torch.dstack(
        torch.meshgrid(torch.linspace(0, 1, sizes[0], device=device), 
                       torch.linspace(0, 1, sizes[1], device=device), indexing='xy')
    )


def init_queries_sincos2(base_size, pe_dim, dim, fac=1):
    pos = torch.dstack(torch.meshgrid(torch.arange(base_size[1]), torch.arange(base_size[0]), indexing='xy'))
    return torch.cat([
        pos_enc_sincos(pos*fac, dim=pe_dim // 2, flatten=True),
        torch.randn(base_size[0],base_size[1], dim)
    ], dim=2).flatten(0, 1)


def bilinear_matrix_dense(H, W, H_new, W_new, align_corners=True):
    """ Return dense bilinear interpolation matrix B. """
    import numpy as np
    # Output pixel coordinates
    jj, ii = np.meshgrid(np.arange(H_new), np.arange(W_new), indexing='ij')

    if align_corners:
        scale_y = (H - 1) / (H_new - 1) if H_new > 1 else 0
        scale_x = (W - 1) / (W_new - 1) if W_new > 1 else 0
        y = jj * scale_y
        x = ii * scale_x
    else:
        scale_y = H / H_new
        scale_x = W / W_new
        y = (jj + 0.5) * scale_y - 0.5
        x = (ii + 0.5) * scale_x - 0.5
        y = np.clip(y, 0, H - 1)
        x = np.clip(x, 0, W - 1)

    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)

    dy = y - y0
    dx = x - x0

    w00 = (1 - dy) * (1 - dx)
    w01 = (1 - dy) * dx
    w10 = dy * (1 - dx)
    w11 = dy * dx

    B = np.zeros((H_new * W_new, H * W), dtype=float)

    out_idx = np.arange(H_new * W_new)
    in00 = y0 * W + x0
    in01 = y0 * W + x1
    in10 = y1 * W + x0
    in11 = y1 * W + x1

    B[out_idx, in00.ravel()] = w00.ravel()
    B[out_idx, in01.ravel()] = w01.ravel()
    B[out_idx, in10.ravel()] = w10.ravel()
    B[out_idx, in11.ravel()] = w11.ravel()

    return torch.from_numpy(B).float()



class SimpleBase(nn.Module):

    def precompute_features(self, loader, dtype=None, amp=False, device=None):
        self.backbone.precompute(loader, dtype=dtype, amp=amp)

    def get_features(self, sample):
        device = next(self.parameters()).device

        bs = sample['image'].shape[0]

        x = torch.nn.functional.interpolate(sample['image'][:,:,0].to(device), self.backbone.img_size, mode='bicubic', antialias=True)[:,:,None]        
        return self.backbone.features_compute_or_cached(x, sample['id'] if 'id' in sample else None, device=device)


class LayerFeatureMLP(nn.Module):
    def __init__(self, p, hidden_dims=(), out_dim=48, on_feats=False, is_first=False, size=100):
        super().__init__()
        # size = p.base_size if not on_feats else p.feat_size
        layers = []
        self.size = size

        hidden_dims = [p.inp_dim] + list(hidden_dims) + [out_dim]
        for i in range(len(hidden_dims)-1):
            layers += [nn.Conv2d(hidden_dims[i], hidden_dims[i+1], kernel_size=1)]
            if i != len(hidden_dims)-2:
                layers += [nn.ReLU()]

        self.proj = nn.Sequential(*layers)

    def forward(self, x, feats):
        bs = x.shape[0]
        assert feats.shape[2] == 1
        feats = self.proj(feats[:,:,0])

        if self.size is not None:
            feats = nn.functional.interpolate(feats, (self.size, self.size), mode='bilinear')    

        return x, feats[:,:,None]





class ImplicitNet(nn.Module):

    def __init__(self, exp=True, n_hidden=1, dim=32, inputs=4, n_heads=8, only_rel=False, rel_fac=3):
        super().__init__()
        
        self.exp = exp
        self.n_inputs = inputs
        self.only_rel = only_rel

        class Sine(nn.Module):
            def forward(self, x):
                return torch.sin(x)
        
        layers = [nn.Linear(inputs, dim), Sine()]
        layers += [nn.Linear(dim, dim), Sine()]*n_hidden
        layers += [nn.Linear(dim, n_heads)]
        self.net = nn.Sequential(*layers)
        self.skip = nn.Linear(inputs, n_heads, bias=False)
        
        if rel_fac is not None:
            #skip_init = 0.0*torch.ones(8,4)
            skip_init = 0.01*torch.rand(n_heads, self.n_inputs)
            skip_init[:,[2,3]] = -rel_fac
            self.skip.weight = nn.Parameter(skip_init, requires_grad=True)

    def forward(self, x):
        x = x[:,:,:self.n_inputs]

        if self.only_rel:
            x = x * torch.tensor([0,0,1,1] + [0]*(self.n_inputs-4), device=x.device)

        if self.exp:
            return self.skip(x).exp() + self.net(x).exp()
        else:
            return self.skip(x) + self.net(x)



class LayerImplicitAtt(nn.Module):

    def __init__(self, p, is_first=True, n_heads=8, vdim_fac=2, dropout=0, dim_out=None, content_dim=0, 
                 implicit_net=None, content_pe=True, content_self_att=False, inp_dim=None, ff_net='mlp'):
        super().__init__()

        self.implicit_net = implicit_net

        dim = p.dim // n_heads
        # self.feat_token_shape = p.feat_size
        self.base_size = p.base_size
        self.feat_size = p.feat_size
        inp_dim = p.inp_dim if inp_dim is None else inp_dim

        self.att_fac = None
        if implicit_net == 'fixed_bilinear':
            self.bil_interpolation = nn.Parameter(bilinear_matrix_dense(*p.base_size, *p.feat_size), requires_grad=False)
            self.att_fac = nn.Parameter(11*torch.ones(1), requires_grad=True)

        if self.implicit_net is None:
            self.att_fac = nn.Parameter(11*torch.ones(1), requires_grad=True)

        self.n_heads = n_heads
        self.is_first = is_first
        self.content_dim = content_dim
        self.content_self_att = content_self_att

        self.proj_v = nn.Linear(inp_dim, n_heads*vdim_fac*dim)

        content_pe_dim = 16 if content_pe else 0
        self.proj_k = nn.Linear(inp_dim + content_pe_dim, n_heads*content_dim)
        self.feats_pe = nn.Parameter(init_queries_sincos2(self.feat_size, 16, 0), requires_grad=False) if content_pe else None

        if content_self_att:
            self.proj_q = nn.Linear(inp_dim + content_pe_dim, n_heads*content_dim)
            self.q = None
        else:
            self.q = nn.Parameter(init_queries_sincos2(p.base_size, 16, 0, fac=10), requires_grad=False)
            self.proj_q = nn.Linear(16 if is_first else p.dim, n_heads*content_dim)

        m1 = rect_meshgrid((self.feat_size[1], self.feat_size[0])).flatten(0,1)
        m2 = rect_meshgrid((self.base_size[1], self.base_size[0])).flatten(0,1)
        inp = (m1[:,None] - m2[None, :])
        m1_ = m1[:,None].repeat(1,m2.shape[0],1)
        m2_ = m2[None,:].repeat(m1.shape[0],1,1)
        self.inp = nn.Parameter(torch.cat([inp, inp.pow(2),m1_,m2_], dim=2), requires_grad=False)  # implicit input remains the same
        
        self.att_prior = None

        dim_out = dim_out if dim_out is not None else n_heads*dim

        self.norm2 = nn.LayerNorm(dim_out, eps=1e-5, bias=True)
        if ff_net == 'mlp':
            dim_feedforward = vdim_fac*2*n_heads*dim
            self.linear1 = nn.Linear(vdim_fac*n_heads*dim, dim_feedforward, bias=True)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, dim_out, bias=False)  # because of layer norm
        elif ff_net == 'linear':
             self.linear1 = nn.Linear(vdim_fac*n_heads*dim, dim_out, bias=True)
             self.linear2 = None
        else:
            raise ValueError()

    def forward(self, x, feats):

        bs = feats.shape[0]
        device = feats.device
        feats_ = feats.flatten(2).permute(0,2,1)

        x_inp = x

        if self.feats_pe is not None:
            feats_pe = self.feats_pe[None].repeat(bs, 1, 1, 1, 1).flatten(1,3)
            feats_with_pe = torch.cat([feats_, feats_pe], dim=2)
        else:
            feats_with_pe = feats_

        feats_v = self.proj_v(feats_)
        feats_v = feats_v.view(bs, feats_v.shape[1], self.n_heads, feats_v.shape[2] // self.n_heads)

        if self.is_first:
            if self.q is None:  # use content self-attention in first layer
                feats_with_pe_ = feats_with_pe.unflatten(1, self.feat_size).permute(0,3,1,2)
                feats_with_pe_ = torch.nn.functional.interpolate(feats_with_pe_, self.base_size).permute(0,2,3,1)
                q = self.proj_q(feats_with_pe_.flatten(1,2))
            else:
                q = self.proj_q(self.q[None].repeat(bs, 1, 1))
        else:
            q = self.proj_q(x)

        k = self.proj_k(feats_with_pe)

        k = k.view(*k.shape[:2], self.n_heads, k.shape[2] // self.n_heads)  # split into heads
        q = q.view(*q.shape[:2], self.n_heads, q.shape[2] // self.n_heads)  # split into heads

        content_att = torch.einsum('bnhd,bmhd->bhmn', q, k)  # / q.shape[0]

        if self.implicit_net == 'fixed_bilinear':
            n_heads = content_att.shape[1]
            att = (self.bil_interpolation)[None].repeat(bs, n_heads, 1, 1)

        elif self.implicit_net is not None:
            m = self.implicit_net(self.inp).permute(2,0,1) if self.att_prior is None else self.att_prior.permute(2,0,1)
            
            att = (m[None].repeat(bs, 1, 1, 1) + content_att).softmax(2)
        else:
            m = ((self.inp[:,:,2]==0) & (self.inp[:,:,3]==0)).float()
            att = (self.att_fac*m + content_att).softmax(2)

        if self.att_fac is not None and torch.rand(1).item() < 0.03:
            print('att fac', self.att_fac)

        x = torch.einsum('bnhd,bhnm->bmhd', feats_v, att)
        x = x.flatten(2)

        if self.linear2 is not None:
            x = self.norm2(self.linear2(self.dropout(nn.functional.relu(self.linear1(x)))))
        else:
            x = self.norm2(self.linear1(x))

        return x_inp + x, feats


class InterpolateFac(nn.Module):

    def __init__(self, fac, bilinear=False):
        super().__init__()
        self.fac = fac
        self.interp_args = dict()
        if bilinear:
            self.interp_args = dict(mode='bilinear', antialias=True)

    def forward(self, x):
        
        size = x.shape[3:]
        assert x.shape[2] == 1

        size_new = [s*self.fac for s in size]
        return torch.nn.functional.interpolate(x[:,:,0], size_new, **self.interp_args)[:,:,None]


class UpsampleCNN(nn.Module):

    def __init__(self, p, up, n_classes, dim_interm=None, out_bias=None, out_div=1):

        super().__init__()

        dim = p.dim

        dim_interm = max(dim // 4, n_classes // 2) if dim_interm is None else dim_interm
        out_mlp = []
        for i, s in enumerate(up):
            dim_in = dim if i==0 else dim_interm
            out_mlp += [
                nn.Conv3d(dim_in, dim_interm, kernel_size=(1,3,3), padding=(0,1,1)), nn.ReLU(), 
                nn.ConvTranspose3d(dim_interm, dim_interm, kernel_size=(1, s, s), stride=(1, s, s)), nn.ReLU()
            ]
                
        out_mlp += [nn.Conv3d(dim_interm, n_classes, kernel_size=1)]
        self.out_mlp = nn.Sequential(*out_mlp)
        self.skip_mlp = nn.Sequential(
            InterpolateFac(torch.prod(torch.tensor(up)).item(), bilinear=True),
            nn.Conv3d(dim, n_classes, kernel_size=1),
        )

        self.out_mlp[-1].weight = nn.Parameter(self.out_mlp[-1].weight/out_div)
        self.skip_mlp[1].weight = nn.Parameter(self.skip_mlp[1].weight/out_div)
        self.skip_mlp[1].bias = nn.Parameter(self.skip_mlp[1].bias/out_div)
        self.out_mlp[-1].bias = nn.Parameter(self.out_mlp[-1].bias/out_div)

        if out_bias is not None:
            self.skip_mlp[1].bias = nn.Parameter(self.skip_mlp[1].bias+out_bias)
            self.out_mlp[-1].bias = nn.Parameter(self.out_mlp[-1].bias+out_bias)

    def forward(self, out):
        out = self.skip_mlp(out) + self.out_mlp(out)
        return out



class LiDeRe(SimpleBase):

    def __init__(self, dim, base_size, backbone, key_name='mask', out_bias=0, n_classes=100, no_freeze=False):
        super().__init__()
        
        self.backbone = backbone

        if not no_freeze:
            self.backbone.freeze()

        self.key_name = key_name

        self.dim = dim

        self.inp_dim = self.backbone.feature_dim()
        self.base_size = (base_size, base_size) if isinstance(base_size, int) else base_size
        self.feat_size = self.backbone.base_size(self.backbone.img_size)

        self.layers = nn.ModuleList([
            LayerFeatureMLP(self, hidden_dims=(), out_dim=dim, size=None),
            LayerImplicitAtt(self, is_first=False, vdim_fac=2, content_dim=2, inp_dim=dim, implicit_net=ImplicitNet())
        ])

        self.mlp = UpsampleCNN(self, up=[2, 2], n_classes=n_classes, out_bias=out_bias)

        self.precomputed_features = None         
        self.no_early_device_copy = False
        self.index_order = None

    def forward(self, sample):

        # bs = sample['image'].shape[0]
        feats = self.get_features(sample)
        return self.forward_feats(feats)
        
    def forward_feats(self, feats):

        bs = feats.shape[0]

        x = torch.zeros(bs, self.base_size[0] * self.base_size[1], self.dim).to(feats.device)
        for layer in self.layers:
            x, feats = layer(x, feats)

        x = x.permute(0, 2, 1).view(bs, self.dim, 1, self.base_size[0], self.base_size[1])
        return {self.key_name: self.mlp(x)}
            


class LiDeReMulti(SimpleBase):

    def __init__(self, dim, base_size, backbone, key_names=(('mask', 1),), out_bias=None, n_classes=None, embed_proj=None):
        """
        This is a bit hacky to make CenterNet-like predictions work.
        embed_proj: only for center net classification
        """
        super().__init__()


        self.backbone = backbone
        self.backbone.freeze()
        self.key_names = key_names

        assert n_classes is None 
        assert out_bias is None or len(key_names) == len(out_bias)
        self.dim = dim

        self.inp_dim = self.backbone.feature_dim()
        self.base_size = (base_size, base_size) if isinstance(base_size, int) else base_size
        self.feat_size = self.backbone.base_size(self.backbone.img_size)
        
        out_bias = [0 for _ in key_names] if out_bias is None else out_bias

        self.heads = nn.ModuleDict()
        for (key_name, n_classes), out_bias_ in zip(self.key_names, out_bias):
            self.heads[key_name] = LiDeRe(dim, base_size, backbone, n_classes=n_classes, out_bias=out_bias_, no_freeze=True)

        self.precomputed_features = None         
        self.no_early_device_copy = False
        self.index_order = None

        self.embed_proj = nn.Linear(embed_proj[0], embed_proj[1]) if embed_proj is not None else None
        # _w = self.embed_proj.weight.detach()
        # _w[0] += 2
        # self.embed_proj.weight = nn.Parameter(_w)
        # self.embed_proj.bias = nn.Parameter(0.01*self.embed_proj.bias)


    def forward(self, sample):

        bs = sample['image'].shape[0]
        feats = self.get_features(sample)
        
        return {k: self.heads[k].forward_feats(feats)['mask'] for k,_ in self.key_names}
