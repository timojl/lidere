import torch
import math
from torch import nn
from functools import partial


class BackboneBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.feat_stride = None
        self.feat_dim = None
        self.precomputed_features = None
        self._post_freeze_done = False

    def freeze(self):

        print('freeze backbone')
        for p in self.parameters():
            p.requires_grad = False

        # avoid running post_freeze_init twice
        if not self._post_freeze_done:
            self.post_freeze_init()
            self._post_freeze_done = True

    def post_freeze_init(self):
        pass

    def feature_dim(self):
        return self.feat_dim
    
    def precompute(self, loader, amp=True, dtype=None):
        """ this is the new precomputation function. """

        if self.precomputed_features is None:
            self.precomputed_features = dict()

        import tqdm
        device = next(self.parameters()).device
        for sample in tqdm.tqdm(loader):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                with torch.no_grad():
                    img = sample['image']
                    img = torch.nn.functional.interpolate(img[:,:,0], self.img_size, mode='bicubic', antialias=True)[:,:,None]
                    feats = self.forward(img.to(device))
            assert 'id' not in sample or isinstance(sample['id'][0], str)
            for f, key in zip(feats, sample['id']):
                if dtype is not None:
                    f = f.to(dtype)
                self.precomputed_features[key] = f.to(device)

    def features_compute_or_cached(self, x, keys, device):
        if self.precomputed_features is not None and all(k in self.precomputed_features for k in keys):
            return torch.stack([self.precomputed_features[k] for k in keys]).to(device)
        else:
            return self.forward(x)

    def base_size(self, img_size):
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        return img_size[0] // self.feat_stride, img_size[1] // self.feat_stride


class NoBackbone(BackboneBase):

    def __init__(self, feat_stride, feat_dim):
        super().__init__()
        self.feat_stride = feat_stride
        self.feat_dim = feat_dim



class TimmBackbone(BackboneBase):

    def __init__(self, model_name, feat_dim, feat_stride, img_size, cls_token='first', concat_layers=None, no_post=False, 
                 pretrained=True, normalize=False, lora=None):
        super().__init__()
        import timm

        self.lora = lora
        from lidere import utilities as ut

        with ut.TimeBlock('load model'):
            import os
            
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size, global_pool='')
        
        data_cfg = timm.data.resolve_data_config(self.backbone.pretrained_cfg)
        self.transform = timm.data.create_transform(**data_cfg)
        self.transform.transforms = self.transform.transforms[3:]

        self.feat_dim = feat_dim + (feat_dim*len(concat_layers) if concat_layers is not None else 0)
        self.feat_stride = feat_stride
        self.img_size = img_size if isinstance(img_size, (list, tuple)) else [img_size, img_size]
        self.cls_token = cls_token
        self.no_post = no_post
        self.concat_layers = concat_layers
        self.normalize = normalize

    def post_freeze_init(self):
        
        if self.lora is not None:
            print('lora', self.lora)
            from lidere.models.components import LoRALinear
            for block in self.backbone.blocks:
                attn = block.attn
                attn.qkv = LoRALinear(attn.qkv, r=self.lora['r'], alpha=self.lora['alpha'])


    def post_backbone(self, x, height, width):

        bs = x.shape[0]
        feat_dim = x.shape[2]
        s = {
            'first': lambda: slice(1, None), 
            'last': lambda: slice(0, -1), 
            'none': lambda: slice(0, None),
            'auto': lambda: slice(self.backbone.num_prefix_tokens, None)
        }[self.cls_token]()

        x = x.permute(0,2,1)[:,:,s]

        x = x.view(bs, feat_dim, height // self.feat_stride, width // self.feat_stride)
        
        if self.normalize:
            x = torch.nn.functional.normalize(x, p=2, dim=1)

        return x
    
    def forward(self, x):

        if self.concat_layers is not None:
            extra_activations = dict()
            def hook(model, input, output, layer_i): 
                extra_activations[layer_i] = output.detach()

            # this is specific to DinoV2
            for layer_i in self.concat_layers:
                self.backbone.blocks[layer_i].register_forward_hook(partial(hook, layer_i=layer_i))
    
    
        bs, _, n_frames, height, width = x.shape
        x = x.permute(0,2,1,3,4).flatten(0,1)

        x = self.transform(x)
        x = self.backbone(x)

        if not self.no_post:
            x = self.post_backbone(x, height, width)

        if self.concat_layers is not None:
            x = torch.cat([x] + [self.post_backbone(extra_activations[l], height, width) 
                                 for l in self.concat_layers], dim=1)
        
        x = x.view(bs, n_frames, *x.shape[1:])
        x = x.transpose(1,2)
        return x



class TimmSwinBackbone(TimmBackbone):
    def post_backbone(self, x, height, width):

        bs = x.shape[0]
        feat_dim = x.shape[2]
        s = {
            'first': lambda: slice(1, None), 
            'last': lambda: slice(0, -1), 
            'none': lambda: slice(0, None),
            'auto': lambda: slice(self.backbone.num_prefix_tokens, None)
        }[self.cls_token]()

        return x.permute(0,3,1,2)
    

class TimmCNNBackbone(BackboneBase):

    def __init__(self, model_name, base_size, feat_dim, layers='last', pretrained=True, normalize=False, img_size=224):
        super().__init__()
        import timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        data_cfg = timm.data.resolve_data_config(self.backbone.pretrained_cfg)
        self.transform = timm.data.create_transform(**data_cfg)
        self.transform.transforms = self.transform.transforms[3:]
        self.layers = layers

        # self.feat_dim = feat_dim
        # self.feat_stride = feat_stride
        self.base_size_ = base_size
        self.img_size = img_size
        self.normalize = normalize

        self.feat_dim = feat_dim

    def base_size(self, s):
        return self.base_size_

    def forward(self, x):

        bs, _, n_frames, height, width = x.shape
        x = x.permute(0,2,1,3,4).flatten(0,1)
        x = self.transform(x)
        x = self.backbone(x)

        if self.layers == 'last':
            x = x[-1]
        elif self.layers == 'last_two':
            a = torch.nn.functional.interpolate(x[-1], (20, 20), mode='bilinear')
            x = torch.cat([x[-2], a], dim=1)

        x = x.view(bs, n_frames, *x.shape[1:])
        x = x.transpose(1,2)

        if self.normalize:
            x = torch.nn.functional.normalize(x, p=2, dim=1)

        return x



class TimmAutoBackbone(BackboneBase):

    def __init__(self, model_name, no_post=False, pretrained=True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
        data_cfg = timm.data.resolve_data_config(self.backbone.pretrained_cfg)
        self.transform = timm.data.create_transform(**data_cfg)
        self.transform.transforms = self.transform.transforms[3:]

        self.feat_dim = self.backbone.num_features

        img_size = self.backbone.default_cfg['input_size']
        
        # figure out the stride
        o = self.backbone(torch.rand(1,*img_size))
        o = self.post_backbone(o, None, None)
        self.feat_stride = img_size[-1] // o.shape[-1]


        assert img_size[-2] == img_size[-1], 'currently only square input supported'
        self.img_size = img_size[-1]
        self.no_post = no_post

    def post_backbone(self, x, height, width):
        bs = x.shape[0]
        s = slice(self.backbone.num_prefix_tokens, None)
        x = x.permute(0,2,1)[:,:,s]

        out_size = math.isqrt(x.shape[-1])
        assert out_size**2 == x.shape[-1]
        return x.view(bs, self.feat_dim, out_size, out_size)
    
    def forward(self, x):

        bs, _, n_frames, height, width = x.shape
        x = x.permute(0,2,1,3,4).flatten(0,1)
        x = self.transform(x)
        x = self.backbone(x)

        if not self.no_post:
            x = self.post_backbone(x, height, width)
        
        x = x.view(bs, n_frames, *x.shape[1:])
        x = x.transpose(1,2)
        return x



class DinoBackbone(BackboneBase):

    def __init__(self, version='vit_s', img_size=224):
        super().__init__()
        import timm

        if version == 'vit_s':
            self.backbone = timm.create_model('vit_small_patch8_224.dino', pretrained=True, img_size=img_size, num_classes=0, global_pool='')
            self.feat_dim = 384
        elif version == 'vit_b':
            self.backbone = timm.create_model('vit_base_patch8_224.dino', pretrained=True, img_size=img_size, num_classes=0, global_pool='')
            self.feat_dim = 768
        else:
            raise ValueError('invalid version')
        
        data_cfg = timm.data.resolve_data_config(self.backbone.pretrained_cfg)
        # data_cfg = {k: v for k,v in data_cfg.items() if k in {'mean', 'std'}}
        self.transform = timm.data.create_transform(**data_cfg)
        self.transform.transforms = self.transform.transforms[3:]
        self.feat_stride = 8
        

    def post_backbone(self, x, height, width):
        bs = x.shape[0]
        return x.permute(0,2,1)[:,:,1:].view(bs, self.feat_dim, height // 8, width // 8)
    
    def feature_dim(self):
        return self.feat_dim
    
    def forward(self, x):

        bs, _, n_frames, height, width = x.shape
        x = x.permute(0,2,1,3,4).flatten(0,1)
        
        x = self.transform(x)
        # self.backbone.eval()
        x = self.backbone(x)
        x = self.post_backbone(x, height, width)

        x = x.view(bs, n_frames, *x.shape[1:])
        x = x.transpose(1,2)
        return x



class ResnetBackbone(BackboneBase):

    def __init__(self, version='resnet18', pretrained=False, output_layer=4):
        super().__init__()
        import timm

        print('RN pretrained', pretrained)

        self.backbone = timm.create_model(version, pretrained=pretrained, num_classes=0, global_pool='')
        data_cfg = timm.data.resolve_data_config(self.backbone.pretrained_cfg)
        # data_cfg = {k: v for k,v in data_cfg.items() if k in {'mean', 'std'}}
        self.transform = timm.data.create_transform(**data_cfg)
        self.transform.transforms = self.transform.transforms[3:]

        self.feat_dim = {'resnet18': 512, 'resnet50': 2048}[version]
        self.feat_dim = self.feat_dim // {4: 1, 3: 2, 2: 4}[output_layer]

        self.feat_stride = {4: 32, 3:16, 2:8}[output_layer]

        self.output_layer = output_layer
        

    def forward(self, x):

        bs, _, n_frames, height, width = x.shape
        x = x.permute(0,2,1,3,4).flatten(0,1)
        
        x = self.transform(x)
        # x = self.backbone(x)

        # resnet forward
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        
        if self.output_layer >= 2:
            x = self.backbone.layer2(x)
        
        if self.output_layer >= 3:
            x = self.backbone.layer3(x)
        
        if self.output_layer >= 4:
            x = self.backbone.layer4(x)

        x = x.view(bs, n_frames, *x.shape[1:])
        x = x.transpose(1,2)

        return x


class ResnetBackboneJoin(BackboneBase):

    def __init__(self, version='resnet18', pretrained=False):
        super().__init__()
        import timm

        print('RN pretrained', pretrained)
        

        self.backbone = timm.create_model(version, pretrained=pretrained, num_classes=0, global_pool='')
        data_cfg = timm.data.resolve_data_config(self.backbone.pretrained_cfg)
        # data_cfg = {k: v for k,v in data_cfg.items() if k in {'mean', 'std'}}
        self.transform = timm.data.create_transform(**data_cfg)
        self.transform.transforms = self.transform.transforms[3:]

        self.feat_dim = {'resnet18': 512, 'resnet50': 2048}[version]

        self.feat_dim = 256
        self.feat_stride = 8

        self.proj3 = nn.Conv2d(256, 128, kernel_size=1)
        self.proj4 = nn.Conv2d(512, 128, kernel_size=1)

        self.final = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(128*3, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )
        

    def forward(self, x):

        bs, _, n_frames, height, width = x.shape
        x = x.permute(0,2,1,3,4).flatten(0,1)
        
        x = self.transform(x)
        # x = self.backbone(x)

        # resnet forward
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        
        x2 = self.backbone.layer2(x)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        x3_up = nn.functional.interpolate(self.proj3(x3), x2.shape[-2:])
        x4_up = nn.functional.interpolate(self.proj4(x4), x2.shape[-2:])

        x = torch.cat([x2, x3_up, x4_up], dim=1)
        x = self.final(x)

        x = x.view(bs, n_frames, *x.shape[1:])
        x = x.transpose(1,2)

        return x

