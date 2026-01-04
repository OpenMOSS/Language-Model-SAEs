# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from functools import partial
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init
from torch import Tensor, nn
from transformer_lens.hook_points import HookPoint, HookedRootModule
from lm_saes.backend.language_model import LanguageModel, LanguageModelConfig
from typing import Any, Optional, cast
from torchvision.transforms import v2
from lm_saes.utils.timer import timer
from einops import rearrange


logger = logging.getLogger("dinov3")

def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None) -> None:
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    d_in * d_out * k * k
    b * d_in * H * W
    
    1 * dim * 7 * 7 -> dim * [1 1 7 7]
    
    b dim H W
    [b 1 H W] -> [b 1 H W] -> [b dim H W]
    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.layer_scale_init_value = layer_scale_init_value
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.hook_resid_pre = HookPoint()
        self.hook_conv_out = HookPoint()
        self.hook_block_out = HookPoint()
        self.hook_resid_post = HookPoint()
        

    def forward(self, x): # b d_in H W
        input = self.hook_resid_pre(x)
        x = self.dwconv(x) # b d_in H W
        x = self.hook_conv_out(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x) 
        x = self.pwconv1(x) # b H W 4*d_in
        x = self.act(x) 
        x = self.pwconv2(x) # b H W d_in
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.hook_block_out(self.drop_path(x))
        
        x = self.hook_resid_post(input + x)
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).

    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(normalized_shape))
        self.bias = nn.Parameter(torch.empty(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
        self.hook_normalized = HookPoint()

    def init_weights(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.data_format == "channels_last":
            x = self.hook_normalized(F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps))
            return x
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.hook_normalized(self.weight[:, None, None] * x + self.bias[:, None, None])
            return x


class ConvNeXt(HookedRootModule):
    r"""
    Code adapted from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.pyConvNeXt

    A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        patch_size (int | None): Pseudo patch size. Used to resize feature maps to those of a ViT with a given patch size. If None, no resizing is performed
    """

    def __init__(
        self,
        # original ConvNeXt arguments
        in_chans: int = 3,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        # DINO arguments
        patch_size: int | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        # ==== ConvNeXt's original init =====
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x for x in np.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        # ==== End of ConvNeXt's original init =====

        # ==== DINO adaptation ====
        self.head = nn.Identity()  # remove classification head
        self.embed_dim = dims[-1]
        self.embed_dims = dims  # per layer dimensions
        self.n_blocks = len(self.downsample_layers)  # 4
        self.chunked_blocks = False
        self.n_storage_tokens = 0  # no registers

        self.norms = nn.ModuleList([nn.Identity() for i in range(3)])
        self.norms.append(self.norm)

        self.patch_size = patch_size
        self.input_pad_size = 4  # first convolution with kernel_size = 4, stride = 4
        # ==== End of DINO adaption ====
        
        # ==== Hook Setting ====
        self.hook_downsample_outs = [HookPoint() for i in range(4)]
        self.hook_stage_outs = [HookPoint() for i in range(4)]
        self.hook_pool_out = HookPoint()
        self.hook_normalized = HookPoint()
        
        self.setup()

    def init_weights(self):
        self.apply(self._init_weights)
        for stage_id, stage in enumerate(self.stages):
            for block_id, block in enumerate(stage):
                if block.gamma is not None:
                    nn.init.constant_(self.stages[stage_id][block_id].gamma, block.layer_scale_init_value)

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.reset_parameters()
        if isinstance(module, LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        output = []
        for x, masks in zip(x_list, masks_list):
            h, w = x.shape[-2:]
            for i in range(4):
                x = self.hook_downsample_outs[i](self.downsample_layers[i](x))
                x = self.hook_stage_outs[i](self.stages[i](x))
            x_pool = self.hook_pool_out(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
            x = torch.flatten(x, 2).transpose(1, 2)

            # concat [CLS] and patch tokens as (N, HW + 1, C), then normalize
            x_norm = self.hook_normalized(self.norm(torch.cat([x_pool.unsqueeze(1), x], dim=1)))
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_storage_tokens": x_norm[:, 1 : self.n_storage_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.n_storage_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )

        return output

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])

    def _get_intermediate_layers(self, x, n=1):
        h, w = x.shape[-2:]
        output, total_block_len = [], len(self.downsample_layers)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i in range(total_block_len):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in blocks_to_take:
                x_pool = x.mean([-2, -1])
                x_patches = x
                if self.patch_size is not None:
                    # Resize output feature maps to that of a ViT with given patch_size
                    x_patches = nn.functional.interpolate(
                        x,
                        size=(h // self.patch_size, w // self.patch_size),
                        mode="bilinear",
                        antialias=True,
                    )
                output.append(
                    [
                        x_pool,  # CLS (B x C)
                        x_patches,  # B x C x H x W
                    ]
                )
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output
    

    def get_intermediate_layers(
        self,
        x,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ):
        outputs = self._get_intermediate_layers(x, n)

        if norm:
            nchw_shapes = [out[-1].shape for out in outputs]
            if isinstance(n, int):
                norms = self.norms[-n:]
            else:
                norms = [self.norms[i] for i in n]
            outputs = [
                (
                    norm(cls_token),  # N x C
                    norm(patches.flatten(-2, -1).permute(0, 2, 1)),  # N x HW x C
                )
                for (cls_token, patches), norm in zip(outputs, norms)
            ]
            if reshape:
                outputs = [
                    (cls_token, patches.permute(0, 2, 1).reshape(*nchw).contiguous())
                    for (cls_token, patches), nchw in zip(outputs, nchw_shapes)
                ]
        elif not reshape:
            # force B x N x C format for patch tokens
            outputs = [(cls_token, patches.flatten(-2, -1).permute(0, 2, 1)) for (cls_token, patches) in outputs]
        class_tokens = [out[0] for out in outputs]
        outputs = [out[1] for out in outputs]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

convnext_sizes = {
    "tiny": dict(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
    ),
    "small": dict(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
    ),
    "base": dict(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
    ),
    "large": dict(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
    ),
}


def get_convnext_arch(arch_name):
    size_dict = None
    query_sizename = arch_name.split("_")[1]
    try:
        size_dict = convnext_sizes[query_sizename]
    except KeyError:
        raise NotImplementedError("didn't recognize vit size string")

    return partial(
        ConvNeXt,
        **size_dict,
    )
    
def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.CenterCrop((resize_size, resize_size))
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

CROP_DEFAULT_SIZE = 224
RESIZE_DEFAULT_SIZE = int(256 * CROP_DEFAULT_SIZE / 224)

def image_to_token(x: torch.Tensor) -> torch.Tensor:
    """
    x: torch.Tensor, shape [B, 3, H, W]
       每个通道的取值在 {0,...,255}（可以是 uint8 或 float）

    返回: torch.Tensor, shape [B, H, W]
          取值范围在 [0, 256^3 - 1]
    """
    assert x.ndim == 4, f"Expected 4D tensor [B,3,H,W], got {x.shape}"
    assert x.size(1) == 3, f"Channel dimension must be 3, got {x.size(1)}"

    # 转为长整型，避免溢出
    x_long = x.to(torch.long)

    R = x_long[:, 0, ...]  # [B, H, W]
    Gc = x_long[:, 1, ...] # [B, H, W]
    Bc = x_long[:, 2, ...] # [B, H, W]

    out = R + 256 * Gc + 256**2 * Bc  # [B, H, W]
    return out

# def token_to_

class dinov3(LanguageModel):
    def __init__(self, cfg:LanguageModelConfig):
        self.cfg = cfg
        
        self.device = (
            torch.device(f"cuda:{torch.cuda.current_device()}") if cfg.device == "cuda" else torch.device(cfg.device)
        )
        
        assert cfg.model_from_pretrained_path
        
        state_dict = torch.load(cfg.model_from_pretrained_path)
        
        self.model = get_convnext_arch(cfg.model_name)()
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(cfg.device)
        self.transform = make_transform()
        self.to_tensor = v2.Compose([
            v2.ToImage(),
            v2.CenterCrop((256, 256)),
            v2.ToDtype(torch.long, scale=False)
        ])
        
    @property
    def eos_token_id(self) -> int | None:
        return None

    @property
    def bos_token_id(self) -> int | None:
        return None

    @property
    def pad_token_id(self) -> int | None:
        return None
    
    @torch.no_grad()
    def preprocess_raw_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        """
        Expected raw format from ActivationGenerator:
        - raw["image"]: list[PIL.Image.Image]
        - raw.get("meta"): list[dict] (from HuggingFaceDatasetLoader), may include reliance_suppression config
        """
        from PIL import Image, ImageFilter
        import random

        def apply_patch_shuffle(img: "Image.Image", grid_size: int, *, seed: int) -> "Image.Image":
            if grid_size <= 1:
                return img
            w, h = img.size
            patch_w = w // grid_size
            patch_h = h // grid_size
            if patch_w <= 0 or patch_h <= 0:
                return img

            region_w = patch_w * grid_size
            region_h = patch_h * grid_size
            base = img.crop((0, 0, region_w, region_h))

            patches: list[Image.Image] = []
            for gy in range(grid_size):
                for gx in range(grid_size):
                    left = gx * patch_w
                    top = gy * patch_h
                    patches.append(base.crop((left, top, left + patch_w, top + patch_h)))

            rng = random.Random(seed)
            rng.shuffle(patches)

            out = Image.new("RGB", (region_w, region_h))
            idx = 0
            for gy in range(grid_size):
                for gx in range(grid_size):
                    out.paste(patches[idx], (gx * patch_w, gy * patch_h))
                    idx += 1

            if (region_w, region_h) != (w, h):
                out = out.resize((w, h), resample=Image.BILINEAR)
            return out

        def apply_reliance_suppression(img: "Image.Image", meta_i: dict[str, Any] | None) -> "Image.Image":
            if not meta_i:
                return img
            cfg = meta_i.get("reliance_suppression")
            if not isinstance(cfg, dict):
                return img
            kind = cfg.get("kind", None)
            if kind is None or kind == "none":
                return img

            # Deterministic per-sample seed if provided
            ctx = meta_i.get("context_idx", 0)
            try:
                ctx_int = int(ctx)
            except Exception:
                ctx_int = 0
            seed_offset = int(cfg.get("seed_offset", 0))
            seed = ctx_int + seed_offset

            if kind == "shape":
                grid_size = int(cfg.get("grid_size", 8))
                return apply_patch_shuffle(img, grid_size=grid_size, seed=seed)
            if kind == "texture":
                radius = float(cfg.get("gaussian_radius", 2.0))
                return img.filter(ImageFilter.GaussianBlur(radius=radius))
            if kind == "color":
                return img.convert("L").convert("RGB")

            return img

        images = []
        images_raw = []
        metas = raw.get("meta")
        meta_list: list[dict[str, Any] | None]
        if isinstance(metas, list):
            meta_list = metas
        else:
            meta_list = [None] * len(raw["image"])

        for i, image in enumerate(raw["image"]):
            if image.mode == "CMYK":
                image = image.convert("RGB")
            elif image.mode != "RGB":
                image = image.convert("RGB")

            image = apply_reliance_suppression(image, meta_list[i] if i < len(meta_list) else None)

            images.append(self.transform(image))
            images_raw.append(self.to_tensor(image))

        images = torch.stack(images, dim=0)
        images_raw = torch.stack(images_raw, dim=0)
        raw["images"] = images.to(self.cfg.device)
        raw["images_raw"] = images_raw.to(self.cfg.device)
        raw["text"] = str(raw.get("label", ""))
        raw["tokens"] = raw["images"]

        return raw
    
    @torch.no_grad()
    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        with timer.time("run_with_cache_until"):
            # print(raw['images'].shape)
            _, activations = self.model.run_with_cache_until(raw['images'], names_filter=hook_points)
        # print('image', raw['images'].min(), raw['images'].max())
        # print(hook_points[0], type(activations[hook_points[0]]))
        # batch_size, _, H, W = raw["images"].shape
        # images_token = torch.ones(batch_size,H*W, dtype=torch.long, device=self.cfg.device)
        # print("dummy_tokens", dummy_tokens.shape)
        
        images_token = image_to_token(raw['images_raw'])
        images_token = rearrange(images_token, "b h w -> b (h w)")
        for hook_point in hook_points:
            # b, d, h, w = activations[hook_point].shape
            # activations[hook_point] = activations[hook_point].reshape(b, d, h*w).permute(0,2,1)
            activations[hook_point] = rearrange(activations[hook_point], "b d h w -> b (h w) d")
            
        return {hook_point: activations[hook_point].contiguous() for hook_point in hook_points} | {"tokens": images_token.contiguous()}
    
    def to_activations_gradient(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        with timer.time("run_with_cache_until"):
            # print(raw['images'].shape)
            _, activations = self.model.run_with_cache_until(raw['images'], names_filter=hook_points)
        # print('image', raw['images'].min(), raw['images'].max())
        # print(hook_points[0], type(activations[hook_points[0]]))
        # batch_size, _, H, W = raw["images"].shape
        # images_token = torch.ones(batch_size,H*W, dtype=torch.long, device=self.cfg.device)
        # print("dummy_tokens", dummy_tokens.shape)
        
        images_token = image_to_token(raw['images_raw'])
        images_token = rearrange(images_token, "b h w -> b (h w)")
        for hook_point in hook_points:
            # b, d, h, w = activations[hook_point].shape
            # activations[hook_point] = activations[hook_point].reshape(b, d, h*w).permute(0,2,1)
            activations[hook_point] = rearrange(activations[hook_point], "b d h w -> b (h w) d")
            
        return {hook_point: activations[hook_point].contiguous() for hook_point in hook_points} | {"tokens": images_token.contiguous()}
    
    def trace(self, raw: dict[str, Any], n_context: Optional[int] = None) -> list[list[Any]]:
        return None


        