r"""
Has a quantized VIT
"""
from functools import partial
from typing import Optional, Callable, Union, Tuple

import torch
from torch import nn

from torch_int.nn.linear import (
        W8A8BFP32OFP32Linear,
        W8A8B8O8Linear,
        W8A8B8O8LinearReLU,
        W8A8B8O8LinearGELU,
        W8A8BFP32OFP32LinearGELU
        )

from torch_int.nn.bmm import BMM_S8T_S8N_S8T, BMM_S8T_S8N_F32T
from torch_int.nn.fused import LayerNormQ

# from detectron2.modeling.backbone.vit import Attention, Block
# from detectron2.modeling.backbone.utils import (
        # window_partition,
        # window_unpartition,
        # add_decomposed_rel_pos)


class Int8Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim: int, num_heads: int, use_rel_pos: bool,
            input_size = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )

        self.attention_weight_scale = 1.0

        self.qk_bmm = BMM_S8T_S8N_F32T(1.0)
        self.pv_bmm = BMM_S8T_S8N_S8T(1.0)

        # self.qkv = W8A8B8O8Linear(embed_dim, 3 * embed_dim)
        self.q = W8A8B8O8Linear(embed_dim, embed_dim)
        self.k = W8A8B8O8Linear(embed_dim, embed_dim)
        self.v = W8A8B8O8Linear(embed_dim, embed_dim)
        self.proj = W8A8BFP32OFP32Linear(embed_dim, embed_dim)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # added to the (unnformalized) attention weight.s Keep in float 
            # to simipliciy, as the unnformalized weights are also in float
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, self.head_dim))

    @staticmethod
    @torch.no_grad()
    def from_float(module: Attention,
                   input_scale: float,
                   q_output_scale: float,
                   k_output_scale: float,
                   v_output_scale: float,
                   proj_input_scale: float):
        int8_module = Int8Attention(module.qkv.in_features, module.num_heads,
                use_rel_pos=True,
                input_size=(14, 14))
        in_features = module.qkv.in_features
        # out_features = module.qkv.out_features  // 3
        q_module = nn.Linear(in_features=in_features,
                out_features=out_features,device='cuda')
        k_module = nn.Linear(in_features=in_features,
                out_features=out_features,
                device='cuda')
        v_module = nn.Linear(in_features=in_features,
                out_features=out_features,
                device='cuda')
        q_module.weight = torch.nn.Parameter(module.qkv.weight[:out_features, :])
        q_module.bias = torch.nn.Parameter(module.qkv.bias[:out_features])
        k_module.weight = torch.nn.Parameter(module.qkv.weight[out_features:2*out_features, :])
        k_module.bias = torch.nn.Parameter(module.qkv.bias[out_features: 2*out_features])
        v_module.weight = torch.nn.Parameter(module.qkv.weight[2 * out_features:, :])
        v_module.bias = torch.nn.Parameter(module.qkv.bias[2 * out_features:])
        # Fuse the scaling into the q_proj output scale
        # The scale is the qkv / scale thing
        # qkv_output_scale = qkv_output_scale * module.scale
        # int8_module.qkv = W8A8B8O8Linear.from_float(
                # module.qkv, input_scale, q_output_scale)
        q_output_scale *= module.scale
        q_module.weight *= module.scale
        q_module.bias *= module.scale
        breakpoint()
        # module.proj.weight *= module.scale
        # module.proj.bias *= module.scale
        # qkv_output_scale = qkv_output_scale
        #Seperate three linear layers, in particular v has a different scale
        int8_module.q = W8A8B8O8Linear.from_float(
            q_module, input_scale, q_output_scale)
        int8_module.k = W8A8B8O8Linear.from_float(
            k_module, input_scale, k_output_scale)
        int8_module.v = W8A8B8O8Linear.from_float(
            v_module, input_scale, v_output_scale)
        int8_module.proj = W8A8BFP32OFP32Linear.from_float(
            module.proj, proj_input_scale)
        int8_module.qk_bmm = BMM_S8T_S8N_F32T.from_scale(
            q_output_scale, k_output_scale)

        if int8_module.use_rel_pos:
            int8_module.rel_pos_h = module.rel_pos_h
            int8_module.rel_pos_w = module.rel_pos_w
            # not sure if the same scales make sense...

        # alpha = s_prob * s_v / s_out, where s_prob = 1 / 127
        int8_module.pv_bmm = BMM_S8T_S8N_S8T.from_scale(
            1.0 / 127, v_output_scale, proj_input_scale)
        return int8_module

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x): #25x14x14x1024 (for large)
        B, H, W, _ = x.shape #H, W: Number of patches
        q = self.q(x).reshape(B, H * W, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B * self.num_heads, H * W, -1)
        k = self.k(x).reshape(B, H * W, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B * self.num_heads, H * W, -1)

        attn = self.qk_bmm(q, k)
        #attn shape: 400x196x196

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, (q * 0.01),
                                          self.rel_pos_h, self.rel_pos_w,
                                          (H, W), (H, W))

        attn_probs = nn.functional.softmax(attn, dim=-1) #why not take max for scaling too?
        attn_probs.mul_(127).round_()
        attn_probs = attn_probs.to(torch.int8)

        #different layout because pv_bmm takes a col major matrix as second arg
        v = self.v(x).reshape(B, H * W, self.num_heads, -1)\
                .permute(0, 2, 3, 1)\
                .reshape(B * self.num_heads, 64, H * W)
        anew = torch.zeros((B * self.num_heads, 196, 224), dtype=torch.int8, device='cuda')
        bnew = torch.zeros((B * self.num_heads, 64, 224), dtype=torch.int8, device='cuda')
        bnew[:, :, :196] = v
        anew[:, :, :196] = attn_probs
        x = self.pv_bmm(anew, bnew)

        x = x.reshape(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x


class Int8Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        use_residual_block=False,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = LayerNormQ(dim)
        self.attn = Int8Attention(
            dim,
            num_heads=num_heads,
            use_rel_pos=True,
            input_size=(14, 14))
            # qkv_bias=qkv_bias)
            # use_rel_pos=use_rel_pos,
            # rel_pos_zero_init=rel_pos_zero_init,
            # input_size=input_size if window_size == 0 else (window_size, window_size),
        # )

        from timm.models.layers import DropPath, Mlp

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = LayerNormQ(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)
        # del self.mlp.act

        self.window_size = window_size

        self.use_residual_block = use_residual_block
        assert not self.use_residual_block, "residual block not yet supported"
        # if use_residual_block:
            # # Use a residual block with bottleneck channel as dim // 2
            # self.residual = ResBottleneckBlock(
                # in_channels=dim,
                # out_channels=dim,
                # bottleneck_channels=dim // 2,
                # norm="LN",
                # act_layer=act_layer,
            # )

    @staticmethod
    def from_float(module: Block,
                   attn_input_scale: float,
                   q_output_scale: float,
                   k_output_scale: float,
                   v_output_scale: float,
                   proj_input_scale: float,
                   fc1_input_scale: float,
                   fc2_input_scale: float):
        #this is equation 1 from the paper.
        int8_module = Int8Block(
            module.attn.qkv.in_features,
            module.attn.num_heads,
            window_size=module.window_size)
            # module.attn.qkv.out_features,
        int8_module.norm1 = LayerNormQ.from_float(module.norm1, attn_input_scale)
        int8_module.attn = Int8Attention.from_float(
            module.attn, attn_input_scale, q_output_scale, k_output_scale,
            v_output_scale, proj_input_scale)
        breakpoint()
        int8_module.norm2 = LayerNormQ.from_float(
            module.norm2, fc1_input_scale)
        int8_module.mlp.fc1 = W8A8BFP32OFP32Linear.from_float(
            module.mlp.fc1, fc1_input_scale)
        # int8_module.mlp.fc1 = W8A8B8O8LinearGELU.from_float(
            # module.mlp.fc1, fc1_input_scale, fc2_input_scale)
        int8_module.mlp.fc2 = module.mlp.fc2
        # int8_module.mlp.fc2 = W8A8BFP32OFP32Linear.from_float(
            # module.mlp.fc2, fc2_input_scale)
        return int8_module

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x) #2% error
        breakpoint()
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x_tmp = x
        x = self.norm2(x)
        x = self.mlp.fc1(x)
        x = self.mlp.act(x)
        x = self.mlp.drop1(x)
        x = self.mlp.fc2(x)
        x = self.mlp.drop2(x)
        x = x_tmp + self.drop_path(x)

        # x = shortcut + self.drop_path(x)
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.use_residual_block:
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return x


class VITINT8(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image
    Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            weight_init: str = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[Callable] = None,
            act_layer: Optional[Callable] = None,
            block_fn: Callable = Int8Block,
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
    ):
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_class_token: bool = False,
            norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    @staticmethod
    def from_float(module, decoder_layer_scales):
        int8module = VITINT8()
        for i, layer in enumerate(module.blocks):
            int8module.blocks[i] = Int8Block.from_float(layter, 
                    decoder_layer_scales[i])
        return int8module
