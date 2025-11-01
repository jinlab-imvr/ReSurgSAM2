import torch.nn as nn
# from mamba_ssm import Mamba
from mamba_ssm.modules.mamba_simple import Mamba

import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from sam2.modeling.sam2_utils import MLP
from typing import Optional
from functools import partial

import torch
from torch import nn, Tensor

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class MambaLayer(nn.Module):
    def __init__(
        self,
        dim,
        d_state = 16,
        d_conv = 4,
        expand = 2,
        mlp_ratio=4,
        drop_path=0.,
        bimamba=False,
        use_dwconv=False,
        sp_bimamba=False
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim, # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,    # Local convolution width
            expand=expand,    # Block expansion factor
            bimamba=bimamba,
            sp_bimamba=sp_bimamba,
            # use_fast_path=False,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.use_dwconv = use_dwconv
        self.sp_bimamba = sp_bimamba

        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_dwconv:
            self.mlp = DWMLP(dim, mlp_hidden_dim, dim)
        else:
            self.mlp = MLP(dim, mlp_hidden_dim, dim, 2)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 打印m的类型
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        else:
            pass

    def forward(self, x, vol_sizes=None):
        B, N, C = x.shape

        if self.sp_bimamba:
            x_mamba = x + self.drop_path(self.mamba(self.norm1(x), vol_sizes=vol_sizes))
        else:
            x_mamba = x + self.drop_path(self.mamba(self.norm1(x)))

        if self.use_dwconv:
            x_mamba = x_mamba + self.drop_path(self.mlp(self.norm2(x_mamba), vol_sizes))
        else:
            x_mamba = x_mamba + self.drop_path(self.mlp(self.norm2(x_mamba)))

        return x_mamba


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

    def forward(self, x, vol_sizes):

        B, N, C = x.shape  # (B, T*H*W, C)
        T, H, W = vol_sizes
        x = x.view(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, T, H, W, C) -> (B*T, C, H, W)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).reshape(B, T * H * W, C)  # (B * T, C, H, W) -> (B, T*H*W, C)

        return x  # (B, T*H*W, C)

class DWMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, max_num_frames=10):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.dwconv = DWConv(in_features)
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.grn = GRN(hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):  # 新增Conv3d初始化
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, vol_sizes):
        x = self.dwconv(x, vol_sizes)
        x = self.norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.fc2(x)

        return x  # residual will be added outside

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True).contiguous()
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True).contiguous() + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
        use_checkpoint=False
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        # if use_checkpoint:
        #     hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        # else:
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    bimamba=True,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

if __name__ == "__main__":
    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # mamba_layer = MambaLayer(256, bimamba=True)
    # mamba_layer.to(device)
    # x = torch.randn(4, 10, 256).to(device)
    # x = x.to(device)
    # y = mamba_layer(x)
    # print(y.shape)

    ssm_cfg = {}
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    rms_norm = True
    norm_epsilon = 1e-5
    d_model = 256
    drop_path = 0.1
    fused_add_norm = True
    residual_in_fp32 = True
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=1, bimamba=True, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = create_block(
        d_model,
        ssm_cfg=ssm_cfg,
        norm_epsilon=norm_epsilon,
        drop_path=drop_path,
        rms_norm=rms_norm,
        residual_in_fp32=residual_in_fp32,
        fused_add_norm=fused_add_norm,
        layer_idx=1,
        bimamba=True,
        device=device,
        dtype=dtype,
    )
    block.to(device)
    x = torch.randn(4, 10, 256).to(device)
    x = x.to(device)
    y, residual = block(x)
    print(y.shape)
    print(residual.shape)

