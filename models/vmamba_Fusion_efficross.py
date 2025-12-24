# 导入所需的库和模块
import time
import math
from functools import partial
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# 导入Mamba SSM的选择性扫描接口
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
#from models.cross import VSSBlock_Cross_new
#from models.cross import VSSBlock_new
from .wavelet import MobileMambaBlock

# 尝试导入不同的选择性扫描实现版本
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

# 修改DropPath的字符串表示方法
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class MobileMambaAdapter(nn.Module):
    """
    适配器: 将 FusionMamba 的 BHWC 格式转换为 MobileMambaBlock 需要的 BCHW 格式。
    """
    def __init__(self, dim, d_state=16, d_conv=3, expand=2, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        # 初始化你在 wavelet.py 里写的 Block
        self.block = MobileMambaBlock(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            drop_path=drop_path,
            ratio_mamba=0.6,
            ratio_conv=0.3
        )
        # FusionMamba 的输入通常没有 LayerNorm，为了稳定我们在 Adapter 里加一个
        self.norm = norm_layer(dim)

    def forward(self, x):
        # 输入 x: [B, H, W, C] (FusionMamba 标准格式)
        
        # 1. 归一化
        x_norm = self.norm(x)
        
        # 2. 核心转换: BHWC -> BCHW
        # permute(0, 3, 1, 2) 将 Channel 移到第二维
        x_permuted = x_norm.permute(0, 3, 1, 2).contiguous()
        
        # 3. 调用专注于 BCHW 的 MobileMambaBlock
        out = self.block(x_permuted)
        
        # 4. 核心转换: BCHW -> BHWC
        # permute(0, 2, 3, 1) 将 Channel 移回最后一维
        out_original = out.permute(0, 2, 3, 1).contiguous()

        return out_original
class MobileMambaCrossAdapter(nn.Module):
    """
    跨模态适配器：专门用于替代 VSSBlock_Cross_new。
    逻辑：先拼接融合，再过 MobileMambaBlock。
    Input: x1 (BHWC), x2 (BHWC) -> Output: fused_x (BHWC)
    """
    def __init__(self, dim, d_state=16, d_conv=3, expand=2, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        
        # 1. 特征降维融合层 (2*dim -> dim)
        # 输入是 BHWC 格式，所以用 Linear 处理 Channel 维度最方便
        self.fusion_reduce = nn.Linear(dim * 2, dim)
        
        # 2. 核心 MobileMamba 模块 (复用之前的 Adapter 即可)
        self.mobile_mamba = MobileMambaAdapter(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            drop_path=drop_path,
            norm_layer=norm_layer
        )
        
        # 3. 激活函数 (可选，用于 Linear 之后)
        self.act = nn.SiLU()

    def forward(self, x1, x2):
        # x1, x2: [B, H, W, C]
        
        # 1. 拼接 (Concat)
        x_cat = torch.cat([x1, x2], dim=-1) # [B, H, W, 2C]
        
        # 2. 降维融合
        x_fused = self.fusion_reduce(x_cat) # [B, H, W, C]
        x_fused = self.act(x_fused)
        
        # 3. 输入 MobileMamba 进行增强
        out = self.mobile_mamba(x_fused)
        
        return out
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    计算选择性扫描操作的FLOPs（浮点运算次数）
    
    参数:
        B: 批次大小
        L: 序列长度
        D: 特征维度
        N: 状态维度
        with_D: 是否包含D参数
        with_Z: 是否包含z参数
        with_Group: 是否使用分组
        with_complex: 是否使用复数
    
    返回:
        flops: 浮点运算次数
    """
    
    # 内部函数：计算einsum操作的FLOPs
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # 除以2因为我们将MAC（乘加）计为一次flop
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # 初始化flops计数器
    
    # 计算deltaA和deltaB_u的einsum操作flops
    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    
    # 计算循环内的flops
    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    
    # 添加D和Z参数的flops
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
        
    return flops


class PatchEmbed2D(nn.Module):
    r""" 图像到补丁嵌入的转换
    将输入图像分割成补丁并进行嵌入
    
    参数:
        patch_size (int): 补丁大小. 默认: 4.
        in_chans (int): 输入图像通道数. 默认: 3.
        embed_dim (int): 线性投影输出通道数. 默认: 96.
        norm_layer (nn.Module, optional): 归一化层. 默认: None
    """

    def __init__(self, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        # 使用卷积进行补丁嵌入
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 可选的归一化层
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # 卷积投影并调整维度顺序 (B, C, H, W) -> (B, H, W, C)
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" 补丁合并层
    用于降低特征图分辨率，同时增加通道数
    
    参数:
        dim (int): 输入通道数.
        norm_layer (nn.Module, optional): 归一化层.  默认: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # 线性变换将4个相邻补丁合并为一个
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        # 处理奇数尺寸的情况
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        # 提取四个相邻位置的特征
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C 左上
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C 右上
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C 左下
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C 右下

        # 处理尺寸不匹配情况
        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        # 拼接四个位置的特征
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        # 归一化和线性变换
        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand2D(nn.Module):
    """补丁扩展层，用于上采样"""
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        # 线性扩展通道数
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        # 重新排列张量以实现上采样
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


class Final_PatchExpand2D(nn.Module):
    """最终的补丁扩展层"""
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        # 线性扩展通道数
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)
        self.post_conv = nn.Conv2d(self.dim // dim_scale, self.dim // dim_scale, 
                                   kernel_size=3, stride=1, padding=1,groups=self.dim // dim_scale)
    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        # 重新排列张量以实现上采样
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.post_conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class SS2D(nn.Module):
    """
    2D选择性扫描模块 (Swin Transformer风格)
    实现了Mamba中的状态空间模型在2D数据上的应用
    """
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        # 内部维度计算
        self.d_inner = int(self.expand * self.d_model)
        # 时间步长秩的计算
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 输入投影层，将输入映射到内部维度的两倍
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        # 深度可分离卷积层
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        # 激活函数
        self.act = nn.SiLU()

        # 四个方向的投影层 (水平、垂直、对角线等)
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        # 将投影权重堆叠为参数
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        # 时间步长投影层
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        # A日志初始化和D参数初始化
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # 核心前向传播函数
        self.forward_core = self.forward_corev0

        # 输出归一化和投影层
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        """
        初始化时间步长投影层
        """
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # 初始化特殊的时间步长投影以保持初始化时的方差
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # 初始化时间步长偏置，使F.softplus(dt_bias)在dt_min和dt_max之间
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # softplus的逆运算
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # 标记此偏置不需要重新初始化
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        """
        S4D实数初始化
        """
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # 保持A_log为fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        """
        D "跳跃"参数初始化
        """
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # 保持为fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        """
        核心前向传播函数v0版本
        """
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W  # 序列长度
        K = 4  # 方向数

        # 构造不同方向的输入序列
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        # 投影计算dt、B、C参数
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        # 数据类型转换和视图重塑
        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        # 执行选择性扫描
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # 处理不同方向的输出
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        """
        前向传播主函数
        """
        B, H, W, C = x.shape

        # 输入投影，分为x和z两部分
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        # 卷积处理
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        
        # 核心选择性扫描处理
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4  # 合并不同方向的结果
        
        # 输出处理
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)  # 门控机制
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class Conv2d_Hori_Veri_Cross(nn.Module):
    """
    水平垂直交叉卷积层
    用于提取水平和垂直方向的特征
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Hori_Veri_Cross, self).__init__()
        # 基础卷积层 (1x5卷积核)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        # 获取卷积权重形状
        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        # 创建零填充张量
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        
        # 重构卷积核为3x3形式
        conv_weight = torch.cat((tensor_zeros, self.conv.weight[:, :, :, 0], tensor_zeros, self.conv.weight[:, :, :, 1],
                                 self.conv.weight[:, :, :, 2], self.conv.weight[:, :, :, 3], tensor_zeros,
                                 self.conv.weight[:, :, :, 4], tensor_zeros), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        # 正常卷积
        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride,
                              padding=self.conv.padding)

        # 如果theta为0，直接返回正常卷积结果
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # 计算差分响应
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            # 返回组合结果
            return out_normal - self.theta * out_diff


class Conv2d_Diag_Cross(nn.Module):
    """
    对角线交叉卷积层
    用于提取对角线方向的特征
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Diag_Cross, self).__init__()
        # 基础卷积层 (1x5卷积核)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        # 获取卷积权重形状
        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        # 创建零填充张量
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        
        # 重构卷积核为对角线形式
        conv_weight = torch.cat((self.conv.weight[:, :, :, 0], tensor_zeros, self.conv.weight[:, :, :, 1], tensor_zeros,
                                 self.conv.weight[:, :, :, 2], tensor_zeros, self.conv.weight[:, :, :, 3], tensor_zeros,
                                 self.conv.weight[:, :, :, 4]), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        # 正常卷积
        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride,
                              padding=self.conv.padding)

        # 如果theta为0，直接返回正常卷积结果
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # 计算差分响应
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            # 返回组合结果
            return out_normal - self.theta * out_diff


class Conv2d_CDC(nn.Module):
    """
    CDC (Cross Difference Convolution) 卷积层
    结合水平垂直和对角线特征提取
    """
    def __init__(self, in_channels, basic_conv1=Conv2d_Hori_Veri_Cross, basic_conv2=Conv2d_Diag_Cross, theta=0.8):
        super(Conv2d_CDC, self).__init__()
        # 水平垂直分支
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//2,1,1),  # 1x1卷积降维
            basic_conv1(in_channels//2, in_channels//2, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
        )

        # 对角线分支
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1, 1),
            basic_conv2(in_channels//2, in_channels//2, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
        )

        # 最终融合卷积
        self.lastconv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        self.theta = theta

    def forward(self, x):
        # 分别处理两个分支
        x_H = self.conv1(x)      # 水平垂直特征
        x_D = self.conv1_2(x)    # 对角线特征
        # 特征融合
        depth = torch.cat((x_H, x_D), dim=1)
        depth = self.lastconv3(depth)  # x [1, 32, 32]

        return depth


class LDC(nn.Module):
    """
    LDC (Learnable Difference Convolution) 可学习差分卷积
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        # conv.weight.size() = [out_channels, in_channels, kernel_size, kernel_size]
        super(LDC, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)  # [12,3,3,3]

        # 中心掩码
        self.center_mask = torch.tensor([[0, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 0]]).cuda()
        # 基础掩码
        self.base_mask = nn.Parameter(torch.ones(self.conv.weight.size()), requires_grad=False)  # [12,3,3,3]
        # 可学习掩码
        self.learnable_mask = nn.Parameter(torch.ones([self.conv.weight.size(0), self.conv.weight.size(1)]),
                                           requires_grad=True)  # [12,3]
        # 可学习参数theta
        self.learnable_theta = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)  # [1]

    def forward(self, x):
        # 计算动态掩码
        mask = self.base_mask - self.learnable_theta * self.learnable_mask[:, :, None, None] * \
               self.center_mask * self.conv.weight.sum(2).sum(2)[:, :, None, None]

        # 应用掩码进行卷积
        out_diff = F.conv2d(input=x, weight=self.conv.weight * mask, bias=self.conv.bias, stride=self.conv.stride,
                            padding=self.conv.padding,
                            groups=self.conv.groups)
        return out_diff


class VSSLayer(nn.Module):
    """
    VSS层，包含多个VSS块
    Args:
        dim (int): 输入通道数.
        depth (int): 块的数量.
        drop (float, optional): Dropout率. 默认: 0.0
        attn_drop (float, optional): 注意力dropout率. 默认: 0.0
        drop_path (float | tuple[float], optional): 随机深度率. 默认: 0.0
        norm_layer (nn.Module, optional): 归一化层. 默认: nn.LayerNorm
        downsample (nn.Module | None, optional): 下采样层. 默认: None
        use_checkpoint (bool): 是否使用检查点节省内存. 默认: False.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        layer_list = []
        for i in range(depth):
            layer_list.append(
                        MobileMambaAdapter(
                        dim=dim,
                        d_state=d_state,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer
                    )
            )
        # 创建VSS块列表
        self.blocks = nn.ModuleList(layer_list)

        # 权重初始化
        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        # 下采样层
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        # 依次通过每个MobileMambablock块
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        # 下采样
        if self.downsample is not None:
            x = self.downsample(x)

        return x


class VSSLayer_up(nn.Module):
    """
    VSS上采样层
    Args:
        dim (int): 输入通道数.
        depth (int): 块的数量.
        drop (float, optional): Dropout率. 默认: 0.0
        attn_drop (float, optional): 注意力dropout率. 默认: 0.0
        drop_path (float | tuple[float], optional): 随机深度率. 默认: 0.0
        norm_layer (nn.Module, optional): 归一化层. 默认: nn.LayerNorm
        upsample (nn.Module | None, optional): 上采样层. 默认: None
        use_checkpoint (bool): 是否使用检查点节省内存. 默认: False.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            upsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # 创建mobileMamba块列表
        self.blocks = nn.ModuleList([
            MobileMambaAdapter(
                dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                d_state=d_state,
                norm_layer=norm_layer
            )
            for i in range(depth)])

        # 权重初始化
        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        # 上采样层
        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        # 上采样
        if self.upsample is not None:
            x = self.upsample(x)
            
        # 依次通过每个VSS块
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class VSSM_Fusion(nn.Module):
    """
    VSSM融合模型，用于双模态图像处理
    """
    def __init__(self, patch_size=4, in_chans=1, num_classes=1000, depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        # 两个独立的补丁嵌入层，分别处理两种模态
        self.patch_embed1 = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                         norm_layer=norm_layer if patch_norm else None)
        self.patch_embed2 = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                         norm_layer=norm_layer if patch_norm else None)

        # 绝对位置嵌入 (默认不使用)
        self.ape = False
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed1 = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            self.absolute_pos_embed2 = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度衰减规则
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        # 编码器层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):  # VSS Block
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # 解码器层
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)

        # 最终上采样和输出层
        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4, 1, 1)
        
        # 跨模态融合块
        self.Cross_block = nn.ModuleList()
        for cross_layer in range(self.num_layers):  # VSS Block
            clayer = MobileMambaCrossAdapter(
                dim=dims[cross_layer],
                drop_path=drop_rate,
                norm_layer=norm_layer,
                d_state=d_state,
            )
            self.Cross_block.append(clayer)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        权重初始化函数
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features_1(self, x):
        """
        第一种模态的特征提取
        """
        skip_list = []
        x = self.patch_embed1(x)
        if self.ape:
            x = x + self.absolute_pos_embed1
        x = self.pos_drop(x)

        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
        return x, skip_list

    def forward_features_2(self, x):
        """
        第二种模态的特征提取
        """
        skip_list = []
        x = self.patch_embed2(x)
        if self.ape:
            x = x + self.absolute_pos_embed2
        x = self.pos_drop(x)

        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
        return x, skip_list

    def Fusion_network(self, skip_list1, skip_list2):
        """
        融合网络，将两种模态的特征进行融合
        """
        fused_skip_list = []
        for Cross_layer, skip1, skip2 in zip(self.Cross_block, skip_list1, skip_list2):
            fused_skip = Cross_layer(skip1, skip2)
            fused_skip_list.append(fused_skip)
        return fused_skip_list

    def forward_features_up(self, x, skip_list):
        """
        上采样特征提取
        """
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skip_list[-inx])

        return x

    def forward_final(self, x):
        """
        最终输出处理
        """
        x = self.final_up(x)
        x = x.permute(0, 3, 1, 2)
        x = self.final_conv(x)
        return x

    def forward_backbone(self, x):
        """
        主干网络前向传播
        """
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x1, x2):
        """
        主前向传播函数
        """
        x_1 = x1  # 保存原始输入用于最终残差连接
        x_2 = x2  # 保存原始输入用于最终残差连接
        
        # 分别提取两种模态的特征
        x1, skip_list1 = self.forward_features_1(x1)
        x2, skip_list2 = self.forward_features_2(x2)
        
        # 特征融合
        x = x1 + x2  # 初始融合
        skip_list = self.Fusion_network(skip_list1, skip_list2)  # 跳跃连接融合

        # 上采样处理
        x = self.forward_features_up(x, skip_list)
        # 最终输出 (包含残差连接)
        x = self.forward_final(x) + x_1 + x_2 #+ x_1 + x_2

        return x