import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from timm.models.layers import DropPath, SqueezeExcite

from .ss2d_compat import SS2D
def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    """构建小波变换滤波器 (保持不变)"""
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)
    
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type)
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type)
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters

class Conv2d_BN(torch.nn.Sequential):
    """基础卷积块：Conv + BN"""
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

class LocalPerception(nn.Module):
    """
    局部感知分支 (Branch 2)
    对应 MobileMamba 中的 'Local Branch'
    使用 3x3 和 5x5 卷积并行提取特征，增加局部感受野 (2n+1)
    """
    def __init__(self, dim, groups=1):
        super().__init__()
        # 3x3 卷积
        self.conv3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=groups, bias=False)
        # 5x5 卷积 (通过 padding=2 保持尺寸)
        self.conv5 = nn.Conv2d(dim, dim, 5, 1, 2, groups=groups, bias=False)
        
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # 并行计算并相加
        out = self.conv3(x) + self.conv5(x)
        return self.act(self.bn(out))

class WaveletMambaBranch(nn.Module):
    """
    长程依赖分支 (Branch 1)
    对应 MobileMamba 中的 'Global Branch' (Wavelet + Mamba + Attention)
    """
    def __init__(self, dim, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.dim = dim
        
        # 1. 小波参数
        self.weight_dec, self.weight_rec = create_wavelet_filter('db1', dim, dim, torch.float)
        self.register_buffer('dec_filters', self.weight_dec)
        self.register_buffer('rec_filters', self.weight_rec)
        
        # 2. SS2D (Mamba) 核心
        # MobileMamba 在低频部分使用 SS2D
        self.vssm_encoder = SS2D(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm_ll = nn.LayerNorm(dim)
        
        # 3. 高频注意力 (SE Module)
        # 高频有 3 个子带 (LH, HL, HH)，通道数是 3 * dim
        self.high_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 3, (dim * 3) // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d((dim * 3) // 16, dim * 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 记录原始尺寸，防止奇数丢失
        B, C, H, W = x.shape
        
        # === 1. DWT (下采样) ===
        x_dwt = F.conv2d(x, self.dec_filters, stride=2, groups=self.dim)
        # 拆分 Low-Freq (LL) 和 High-Freq (LH, HL, HH)
        x_dwt = x_dwt.view(B, self.dim, 4, H // 2, W // 2) # 注意：如果是奇数这里会报错，所以后面有插值修复逻辑
        x_ll = x_dwt[:, :, 0, :, :]
        x_high = x_dwt[:, :, 1:, :, :].reshape(B, self.dim * 3, H // 2, W // 2)
        
        # === 2. Process Low-Freq (Mamba) ===
        # SS2D 需要 BHWC 输入
        x_ll = x_ll.permute(0, 2, 3, 1).contiguous() 
        x_ll = self.norm_ll(x_ll)
        x_ll = self.vssm_encoder(x_ll) 
        x_ll = x_ll.permute(0, 3, 1, 2).contiguous() # 变回 BCHW
        
        # === 3. Process High-Freq (Attention) ===
        attn = self.high_se(x_high)
        x_high = x_high * attn
        
        # === 4. IDWT (上采样重构) ===
        x_ll = x_ll.unsqueeze(2)
        x_high = x_high.view(B, self.dim, 3, H // 2, W // 2)
        x_rec_in = torch.cat([x_ll, x_high], dim=2).flatten(1, 2)
        
        x_rec = F.conv_transpose2d(x_rec_in, self.rec_filters, stride=2, groups=self.dim)
        
        # === 5. 尺寸安全检查 (Robustness) ===
        # 如果原始输入是奇数，DWT后重构回来会少1个像素，必须插值补全
        if x_rec.shape[2:] != (H, W):
            x_rec = F.interpolate(x_rec, size=(H, W), mode='bilinear', align_corners=False)
            
        return x_rec

class MobileMambaBlock(nn.Module):
    """
    MobileMambaBlock (MRFFI 模块)
    完全遵循 MobileMamba 论文结构：
    1. Input Split -> [Mamba Branch, Local Branch, Identity Branch]
    2. Concat
    3. Projection
    """
    def __init__(self, dim, d_state=16, d_conv=3, expand=2, drop_path=0., 
                 ratio_mamba=0.6, ratio_conv=0.3): 
        super().__init__()
        
        # 1. 计算三个分支的通道数
        self.dim_mamba = int(dim * ratio_mamba)
        self.dim_conv = int(dim * ratio_conv)
        self.dim_identity = dim - self.dim_mamba - self.dim_conv
        
        # 2. 定义分支
        # Branch A: 小波 + Mamba
        self.branch_mamba = WaveletMambaBranch(self.dim_mamba, d_state, d_conv, expand)
        
        # Branch B: 多尺度局部卷积 (3x3 + 5x5)
        self.branch_conv = LocalPerception(self.dim_conv)
        
        # Branch C: Identity (直接透传，无需定义层)
        
        # 3. 融合层 (1x1 Conv)
        self.proj = Conv2d_BN(dim, dim, 1, 1, 0)
        
        # 4. DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x: [B, C, H, W]
        identity = x
        
        # Step 1: Channel Split
        # 按照 [Mamba, Conv, Identity] 的顺序切分
        c1, c2 = self.dim_mamba, self.dim_mamba + self.dim_conv
        x_mamba = x[:, :c1, :, :]
        x_conv = x[:, c1:c2, :, :]
        x_id = x[:, c2:, :, :]
        
        # Step 2: Forward Pass
        out_mamba = self.branch_mamba(x_mamba)
        out_conv = self.branch_conv(x_conv)
        out_id = x_id
        
        # Step 3: Concat & Fuse
        out = torch.cat([out_mamba, out_conv, out_id], dim=1)
        out = self.proj(out)
        
        # Step 4: Residual
        return identity + self.drop_path(out)