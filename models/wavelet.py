import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import itertools
from timm.models.layers import SqueezeExcite, DropPath

# 必须确保 ss2d_compat.py 存在
from .ss2d_compat import SS2D 

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    """
    创建小波变换滤波器 (完全保留你原本的函数)
    """
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
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()

        self.add_module('c', torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

class MobileMambaBlock(nn.Module):
    """
    基于 wavelet.py 现有组件构建的 MobileMambaBlock。
    专注于处理 [Batch, Channel, Height, Width] 格式的数据。
    """
    def __init__(self, dim, d_state=16, d_conv=3, expand=2, drop_path=0., ratio=0.6):
        super().__init__()
        self.dim = dim
        self.dim_main = int(dim * ratio)       # 分配给小波+Mamba的通道数
        self.dim_identity = dim - self.dim_main # 保持原样的通道数

        # 1. 创建小波滤波器 (复用你文件里的 create_wavelet_filter 函数)
        # 注意：这里我们让输入输出通道数一致，方便重构
        self.weight_dec, self.weight_rec = create_wavelet_filter('db1', self.dim_main, self.dim_main, torch.float)
        
        # 将滤波器注册为 buffer，确保随模型移动到 GPU
        self.register_buffer('dec_filters', self.weight_dec)
        self.register_buffer('rec_filters', self.weight_rec)

        # 2. 低频分支: Mamba (SS2D)
        # SS2D 期望输入 [B, H, W, C]，所以 forward 里需要短暂转置
        self.vssm_encoder = SS2D(
            d_model=self.dim_main, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )
        self.norm_ll = nn.LayerNorm(self.dim_main)

        # 3. 高频分支: 卷积 + 注意力 (复用 Conv2d_BN 和 SqueezeExcite)
        # 高频有3个子带，所以通道数是 3 * dim_main
        hid_dim = self.dim_main * 3
        self.local_perception = nn.Sequential(
            Conv2d_BN(hid_dim, hid_dim, 3, 1, 1, groups=hid_dim),
            nn.ReLU(),
            Conv2d_BN(hid_dim, hid_dim, 3, 1, 1, groups=hid_dim),
            SqueezeExcite(hid_dim, rd_ratio=0.25),
            Conv2d_BN(hid_dim, hid_dim, 1, 1, 0)
        )

        # 4. 融合投影
        self.proj = Conv2d_BN(dim, dim, 1, 1, 0)
        
        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # 输入 x 必须是: [B, C, H, W]
        identity = x
        
        # 1. 通道切分
        x_main = x[:, :self.dim_main, :, :]
        x_id = x[:, self.dim_main:, :, :]

        # 记录原始尺寸，处理奇数尺寸丢失问题
        H_origin, W_origin = x_main.shape[2], x_main.shape[3]

        # 2. 小波分解 (DWT)
        # 直接使用 F.conv2d 调用 self.dec_filters
        x_dwt = F.conv2d(x_main, self.dec_filters, stride=2, groups=self.dim_main)
        
        # 重塑以分离 LL 和 高频
        B, _, H, W = x_dwt.shape
        x_dwt = x_dwt.view(B, self.dim_main, 4, H, W)
        x_ll = x_dwt[:, :, 0, :, :] # LL 子带
        x_high = x_dwt[:, :, 1:, :, :].reshape(B, self.dim_main * 3, H, W) # High 子带

        # 3. 处理低频 (Mamba)
        # 需要维度转换: [B, C, H, W] -> [B, H, W, C]
        x_ll_perm = x_ll.permute(0, 2, 3, 1).contiguous()
        x_ll_perm = self.norm_ll(x_ll_perm)
        x_ll_perm = self.vssm_encoder(x_ll_perm) 
        x_ll = x_ll_perm.permute(0, 3, 1, 2).contiguous() # 转回 [B, C, H, W]

        # 4. 处理高频 (Conv)
        x_high = self.local_perception(x_high)

        # 5. 小波重构 (IDWT)
        # 准备逆卷积输入: [B, 4*C, H, W]
        x_ll_unsqueezed = x_ll.unsqueeze(2)
        x_high_reshaped = x_high.view(B, self.dim_main, 3, H, W)
        x_rec_in = torch.cat([x_ll_unsqueezed, x_high_reshaped], dim=2).flatten(1, 2)
        
        x_main_rec = F.conv_transpose2d(x_rec_in, self.rec_filters, stride=2, groups=self.dim_main)

        # === 核心修复：处理奇数尺寸丢失问题 ===
        # 如果重构后的尺寸和原始的不一样，强制插值回去
        if x_main_rec.shape[2:] != (H_origin, W_origin):
            x_main_rec = F.interpolate(
                x_main_rec, 
                size=(H_origin, W_origin), 
                mode='bilinear', 
                align_corners=False
            )

        # 6. 合并与残差
        x_out = torch.cat([x_main_rec, x_id], dim=1)
        x_out = self.proj(x_out)
        
        return identity + self.drop_path(x_out)