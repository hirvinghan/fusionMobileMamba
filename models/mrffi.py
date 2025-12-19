import torch
import torch.nn as nn
from functools import partial


try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    print("Warning: mamba_ssm not found. Please install it first.")
    Mamba = None

class MK_DeConv(nn.Module):
    """
    MobileMamba 中的 Multi-Kernel Depthwise Convolution (局部特征提取)
    使用不同大小的卷积核来捕获多尺度的局部信息
    """
    def __init__(self, dim, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim, bias=False)
            for k in kernel_sizes
        ])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, len(kernel_sizes) * dim, bias=False),
            nn.Softmax(dim=2)
        )
        self.num_kernels = len(kernel_sizes)
        self.dim = dim

    def forward(self, x):
        # x: [B, C, H, W]
        outputs = [conv(x) for conv in self.convs] # List of [B, C, H, W]
        outputs = torch.stack(outputs, dim=1) # [B, K, C, H, W]
        
        # 动态加权 (SKNet 思想)
        b, k, c, h, w = outputs.shape
        w_att = self.avg_pool(sum(outputs).flatten(2).mean(2)).view(b, c)
        w_att = self.fc(w_att).view(b, c, k, 1).permute(0, 2, 1, 3) # [B, K, C, 1]
        
        # 加权求和
        out = (outputs * w_att[..., None]).sum(dim=1)
        return out

class MRFFI(nn.Module):
    """
    Multi-Receptive Field Feature Interaction (MRFFI) Module
    核心逻辑：Channel Split -> (Global Mamba + Local Conv + Identity) -> Concat
    """
    def __init__(
        self, 
        dim, 
        d_state=16, 
        d_conv=4, 
        expand=2, 
        ratio_global=0.6, # Mamba 分得的通道比例 (论文中 ξ)
        ratio_local=0.3,  # Conv 分得的通道比例 (论文中 µ)
        # 剩下 0.1 是 Identity
    ):
        super().__init__()
        self.dim = dim
        self.dim_global = int(dim * ratio_global)
        self.dim_local = int(dim * ratio_local)
        self.dim_identity = dim - self.dim_global - self.dim_local

        # 1. 全局分支 (Mamba)
        # 这里使用标准的 Mamba 模块，或者你可以替换成 FusionMamba 里的 SS2D
        self.global_mamba = Mamba(
            d_model=self.dim_global,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.norm_global = nn.LayerNorm(self.dim_global)

        # 2. 局部分支 (Multi-Kernel Conv)
        self.local_conv = MK_DeConv(self.dim_local)
        self.norm_local = nn.BatchNorm2d(self.dim_local)

        # 3. 融合层
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x 形状可能是 [B, H, W, C] (Mamba习惯) 或 [B, C, H, W] (Conv习惯)
        # 我们统一先处理成 [B, H, W, C] 进行切分
        
        if x.dim() == 4 and x.shape[1] == self.dim: # [B, C, H, W]
            x = x.permute(0, 2, 3, 1) # -> [B, H, W, C]
        
        B, H, W, C = x.shape
        
        # === Step 1: Split ===
        x_global = x[:, :, :, :self.dim_global]
        x_local = x[:, :, :, self.dim_global:self.dim_global+self.dim_local]
        x_identity = x[:, :, :, self.dim_global+self.dim_local:]

        # === Step 2: Global Processing (Mamba) ===
        # Mamba 需要 [B, L, C] 输入
        x_g_flat = x_global.view(B, -1, self.dim_global)
        x_g_out = self.global_mamba(self.norm_global(x_g_flat))
        x_g_out = x_g_out.view(B, H, W, self.dim_global)

        # === Step 3: Local Processing (Conv) ===
        # Conv 需要 [B, C, H, W]
        x_l_perm = x_local.permute(0, 3, 1, 2)
        x_l_out = self.local_conv(x_l_perm)
        x_l_out = self.norm_local(x_l_out).permute(0, 2, 3, 1) # -> [B, H, W, C]

        # === Step 4: Merge ===
        x_out = torch.cat([x_g_out, x_l_out, x_identity], dim=-1)
        x_out = self.proj(x_out)

        return x_out