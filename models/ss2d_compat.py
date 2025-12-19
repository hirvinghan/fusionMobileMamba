import torch
import torch.nn as nn
import math
from einops import repeat

# === 尝试导入标准 Mamba 算子 ===
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    print("错误: 未找到 mamba_ssm。请确保你已经安装了 mamba_ssm (pip install mamba-ssm)")
    selective_scan_fn = None

class SS2D(nn.Module):
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
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 1. 输入投影 (In-Projection)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # 2. 2D 卷积 (深度卷积)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # 3. 状态空间参数投影 (x -> dt, B, C)
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # 4. 初始化 S4D 参数 (A, D) - 4个方向 (K=4)
        # A_logs: [4, D, N] -> 每个方向有一套独立的 A
        self.A_logs = nn.Parameter(torch.randn(4, self.d_inner, self.d_state, **factory_kwargs)) 
        self.Ds = nn.Parameter(torch.ones(4, self.d_inner, **factory_kwargs))

        # 5. 输出投影 (Out-Projection)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x):
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        L = H * W
        
        # 1. 投影 + 卷积
        xz = self.in_proj(x) # [B, H, W, 2*D]
        x, z = xz.chunk(2, dim=-1) 

        # 2D Conv
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)
        x = self.act(x) # [B, D, H, W]

        # 2. 准备 4 个方向的数据 (Scan Preparation)
        # 为了适配 x_proj，我们先展平为 [B, L, D]
        
        # Direction 1: Forward (左上 -> 右下)
        x_hw = x.permute(0, 2, 3, 1).reshape(B, L, self.d_inner) 
        # Direction 2: Backward (右下 -> 左上)
        x_wh = torch.flip(x_hw, dims=[1])
        # Direction 3: Transpose Forward (左下 -> 右上,近似逻辑)
        x_trans = x.transpose(2, 3).permute(0, 2, 3, 1).reshape(B, L, self.d_inner)
        # Direction 4: Transpose Backward (右上 -> 左下)
        x_trans_back = torch.flip(x_trans, dims=[1])

        # 拼接: [B, 4, L, D] -> Flatten -> [B*4, L, D]
        # 这样可以并行通过 x_proj 算出所有方向的 dt, B, C
        xs = torch.stack([x_hw, x_wh, x_trans, x_trans_back], dim=1) # [B, 4, L, D]
        xs_flat = xs.view(B * 4, L, self.d_inner)

        # 3. 预测 SSM 参数 (dt, B, C)
        x_dbl = self.x_proj(xs_flat) # [B*4, L, dt_rank + 2*d_state]
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = self.dt_proj(dt) # [B*4, L, D]
        
        # 4. 执行 Selective Scan (串行循环 4 次)
        # 这一步是为了解决 A 矩阵形状不匹配的问题
        
        # 先把数据整理回 [B, 4, D, L] 方便按方向取用
        # 注意: selective_scan_fn 需要 [B, D, L] 输入
        u_all = xs.transpose(2, 3) # [B, 4, D, L]
        dt_all = dt.view(B, 4, L, self.d_inner).transpose(2, 3) # [B, 4, D, L]
        B_all = B_ssm.view(B, 4, L, self.d_state).transpose(2, 3) # [B, 4, N, L]
        C_all = C_ssm.view(B, 4, L, self.d_state).transpose(2, 3) # [B, 4, N, L]
        
        y_list = []
        for i in range(4):
            # === 核心修复逻辑 ===
            # 每次只取第 i 个方向的数据，此时 A 的形状是 [D, N]，完全符合标准算子要求
            
            # 准备参数
            u_i = u_all[:, i].contiguous()       # [B, D, L]
            dt_i = dt_all[:, i].contiguous()     # [B, D, L]
            B_i = B_all[:, i].contiguous()       # [B, N, L]
            C_i = C_all[:, i].contiguous()       # [B, N, L]
            A_i = -torch.exp(self.A_logs[i])     # [D, N] (无 Batch 维)
            D_i = self.Ds[i]                     # [D]
            
            # 调用 CUDA 算子
            y_i = selective_scan_fn(
                u_i, dt_i, A_i, B_i, C_i, D_i, 
                z=None, 
                delta_bias=None, 
                delta_softplus=True, 
                return_last_state=False
            ) # -> [B, D, L]
            
            y_list.append(y_i)

        # 5. 合并 4 个方向 (Cross Merge)
        # 还原回图像空间 [B, H, W, D]
        
        # Direction 1
        y_hw = y_list[0].transpose(1, 2).view(B, H, W, self.d_inner)
        # Direction 2 (Flip back)
        y_wh = torch.flip(y_list[1].transpose(1, 2), dims=[1]).view(B, H, W, self.d_inner)
        # Direction 3 (Transpose back)
        y_trans = y_list[2].transpose(1, 2).view(B, W, H, self.d_inner).transpose(1, 2)
        # Direction 4 (Flip & Transpose back)
        y_trans_back = torch.flip(y_list[3].transpose(1, 2), dims=[1]).view(B, W, H, self.d_inner).transpose(1, 2)

        # Sum merge (MobileMamba 逻辑)
        y_out = y_hw + y_wh + y_trans + y_trans_back
        
        # 6. 乘上 Gate (SiLU后的 z)
        y_out = y_out * self.act(z) # [B, H, W, D]

        # 7. 输出投影
        out = self.out_proj(y_out)
        out = self.dropout(out)
        
        return out