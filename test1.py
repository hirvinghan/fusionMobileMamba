import cv2
import numpy as np
import os
import torch
import time
import torch.nn.functional as F
from collections import OrderedDict
from models.vmamba_Fusion_efficross import VSSM_Fusion as net

# 1. 设置环境
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 2. 路径配置 (请修改为你实际的路径)
model_path = "model_mobile_mamba/my_cross/fusion_model_best.pth"
input_folder_ir = './dataset/RoadScene-master/cropinfrared'
input_folder_vis = './dataset/RoadScene-master/crop_LR_visible'
output_folder = './outputs_RoadScene_mobilemamba'

def load_model(model, model_path):
    print(f"正在加载权重: {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') if 'module.' in k else k
        new_state_dict[name] = v
    try:
        model.load_state_dict(new_state_dict)
        print("✅ 权重加载成功！")
    except Exception as e:
        print(f"⚠️ 尝试非严格加载: {e}")
        model.load_state_dict(new_state_dict, strict=False)
    return model

# 初始化模型
model = net().to(device)
if os.path.exists(model_path):
    model = load_model(model, model_path)
else:
    print(f"❌ 错误: 找不到权重文件 {model_path}")
    exit()

model.eval()

def get_image_files(input_folder):
    valid_extensions = (".bmp", ".tif", ".jpg", ".jpeg", ".png")
    return sorted([f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)])

def fusion(input_folder_ir, input_folder_vis, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ir_images = get_image_files(input_folder_ir)
    vis_images = get_image_files(input_folder_vis)
    print(f"开始处理 {len(ir_images)} 对图像...")

    tic = time.time()
    for ir_name, vis_name in zip(ir_images, vis_images):
        path1 = os.path.join(input_folder_ir, ir_name)
        path2 = os.path.join(input_folder_vis, vis_name)

        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

        # 1. 尺寸对齐
        h, w = img1.shape
        if img2.shape != (h, w):
            img2 = cv2.resize(img2, (w, h))

        # 2. 归一化 [0, 1]
        img1_norm = img1.astype(np.float32) / 255.0
        img2_norm = img2.astype(np.float32) / 255.0
        
        input1 = torch.from_numpy(img1_norm).unsqueeze(0).unsqueeze(0).to(device)
        input2 = torch.from_numpy(img2_norm).unsqueeze(0).unsqueeze(0).to(device)

        # 3. Padding (改为 replicate 模式，减少边缘网格效应)
        factor = 32
        h_x, w_x = input1.shape[2], input1.shape[3]
        pad_h = (factor - h_x % factor) % factor
        pad_w = (factor - w_x % factor) % factor
        
        if pad_h != 0 or pad_w != 0:
            # 改用 replicate (复制边缘)，比 reflect (反射) 更少伪影
            input1 = F.pad(input1, (0, pad_w, 0, pad_h), mode='replicate')
            input2 = F.pad(input2, (0, pad_w, 0, pad_h), mode='replicate')

        with torch.no_grad():
            # 推理
            out = model(input2, input1)
            
            # Crop 回原尺寸
            out = out[:, :, :h_x, :w_x]
            out_np = out.squeeze().cpu().numpy()
            
            # ==========================================
            # 【终极修复】鲁棒百分位归一化 (Robust Normalization)
            # ==========================================
            # 1. 计算 2% 和 98% 的分位数 (掐头去尾)
            # 这能自动忽略掉那些极亮/极暗的网格噪点
            p_min = np.percentile(out_np, 2)
            p_max = np.percentile(out_np, 98)
            
            # 2. 截断异常值
            out_np = np.clip(out_np, p_min, p_max)
            
            # 3. 线性拉伸到 [0, 1]
            # 加上 1e-6 防止分母为 0
            out_np = (out_np - p_min) / (p_max - p_min + 1e-6)
            
            # 4. (可选) 伽马校正：如果人物还是偏暗，可以把 1.0 改成 0.7 提亮暗部
            # out_np = np.power(out_np, 1.0/1.0) 

        # 保存
        result = (out_np * 255).astype(np.uint8)
        
        # (可选) 如果网格依然明显，可以加一个极轻微的中值滤波去噪
        # result = cv2.medianBlur(result, 3) 
        
        cv2.imwrite(os.path.join(output_folder, ir_name), result)
        print(f"Processed: {ir_name}")

    toc = time.time()
    print(f'✅ 全部完成！耗时: {toc - tic:.2f}s')

if __name__ == '__main__':
    fusion(input_folder_ir,input_folder_vis,output_folder)