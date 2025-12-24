import cv2
import numpy as np
import os
import torch
import time
from PIL import Image
import torch.nn.functional as F
from models.vmamba_Fusion_efficross import VSSM_Fusion as net

# 1. 设置显卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 2. 加载模型
model = net(in_channel=1)
model_path = "/home/jicheng/Mamba/FusionMamba_Mobile/model_mobile_mamba_coif1/my_cross/fusion_model_best.pth" 

use_gpu = torch.cuda.is_available()

if use_gpu:
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
else:
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)

model.eval()

def get_image_files(input_folder):
    valid_extensions = (".bmp", ".tif", ".jpg", ".jpeg", ".png")
    return sorted([f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)])

def fusion(input_folder_ir, input_folder_vis, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    tic = time.time()
    
    ir_images = get_image_files(input_folder_ir)
    vis_images = get_image_files(input_folder_vis)

    if len(ir_images) != len(vis_images):
        print(f"警告: 红外({len(ir_images)}) 与 可见光({len(vis_images)}) 数量不一致！")

    for ir_image_name, vis_image_name in zip(ir_images, vis_images):
        print(f"Processing: {ir_image_name} + {vis_image_name}")
        
        path1 = os.path.join(input_folder_ir, ir_image_name)
        path2 = os.path.join(input_folder_vis, vis_image_name)

        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE) # IR
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE) # VIS

        # 1. 确保 IR 和 VIS 尺寸一致 (以 IR 为准)
        h, w = img1.shape
        if img2.shape != (h, w):
            img2 = cv2.resize(img2, (w, h))

        # 2. 归一化并转 Tensor
        img1 = np.asarray(img1, dtype=np.float32) / 255.0
        img2 = np.asarray(img2, dtype=np.float32) / 255.0

        img1_tensor = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0).to(device)
        img2_tensor = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0).to(device)

        # ==========================================
        # 【关键修改】自动 Padding 到 32 的倍数
        # ==========================================
        factor = 32
        h, w = img1_tensor.shape[2], img1_tensor.shape[3]
        
        # 计算需要填充的高度和宽度
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        
        # F.pad 参数顺序: (左, 右, 上, 下)
        if pad_h != 0 or pad_w != 0:
            img1_tensor = F.pad(img1_tensor, (0, pad_w, 0, pad_h), mode='reflect')
            img2_tensor = F.pad(img2_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            # 融合 (注意这里参数顺序，如果出来的图反色或奇怪，交换这两个参数)
            out = model(img2_tensor, img1_tensor) 
            
            # 截断
            out = torch.clamp(out, 0, 1)
            
            # ==========================================
            # 【关键修改】Crop 回原始尺寸
            # ==========================================
            # 即使 pad_h/pad_w 是 0，切片操作 [:h, :w] 也是安全的
            out = out[:, :, :h, :w]
            
            out_np = out.squeeze().cpu().numpy()

        # 保存
        result = (out_np * 255).astype(np.uint8)
        output_filename = ir_image_name 
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, result)

    toc = time.time()
    print('Total processing time: {:.2f}s'.format(toc - tic))

if __name__ == '__main__':
    # 路径配置
    input_folder_ir = './dataset/RoadScene-master/cropinfrared'
    input_folder_vis = './dataset/RoadScene-master/crop_LR_visible'
    output_folder = './outputs_RoadScene_mobilemamba_mloss'

    fusion(input_folder_ir, input_folder_vis, output_folder)