#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
import argparse
import time
import datetime
import logging
import warnings
import numpy as np
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

# 导入自定义模块
from models.vmamba_Fusion_efficross import VSSM_Fusion
from TaskFusion_dataset import Fusion_dataset
from logger import setup_logger
from loss import Fusionloss

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Training for FusionMamba')
    parser.add_argument('--model_name', '-M', type=str, default='VSSM_Fusion')
    # 物理Batch Size (单卡)，建议设为 4 或 8
    parser.add_argument('--batch_size', '-B', type=int, default=32)
    parser.add_argument('--epochs', '-E', type=int, default=10)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    parser.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision')
    
    # DDP 必须参数 (由 torchrun 自动传入，不要手动设置)
    parser.add_argument("--local_rank", type=int, default=-1)
    
    return parser.parse_args()

def setup_ddp():
    """初始化分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        dist.barrier()
        return local_rank, rank, world_size
    else:
        print("未检测到DDP环境，回退到单卡模式")
        return 0, 0, 1

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def train_fusion(args):
    # 1. DDP 初始化
    local_rank, rank, world_size = setup_ddp()
    device = torch.device("cuda", local_rank)
    
    # Logger 初始化 (只在 Rank 0)
    logger = None
    if rank == 0:
        logpath = './logs'
        if not os.path.exists(logpath):
            os.makedirs(logpath)
        setup_logger(logpath)
        logger = logging.getLogger()
        logger.info(f"启动分布式训练: World Size={world_size}, Batch Size={args.batch_size} (per GPU)")
        if args.amp:
            logger.info("混合精度训练 (AMP) 已开启 ✅")

    # 2. 模型初始化
    fusionmodel = eval('VSSM_Fusion')().to(device)
    fusionmodel = DDP(fusionmodel, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    fusionmodel.train()

    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=0.0002)
    scaler = GradScaler(enabled=args.amp)
    criteria_fusion = Fusionloss().to(device)

    # 3. 数据集加载 (修复：这里定义 train_loader)
    # ============================================================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    kaist_root = os.path.join(current_dir, 'dataset') 
    
    if rank == 0:
        logger.info(f"数据集根目录: {kaist_root}")
        if not os.path.exists(kaist_root):
            logger.error(f"严重错误: 找不到路径 {kaist_root}")
            return

    # 等待所有进程确认路径存在
    dist.barrier()
    if not os.path.exists(kaist_root):
        return

    train_dataset = Fusion_dataset('train', length=20000, root_dir=kaist_root)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )
    # ============================================================

    # 4. 训练循环
    st = glob_st = time.time()

    best_loss = 1000.0
    
    for epo in range(args.epochs):
        fusionmodel.train()
        train_sampler.set_epoch(epo) # 关键：让DDP每个epoch数据不同
        
        # 学习率调整策略
        lr_start = 0.0001
        lr_decay = 0.75
        lr_this_epo = lr_start * lr_decay ** (epo)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo

        # === Batch 循环 (Step) ===
        for it, (image_vis, image_ir) in enumerate(train_loader):
            optimizer.zero_grad() 

            image_vis = Variable(image_vis).to(device)
            image_ir = Variable(image_ir).to(device)

            with autocast(enabled=args.amp):
                fusion_image = fusionmodel(image_vis, image_ir)
                fusion_image = torch.clamp(fusion_image, 0, 1)

                loss_fusion, loss_in, ssim_loss, loss_grad = criteria_fusion(
                    image_vis=image_vis, image_ir=image_ir, generate_img=fusion_image,
                    i=epo, labels=None
                )
                loss_total = loss_fusion

            # NaN 防护：如果Loss炸了，跳过这一步，不更新权重
            if torch.isnan(loss_total) or torch.isinf(loss_total):
                if rank == 0:
                    logger.warning(f"⚠️ Warning: NaN detected at Epoch {epo} Step {it}! Skipping...")
                optimizer.zero_grad()
                continue

            scaler.scale(loss_total).backward()
            
            # 梯度裁剪 (防后期崩溃)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(fusionmodel.parameters(), max_norm=0.1) 
            
            scaler.step(optimizer)
            scaler.update()

            # 日志打印 (每10步) - 放在这里是对的
            if rank == 0:
                ed = time.time()
                t_intv, glob_t_intv = ed - st, ed - glob_st
                now_it = len(train_loader) * epo + it + 1
                eta = int((len(train_loader) * args.epochs - now_it) * (glob_t_intv / (now_it + 1e-6)))
                eta = str(datetime.timedelta(seconds=eta))
                
                if now_it % 10 == 0:
                    msg = ', '.join(
                        [
                            'Epoch: {epo}/{max_epo}',
                            'Step: {it}/{max_it}',
                            'Total: {loss:.4f}',
                            'SSIM_Loss: {ssim:.4f}', # 这里打印的是Loss，越小越好
                            'ETA: {eta}',
                            'Time: {time:.3f}s',
                        ]
                    ).format(
                        epo=epo + 1,
                        max_epo=args.epochs,
                        it=it + 1,
                        max_it=len(train_loader),
                        loss=loss_total.item(),
                        ssim=ssim_loss.item(),
                        time=t_intv,
                        eta=eta,
                    )
                    logger.info(msg)
                    st = ed
        # === Batch 循环结束 ===

        # ==========================================
        # 保存模型逻辑 (必须在这里，Batch循环之外)
        # ==========================================
        if rank == 0:
            modelpth = os.path.join('model_mobile_mamba_coif1_15101', 'my_cross')
            if not os.path.exists(modelpth):
                os.makedirs(modelpth)
            
            # 1. 保存当前 Epoch (备份)
            epoch_model_file = os.path.join(modelpth, f'checkpoint_epoch_{epo+1}.pth')
            torch.save(fusionmodel.module.state_dict(), epoch_model_file)
            logger.info(f"Checkpoint saved: {epoch_model_file}")

            # 2. 保存最佳模型
            if loss_total < best_loss:
                best_loss = loss_total
                fusion_model_file = os.path.join(modelpth, 'fusion_model_best.pth')
                torch.save(fusionmodel.module.state_dict(), fusion_model_file)
                logger.info(f"Best Model updated: {fusion_model_file} (Loss: {best_loss:.4f})")
    
    cleanup_ddp()

if __name__ == "__main__":
    args = parse_args()
    train_fusion(args)