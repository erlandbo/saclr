# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torchvision

import random


parser = argparse.ArgumentParser(description='SACLR SSL pretrain for ImageNet')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--base_lr', default=1.2, type=float)
parser.add_argument('--weight_decay', default=1.0e-4, type=float)
parser.add_argument('--lr_scale', default="squareroot", type=str, choices=["no_scale", "linear","squareroot"])
parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--num_workers', default=20, type=int)

parser.add_argument('--method', default="saclr-1", type=str, choices=["saclr-1", "saclr-all", "simclr", "saclr-1-cosine", "saclr-all-cosine", "saclr-all-batchmix-cosine", "saclr-all-batchmix"])
parser.add_argument('--rho', default=0.99, type=float)
parser.add_argument('--alpha', default=0.125, type=float)
parser.add_argument('--s_init_t', default=2.0, type=float)
parser.add_argument('--single_s', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--temp', default=0.5, type=float)
parser.add_argument('--bmix_rate', default=0.5, type=float)

parser.add_argument('--data_path', default="~/Datasets/imagenet/", type=str)

parser.add_argument('--savedir', type=str, default='logs/')
parser.add_argument('--checkpoint_interval', default=1000, type=int)
parser.add_argument('--print_freq', default=100, type=int)
parser.add_argument('--random_state', default=None, type=int)

parser.add_argument('--master_port', default="12355", type=str)
parser.add_argument('--resume_dir', type=str, default="")


parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--nmb_crops", type=int, default=[8], nargs="+",
                    help="list of number of crops (example: [2, 8])")


parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
    help="""Scale range of the cropped image before resizing, relatively to the origin image.
    Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
    recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
    local views to generate. Set this parameter to 0 to disable multi-crop training.
    When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
    help="""Scale range of the cropped image before resizing, relatively to the origin image.
    Used for small local view cropping of multi-crop.""")



def main():
    args = parser.parse_args()
    args.world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(args,), nprocs=args.world_size)


def main_worker(gpu, args):
    rank = gpu
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.master_port #"12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=args.world_size)

    torch.backends.cuda.matmul.allow_tf32 = True # to allow tf32 with matmul
    torch.backends.cudnn.allow_tf32 = True # to allow tf32 with cudnn

    if args.random_state is not None:
        torch.manual_seed(args.random_state)
        torch.cuda.manual_seed_all(args.random_state)
        np.random.seed(args.random_state)
        random.seed(args.random_state)
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print("Fixed random_state for reproducibility")


        # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )

    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    dataset = ImageNetIndex(dataset)

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")


    args.N = len(dataset)

    if args.batch_size % args.world_size != 0:
        raise ValueError("batchsize: {} must divide number of GPUs: {}".format(args.batch_size, args.world_size)) 

    
    if args.lr_scale == "linear":
        args.lr = args.base_lr * args.batch_size / 256.0
    elif args.lr_scale == "squareroot":
        args.lr = 0.075 * math.sqrt(args.batch_size)
    elif args.lr_scale == "no_scale":
        args.lr = args.base_lr
    else:
        raise ValueError("Unknown learning rate scale: {}".format(args.lr_scale))

    args.s_init = args.N**(-2.0) * 10**args.s_init_t

    if rank == 0:
        os.makedirs(args.savedir, exist_ok=True)

        with open(Path(args.savedir)/ "logs.txt", "w") as f:
            f.write(json.dumps(args.__dict__))
    
        print(args)

    backbone = torchvision.models.resnet50()
    backbone.fc = nn.Identity()

    model = MultiCropWrapper(backbone).cuda()

    if args.resume_dir != "" and (Path(args.resume_dir) / "checkpoint.pth").is_file():
        resume_ckpt = torch.load(Path(args.resume_dir) / 'checkpoint.pth')
        model.load_state_dict(resume_ckpt['model'])
        print("Loaded weights from {}".format(args.resume_dir))


    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[gpu])
    
    if args.method == "saclr-1-cosine":
        criterion = SACLR1Cosine(
            N=args.N,
            rho=args.rho,
            alpha=args.alpha,
            s_init=args.s_init,
            single_s=args.single_s,
            temp=args.temp
        )
    elif args.method == "saclr-all-cosine":
        criterion = SACLRAllCosine(
            N=args.N,
            rho=args.rho,
            alpha=args.alpha,
            s_init=args.s_init,
            single_s=args.single_s,
            temp=args.temp
        )
    elif args.method == "saclr-all-batchmix-cosine":
        criterion = SACLRAllBatchMixCosine(
            N=args.N,
            rho=args.rho,
            alpha=args.alpha,
            s_init=args.s_init,
            single_s=args.single_s,
            temp=args.temp,
            bmix_rate=args.bmix_rate
        )
    else:
        raise ValueError("Invalid method criterion", args.method)
    
    criterion.cuda()

    #optimizer = LARS(model.parameters(), lr=0.0, weight_decay=args.weight_decay, momentum=0.9)
    optimizer = LARS(model.parameters(), lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    
    torch.backends.cudnn.benchmark = True

    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 1
    start_time = time.time()

    if args.resume_dir != "" and (Path(args.resume_dir) / "checkpoint.pth").is_file():
        resume_ckpt = torch.load(Path(args.resume_dir) / 'checkpoint.pth')
        start_epoch = resume_ckpt['epoch'] + 1
        # model.module.load_state_dict(resume_ckpt['model'])
        optimizer.load_state_dict(resume_ckpt['optimizer'])
        criterion.load_state_dict(resume_ckpt['criterion'])
        scaler.load_state_dict(resume_ckpt['scaler'])
        if rank==0:
            print("Loaded resume checkpoint from {}".format(args.resume_dir))
            for name, buffer in criterion.named_buffers(): print(name, buffer)
            print( (criterion.s_inv / criterion.N**2).mean().item() if "saclr" in args.method else 0.0)
            print(criterion.temp)


    print("Starting training !")

    for epoch in range(start_epoch, args.epochs + 1):

        data_loader.sampler.set_epoch(epoch)

        model.train()

        for batch_idx, batch in enumerate(data_loader):
            
            optimizer.zero_grad(set_to_none=True)

            images, target, idx = batch

            images = [im.cuda(non_blocking=True) for im in images]

            idx = idx.cuda(non_blocking=True)

            iter = (epoch - 1) * len(data_loader) + batch_idx
            adjust_learning_rate(step=iter, len_loader=len(data_loader), optimizer=optimizer, args=args)

            with torch.amp.autocast(device_type="cuda", enabled=True):
                outputs = model(images)

                bs = idx.size(0)

                # ============ swav loss ... ============
                loss = 0
                
                for i, crop_id in enumerate(args.crops_for_assign):
                    feats_a = outputs[bs * crop_id: bs * (crop_id + 1)]

                    subloss = 0.0
                    for v in np.delete(np.arange(args.local_crops_number + 2), crop_id):
                        feats_b = outputs[bs * v: bs * (v + 1)]
                        subloss += criterion(feats_a, feats_b, idx, update_s_value=True if v==1 and crop_id==0 else False)

                    loss += subloss / (args.local_crops_number + 1)

                loss /= len(args.crops_for_assign)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if not math.isfinite(loss.item()):
                print("Break training infinity value in loss")
                raise Exception(f'Loss is NaN')  # This could be exchanged for exit(1) if you do not want a traceback
            
            if rank ==0 and iter % args.print_freq == 0:
                train_stats = {
                    "iter": iter,
                    "epoch": epoch-1,
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]['lr'],
                    "Zhat": 0.0,
                    "time": time.time() - start_time,
                }
                with open(Path(args.savedir) / "logs.txt", "a") as f:
                    f.write(json.dumps(train_stats))

                print(" ".join(("{key}:{val}".format(key=key, val=val)) for key, val in train_stats.items()))

        if rank==0 and epoch % args.checkpoint_interval == 0:
            # Save checkpoint
            torch.save({
                "model": model.module.state_dict(),
                "criterion": criterion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
            }, Path(args.savedir) / "checkpoint.pth")

    if rank == 0:
        # Save last checkpoint
        torch.save(model.module.backbone.state_dict(), Path(args.savedir) / "resnet50_last.pth")
    
    destroy_process_group()



from PIL import ImageFilter, ImageOps


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img



class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops



class ImageNetIndex(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        images, targets = self.dataset.__getitem__(index)
        return images, targets, index


class SACLR1Cosine(nn.Module):
    def __init__(self, N, rho, alpha, s_init, single_s, temp):
        super(SACLR1Cosine, self).__init__()
        self.register_buffer("s_inv", torch.zeros(1 if single_s else N, ) + 1.0 / s_init)
        self.register_buffer("N", torch.zeros(1, ) + N)
        self.register_buffer("rho", torch.zeros(1, ) + rho)
        self.register_buffer("alpha", torch.zeros(1, ) + alpha)
        self.single_s = single_s
        self.temp = temp

    @torch.no_grad()
    def update_s(self, q_attr_a, q_attr_b, q_rep_a, q_rep_b, feats_idx):
        B = q_attr_a.size(0)

        E_attr_a = q_attr_a 
        E_attr_b = q_attr_b 

        E_rep_a = q_rep_a 
        E_rep_b = q_rep_b 

        if self.single_s:
            E_attr_a = torch.sum(E_attr_a) / B  
            E_attr_b = torch.sum(E_attr_b) / B  
            E_rep_a = torch.sum(E_rep_a) / B  
            E_rep_b = torch.sum(E_rep_b) / B 

        xi_div_omega_a = self.alpha * E_attr_a + (1.0 - self.alpha) * E_rep_a  
        xi_div_omega_b = self.alpha * E_attr_b + (1.0 - self.alpha) * E_rep_b  

        s_inv_a = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(2) * xi_div_omega_a  
        s_inv_b = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(2) * xi_div_omega_b  

        self.s_inv[feats_idx] = (s_inv_a + s_inv_b) / 2.0


    def forward(self, feats_a, feats_b, feats_idx, update_s_value=False):
        B = feats_a.shape[0]
        feats_a = F.normalize(feats_a, p=2, dim=1) 
        feats_b = F.normalize(feats_b, p=2, dim=1)

        feats_a_large = gather(feats_a) #torch.cat(all_gather_layer.apply(feats_a), 0)
        feats_b_large = gather(feats_b) #torch.cat(all_gather_layer.apply(feats_b), 0)
        feats_idx_large = concat_all_gather_no_grad(feats_idx) #torch.cat(all_gather_layer.apply(feats_b), 0)
        enlarged_B = feats_a_large.shape[0]
            

        q_attr_a = torch.exp( torch.sum(feats_a_large * feats_b_large, dim=1) / self.temp )  
        q_attr_b = torch.exp( torch.sum(feats_b_large * feats_a_large, dim=1) / self.temp )  

        attractive_forces_a = - torch.log(q_attr_a)
        attractive_forces_b = - torch.log(q_attr_b)
        
        neg_idxs = torch.roll(torch.arange(enlarged_B), shifts=-1, dims=0)
        q_rep_a = torch.exp( torch.sum(feats_a_large * feats_b_large[neg_idxs], dim=1) / self.temp)  
        q_rep_b = torch.exp( torch.sum(feats_b_large * feats_a_large[neg_idxs], dim=1) / self.temp)  

        if self.single_s:
            feats_idx_large = 0

        Z_hat = self.s_inv[feats_idx_large] / self.N.pow(2)

        repulsive_forces_a = q_rep_a / Z_hat
        repulsive_forces_b = q_rep_b / Z_hat

        loss_a = attractive_forces_a.mean() + repulsive_forces_a.mean()
        loss_b = attractive_forces_b.mean() + repulsive_forces_b.mean()
        loss = (loss_a + loss_b) / 2.0

        if update_s_value:
            self.update_s(q_attr_a.detach(), q_attr_b.detach(), q_rep_a.detach(), q_rep_b.detach(), feats_idx_large)

        return loss




class SACLRAllCosine(nn.Module):
    def __init__(self, N, rho, alpha, s_init, single_s, temp):
        super(SACLRAllCosine, self).__init__()
        self.register_buffer("s_inv", torch.zeros(1 if single_s else N, ) + 1.0 / s_init)
        self.register_buffer("N", torch.zeros(1, ) + N)
        self.register_buffer("rho", torch.zeros(1, ) + rho)
        self.register_buffer("alpha", torch.zeros(1, ) + alpha)
        self.temp = temp
        self.single_s = single_s
        print("new version")

    @torch.no_grad()
    def update_s(self, q_attr_a, q_attr_b, q_rep_a, q_rep_b, feats_idx):
        B = q_attr_a.size(0)
        enlarged_B = q_rep_a.size(1)

        E_attr_a = q_attr_a  
        E_attr_b = q_attr_b  

        E_rep_a = torch.sum(q_rep_a, dim=1) / (2.0 * enlarged_B - 2.0)  
        E_rep_b = torch.sum(q_rep_b, dim=1) / (2.0 * enlarged_B - 2.0)  
        if self.single_s:
            E_attr_a = torch.sum(E_attr_a) / B  
            E_attr_b = torch.sum(E_attr_b) / B  
            E_rep_a = torch.sum(E_rep_a) / B  
            E_rep_b = torch.sum(E_rep_b) / B  

        xi_div_omega_a = self.alpha * E_attr_a + (1.0 - self.alpha) * E_rep_a  
        xi_div_omega_b = self.alpha * E_attr_b + (1.0 - self.alpha) * E_rep_b  

        s_inv_a = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(2) * xi_div_omega_a 
        s_inv_b = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(2) * xi_div_omega_b  

        self.s_inv[feats_idx] = (s_inv_a + s_inv_b) / 2.0


    def forward(self, feats_a, feats_b, feats_idx, update_s_value=False):
        LARGE_NUM = 1e9
        B = feats_a.shape[0]
        feats_a = F.normalize(feats_a, dim=1, p=2)    
        feats_b = F.normalize(feats_b, dim=1, p=2)    
            

        feats_a_large = gather(feats_a) #torch.cat(all_gather_layer.apply(feats_a), 0)
        feats_b_large = gather(feats_b) #torch.cat(all_gather_layer.apply(feats_b), 0)
        feats_idx_large = concat_all_gather_no_grad(feats_idx) #torch.cat(all_gather_layer.apply(feats_b), 0)
        enlarged_B = feats_a_large.shape[0]

        labels_idx = torch.arange(enlarged_B, device=feats_a.device)

        masks = F.one_hot(labels_idx)

        logits_aa = torch.matmul(feats_a_large, feats_a_large.T)
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(feats_b_large, feats_b_large.T)
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(feats_a_large, feats_b_large.T)
        logits_ba = torch.matmul(feats_b_large, feats_a_large.T)

        logits_pos_a = torch.diag(logits_ab)
        logits_pos_b = torch.diag(logits_ba)

        logits_ab = logits_ab - masks * LARGE_NUM
        logits_ba = logits_ba - masks * LARGE_NUM

        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)
        
        q_attr_a = torch.exp(logits_pos_a / self.temp)  # (B,)
        q_attr_b = torch.exp(logits_pos_b / self.temp)  # (B,)

        attractive_forces_a = - torch.log(q_attr_a)
        attractive_forces_b = - torch.log(q_attr_b)

        q_rep_a = torch.exp(logits_a / self.temp)  # (B,2*B_enlarged)
        q_rep_b = torch.exp(logits_b / self.temp)  # (B,2*B_enlarged)

        if self.single_s:
            feats_idx_large = 0

        Z_hat = ( self.s_inv[feats_idx_large] / self.N.pow(2) ) * ( 2 * enlarged_B - 2) 

        repulsive_forces_a = torch.sum(q_rep_a / Z_hat.detach().view(-1,1), dim=1) 
        repulsive_forces_b = torch.sum(q_rep_b / Z_hat.detach().view(-1,1), dim=1) 

        loss_a = attractive_forces_a.mean() + repulsive_forces_a.mean()
        loss_b = attractive_forces_b.mean() + repulsive_forces_b.mean()
        loss = (loss_a + loss_b) / 2.0

        if update_s_value:

            self.update_s(q_attr_a.detach(), q_attr_b.detach(), q_rep_a.detach(), q_rep_b.detach(), feats_idx_large)

        return loss




class SACLRAllBatchMixCosine(nn.Module):
    def __init__(self, N, rho, alpha, s_init, single_s, temp, bmix_rate):
        super(SACLRAllBatchMixCosine, self).__init__()
        self.register_buffer("s_inv", torch.zeros(1 if single_s else N, ) + 1.0 / s_init)
        self.register_buffer("N", torch.zeros(1, ) + N)
        self.register_buffer("rho", torch.zeros(1, ) + rho)
        self.register_buffer("alpha", torch.zeros(1, ) + alpha)
        self.temp = temp
        self.single_s = single_s
        self.bmix_rate = bmix_rate

    @torch.no_grad()
    def update_s(self, q_attr_a, q_attr_b, q_rep_a, q_rep_b, feats_idx):
        B = q_attr_a.size(0)
        enlarged_B = q_rep_a.size(1)

        E_attr_a = q_attr_a  
        E_attr_b = q_attr_b  

        E_rep_a = torch.sum(q_rep_a, dim=1) / (2.0 * enlarged_B - 2.0)  
        E_rep_b = torch.sum(q_rep_b, dim=1) / (2.0 * enlarged_B - 2.0)  
        if self.single_s:
            E_attr_a = torch.sum(E_attr_a) / B  
            E_attr_b = torch.sum(E_attr_b) / B  
            E_rep_a = torch.sum(E_rep_a) / B  
            E_rep_b = torch.sum(E_rep_b) / B  

        xi_div_omega_a = self.alpha * E_attr_a + (1.0 - self.alpha) * E_rep_a  
        xi_div_omega_b = self.alpha * E_attr_b + (1.0 - self.alpha) * E_rep_b  

        s_inv_a = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(2) * xi_div_omega_a 
        s_inv_b = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(2) * xi_div_omega_b  

        self.s_inv[feats_idx] = (s_inv_a + s_inv_b) / 2.0


    def forward(self, feats_a, feats_b, feats_idx, update_s_value=False):
        LARGE_NUM = 1e9
        B = feats_a.shape[0]
        feats_a = F.normalize(feats_a, dim=1, p=2)    
        feats_b = F.normalize(feats_b, dim=1, p=2)    
            
        feats_a_large = gather(feats_a) #torch.cat(all_gather_layer.apply(feats_a), 0)
        feats_b_large = gather(feats_b) #torch.cat(all_gather_layer.apply(feats_b), 0)
        feats_idx_large = concat_all_gather_no_grad(feats_idx) #torch.cat(all_gather_layer.apply(feats_b), 0)
        enlarged_B = feats_a_large.shape[0]

        labels_idx = torch.arange(enlarged_B, device=feats_a.device)

        masks = F.one_hot(labels_idx)

        logits_aa = torch.matmul(feats_a_large, feats_a_large.T)
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(feats_b_large, feats_b_large.T)
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(feats_a_large, feats_b_large.T)
        logits_ba = torch.matmul(feats_b_large, feats_a_large.T)

        logits_pos_a = torch.diag(logits_ab)
        logits_pos_b = torch.diag(logits_ba)

        logits_ab = logits_ab - masks * LARGE_NUM
        logits_ba = logits_ba - masks * LARGE_NUM

        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)
        
        q_attr_a = torch.exp(logits_pos_a / self.temp)  # (B,)
        q_attr_b = torch.exp(logits_pos_b / self.temp)  # (B,)

        attractive_forces_a = - torch.log(q_attr_a)
        attractive_forces_b = - torch.log(q_attr_b)

        q_rep_a = torch.exp(logits_a / self.temp)  # (B,2*B_enlarged)
        q_rep_b = torch.exp(logits_b / self.temp)  # (B,2*B_enlarged)

        if self.single_s:
            feats_idx_large = 0

        #Z_hat = ( self.s_inv[feats_idx_large] / self.N.pow(2) ) * ( 2 * enlarged_B - 2)
        Z_hat = ( self.s_inv[feats_idx_large] / self.N.pow(2) ) * ( 2 * enlarged_B - 2)
        Z_hat_a = self.bmix_rate * Z_hat + (1-self.bmix_rate) * torch.sum(q_rep_a.detach(), dim=1) 
        Z_hat_b = self.bmix_rate * Z_hat + (1-self.bmix_rate) * torch.sum(q_rep_b.detach(), dim=1) 
 
        repulsive_forces_a = torch.sum(q_rep_a / Z_hat_a.detach().view(-1,1), dim=1) 
        repulsive_forces_b = torch.sum(q_rep_b / Z_hat_b.detach().view(-1,1), dim=1)

        loss_a = attractive_forces_a.mean() + repulsive_forces_a.mean()
        loss_b = attractive_forces_b.mean() + repulsive_forces_b.mean()
        loss = (loss_a + loss_b) / 2.0

        if update_s_value:
            self.update_s(q_attr_a.detach(), q_attr_b.detach(), q_rep_a.detach(), q_rep_b.detach(), feats_idx_large)

        return loss





class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(2048, 8192, bias=False),
            nn.BatchNorm1d(8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, 8192, bias=False),
            nn.BatchNorm1d(8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, 8192, bias=True)
        )

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.projector(output)


# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# https://arxiv.org/pdf/2006.07733
# https://arxiv.org/pdf/2103.03230
# https://github.com/Optimization-AI/SogCLR
# https://github.com/facebookresearch/barlowtwins/blob/main/main.py


def adjust_learning_rate(step, len_loader, optimizer, args):
    max_steps = args.epochs * len_loader
    warmup_steps = 10 * len_loader
    base_lr = args.lr # args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr 
    #optimizer.param_groups[1]['lr'] = lr



class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


# https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def gather(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)



@torch.no_grad()
def concat_all_gather_no_grad(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output   


if __name__ == "__main__":
    main()
