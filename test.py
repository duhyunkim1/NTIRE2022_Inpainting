import os
import sys

from network import UNet
from dataset import RestList
from utils import psnr, AverageMeter, Evaluation

import torch
from torch import device, nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import data_transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
import cv2
import argparse
import time

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--ckpt', default='', type=str) # weight path
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    
    parser.add_argument('--phase', default='test', type=str)
    parser.add_argument('--img_root', default = None, type = str)
    parser.add_argument('--mask_root', default = None, type = str)
    
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    
    return args


def main():
    
    args = parse_args()
    
    phase = args.phase
    
    weight = args.ckpt
    batch_size = args.batch_size
    model_G = torch.nn.DataParallel(UNet().cuda())
   
    img_dir = args.img_root
    mask_dir = args.mask_root
    
    device = torch.device("cuda")
    ckpt = torch.load(weight, map_location= device)
    dataset = img_dir.split('/')[-1]
    try:
        model_G.load_state_dict(ckpt['model_G'])
        model_G = model_G.module
    except:
        model_G = model_G.module
        model_G.load_state_dict(ckpt['model_G'])
    model_G.eval()
    
    transform = [transforms.ToTensor()]
    
    Test_loader = torch.utils.data.DataLoader(
            RestList(phase, img_dir, mask_dir,  transforms.Compose(transform), transforms.Compose(transform), batch=batch_size),
            batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        
    
    output_dir = os.getcwd()
    
    save_dir = os.path.join(output_dir, 'Test', 'submit', f'{dataset}', 'test')
    
    start = time.time()
    for i, (img, mask, name) in enumerate(Test_loader): #mask = 1
        img_var = img.float().cuda()
        mask_var = mask[0].float().cuda()
        name = name[0]
        
        _, _, h, w = img_var.size()
        img_var = F.interpolate(img_var, size=(h + 16 - h % 16 , w + 16 - w % 16), mode='bilinear')
        mask_var = F.interpolate(mask_var, size=(h + 16 - h % 16 , w + 16 - w % 16), mode='bilinear')
        # masked_img = img_var * (1-mask_var)
        
        stroke_type = name.split('/')[0]
        os.makedirs(save_dir+'/'+stroke_type, exist_ok=True)
        
        with torch.no_grad():
            result = model_G(img_var)
            output = model_G(img_var) * mask_var + img_var 
            
            output = F.interpolate(output, size=(h, w), mode='bilinear')
            
            resultname = os.path.join(save_dir, name[:-4] + '.png')
            save_image(output[0],resultname)

    print(f"runtime per frame: {(time.time() - start)/len(Test_loader):.4f}s")

if __name__ == '__main__':
    main()
