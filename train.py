import os
import time
import shutil
import sys
import logging
from datetime import datetime
from network import UNet, Discriminator
from dataset import RestList
from utils import save_checkpoint, psnr, AverageMeter, FullLoss, GANLoss, Folder_Create, Evaluation, PerceptualLoss, StyleLoss, AdversarialLoss

import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import data_transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np

def Train(loaders, models, optims, criterions, epoch, best_score, output_dir, eval_score=None, print_freq=10, logger=None):
        
    # Counters
    losses_l1 = AverageMeter()
    losses_perc = AverageMeter()
    losses_style = AverageMeter()
    losses_G = AverageMeter()
    losses_D = AverageMeter()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    score_psnr = AverageMeter()
    score_ssim = AverageMeter()
    score_lpip = AverageMeter()
    
    # Loaders, criterions, models, optimizers
    Train_loader, Val_loader = loaders
    criterion_GAN, criterion_eval, criterion_Pix = criterions    
    model_G, model_D = models
    optim_G, optim_D = optims # 

    model_G.train()
    model_D.train()

    end = time.time()
    for i, (img, mask) in enumerate(Train_loader):
        data_time.update(time.time() - end)

        ############################################################
        # (1) Load and construct data
        ############################################################

        img_var = img.float().cuda()
        mask_var = mask[0].cuda()

        masked_img = img_var * (1 - mask_var)

        ############################################################
        # (2) Feed the image to the generator
        ############################################################

        output = model_G(masked_img)
        masked_output = output * mask_var + masked_img
        
        ############################################################
        # (3) Compute loss functions
        # ##########################################################
        
        if i % 1 == 0 : 
            loss_Disc = criterion_GAN(model_D(output.detach()), False) + criterion_GAN(model_D(img_var), True)
            #print("Loss_Disc :",loss_Disc.item())
            optim_D.zero_grad()
            loss_Disc.backward()
            optim_D.step()
        
        loss_l1, loss_perc, loss_style = criterion_Pix(output, img_var, 'pixel')
        # loss_recon_masked = F.l1_loss(masked_output, img_var)
        loss_recon_GAN = criterion_GAN(model_D(output), True)        
        loss = loss_l1 + 0.1 * loss_perc + 0.001 * loss_recon_GAN + 250 * loss_style 
        # print("Loss_recon :",loss_recon.item())
        # print("Loss_recon_maksed :",loss_recon_masked.item())
        # print("Loss_recon_GAN :",loss_recon_GAN.item())
        # print("")
        

        ############################################################
        # (4) Update networks
        # ##########################################################

        optim_G.zero_grad()
        loss.backward()
        optim_G.step()

        losses_l1.update(loss_l1.data, img_var.size(0))
        losses_perc.update(loss_perc.data, img_var.size(0))
        losses_style.update(loss_style.data, img_var.size(0))
        # losses_Recon_masked.update(loss_recon_masked.data, img_var.size(0))
        losses_G.update(loss_recon_GAN.data, img_var.size(0))
        losses_D.update(loss_Disc.data, img_var.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()
    
        if i % print_freq == 0:
            logger.info('E : [{0}][{1}/{2}]\t'
                        'T {batch_time.val:.3f}\n'
                        'L1 {l1.val:.4f} ({l1.avg:.4f})\t'
                        'Perc {perc.val:.4f} ({perc.avg:.4f})\t'
                        'Style {style.val:.4f} ({style.avg:.4f})\t'
                        'G {G.val:.4f} ({G.avg:.4f})\t'
                        'D {D.val:.4f} ({D.avg:.4f})\t'.format(
                epoch, i, len(Train_loader), batch_time=batch_time,
                l1 = losses_l1, perc = losses_perc, style = losses_style, G = losses_G, D=losses_D))
        
                
    model_G.eval()
    model_D.eval()

    ###
    # Evaluate PSNR, SSIN, LPIPS with validation set
    ###

    # save_dir = os.path.join(output_dir, 'epoch_{:04d}'.format(epoch+1),'Val')
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir, exist_ok=True)
    for i, (img, mask) in enumerate(Val_loader):
        img_var = img.float().cuda()
        mask_var = mask[0].cuda()
        # name = name[0]

        _, _, h, w = img_var.size()
        img_var = F.interpolate(img_var, size=(h + 16 - h % 16 , w + 16 - w % 16), mode='bilinear')
        mask_var = F.interpolate(mask_var, size=(h + 16 - h % 16 , w + 16 - w % 16), mode='bilinear')
        masked_img = img_var * (1 - mask_var)
        with torch.no_grad():
            try:
                output = model_G(masked_img) * mask_var + img_var * (1-mask_var)
                score_psnr_  = criterion_eval(output, img_var, 'PSNR')
                score_ssim_  = criterion_eval(output, img_var, 'SSIM')
                score_LPIPS_ = criterion_eval(output, img_var, 'LPIPS')
                score_psnr.update(score_psnr_, img_var.size(0))
                score_ssim.update(score_ssim_, img_var.size(0))
                score_lpip.update(score_LPIPS_, img_var.size(0))
            except: pass
            # resultname = os.path.join(save_dir, name[:-4] +'.jpg')
            # save_image(output[0], resultname, quality=100)

    if logger is not None:
        logger.info(' * PSNR  Score is {s.avg:.3f}'.format(s=score_psnr))
        logger.info(' * SSIM  Score is {s.avg:.4f}'.format(s=score_ssim))
        logger.info(' * LPIPS Score is {s.avg:.3f}'.format(s=score_lpip))

    return score_psnr.avg, score_ssim.avg, score_lpip.avg

def train_rest(args, saveDirName='.', logger=None):
    # Print the systems settings
    logger.info(' '.join(sys.argv))
    logger.info(args.memo)
    for k, v in args.__dict__.items():
        logger.info('{0}:\t{1}'.format(k, v))

    # Hyper-parameters
    batch_size = args.batch_size
    crop_size = args.crop_size
    lr = args.lr 
    
    best_score = 0
    
    img_dir = args.img_root
    mask_dir = args.mask_root

    img_val_dir = args.img_val_root
    mask_val_dir = args.mask_val_root
    
    # Define transform functions
    t_pair = [transforms.RandomCrop(crop_size),
            transforms.Resize(crop_size),
            transforms.RandomFlip(),
            transforms.ToTensor()] # transform function for paired training data
    t_unpair = [transforms.RandomCrop_One(crop_size),
                transforms.Resize_One(crop_size),
                transforms.RandomFlip_One(),
                transforms.ToTensor_One()] # transform function for unpaired training data
    v_pair = [transforms.Resize_pair(), transforms.ToTensor()] # transform function for paired validation data
    v_unpair = [transforms.ToTensor_One()] # transform function for unpaired validation training data

    # Define dataloaders
    Train_loader = torch.utils.data.DataLoader(
        RestList('train', img_dir, mask_dir, transforms.Compose(t_pair), transforms.Compose(t_unpair), batch=batch_size),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    Val_loader = torch.utils.data.DataLoader(
        RestList('train', img_val_dir, mask_val_dir, transforms.Compose(v_pair), transforms.Compose(v_unpair), out_name=True),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    loaders = Train_loader, Val_loader
        
    cudnn.benchmark = True

    # Define networks
    model_G = torch.nn.DataParallel(UNet()).cuda()
    model_D = torch.nn.DataParallel(Discriminator()).cuda()
    models = model_G, model_D

    # Define loss functions
    criterion_GAN  = GANLoss().cuda()
    criterion_eval = Evaluation().cuda()
    criterion_Pix  = FullLoss().cuda()
    criterions = criterion_GAN, criterion_eval, criterion_Pix


   
    # Define optimizers
    optim_G = torch.optim.Adam(model_G.parameters(), lr=lr, betas=(0.9, 0.99))
    optim_D = torch.optim.Adam(model_D.parameters(), lr=lr*0.1, betas=(0.9, 0.99))
    optims = optim_G, optim_D

    for epoch in range(args.epochs): # train and validation
        logger.info('Epoch : [{0}]'.format(epoch))
        
        val_psnr, val_ssim, val_lpips = Train(loaders, models, optims, criterions, epoch, best_score, output_dir=saveDirName+'/val', eval_score=psnr, logger=logger)
        
        ## save the neural network
        if best_score < val_psnr :
            best_score = val_psnr
            history_path = saveDirName + '/' + 'ckpt{:03d}_'.format(epoch + 1) + 'p_' + str(val_psnr)[:6] + '_s_' + str(val_ssim)[7:13] + '_l_' + str(val_lpips)[7:13] + '_Better.pkl'
        else : 
            history_path = saveDirName + '/' + 'ckpt{:03d}_'.format(epoch + 1) + 'p_' + str(val_psnr)[:6] + '_s_' + str(val_ssim)[7:13] + '_l_' + str(val_lpips)[7:13] + '.pkl'

        save_checkpoint({
            'epoch': epoch + 1,
            'model_G': model_G.state_dict(),
            'model_D': model_D.state_dict(),
        }, True, filename=history_path)