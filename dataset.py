import os
import numpy as np
import random

from PIL import Image
from glob import glob
#import rawpy
import data_transforms as transforms
from torchvision import transforms as torch_transforms
import torch


class RestList(torch.utils.data.Dataset):
    def __init__(self, phase, img_dir, mask_dir, t_pair, t_unpair, batch=1, out_name=False):
        self.phase = phase
        self.batch = batch
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        
        self.t_pair = t_pair
        self.t_unpair = t_unpair
        self.out_name = out_name

        self.image_list = None
        self.mask_list = None
        self.test_image_list = None
        self.test_mask_list = None

        self._make_list()


    def __getitem__(self, index):
        np.random.seed()
        random.seed()
        if self.phase == 'train':
           
            index_ = np.random.randint(len(self.image_list))
            img = Image.open(self.image_list[index_]).convert('RGB')
            
            
            index_m = np.random.randint(len(self.mask_list))
            Mask = Image.open(self.mask_list[index_m]).convert('RGB')

            data = list(self.t_unpair(*[img]))
            data.append(self.t_unpair(*[Mask]))

        elif self.phase == 'test' :
            img  = Image.open(self.test_image_list[index]).convert('RGB')
            Mask = Image.open(self.test_mask_list[index]).convert('RGB')

            name = (self.test_image_list[index]).split('/')
            name = name[-2] + '/' + name[-1]

            data = list(self.t_unpair(*[img]))
            data.append(self.t_unpair(*[Mask]))
            data.append(name)

        return tuple(data)


    def __len__(self):
        if self.phase == 'train':
            return 100000 #5000 * self.batch
        elif self.phase == 'test' :
            return len(self.test_image_list)

    def _make_list(self):
        if self.phase == 'train':
            
            img_source = os.path.join(self.img_dir, '*.png')
            mask_source = os.path.join(self.mask_dir, '*.png')
            
            image_list = sorted(glob(img_source))
            print("Img Loaded")
            
            rand_idx = random.sample(range(0, len(image_list)), self.DATA_NUM)
            self.image_list = [image_list[x] for x in rand_idx]
            self.mask_list = sorted(glob(mask_source))
            print()
            print('Num of training images : ' + str(len(self.image_list)))
            print('Num of training masks : ' + str(len(self.mask_list)))
            
        #! TEST
        elif self.phase == 'test' :
            img_source = os.path.join(self.img_dir, '**/*.png')
            mask_source = os.path.join(self.mask_dir, '**/*.png')
            
            self.test_image_list = sorted(glob(img_source))
            print('Num of Test imgs : ' + str(len(self.test_image_list)))
            
            self.test_mask_list = sorted(glob(mask_source))
            print('Num of Test masks : ' + str(len(self.test_mask_list)))
        
       