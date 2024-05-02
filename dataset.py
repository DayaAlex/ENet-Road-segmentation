import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
from torchvision import transforms
import math
import random
from torchmetrics import JaccardIndex
import albumentations as A
from torch.utils.data import DataLoader, Dataset

import metrics

from torch.optim.lr_scheduler import StepLR
#import torch.optim.lr_scheduler as lr_scheduler
from PIL import Image
import glob

import wandb

def augmentation(mode='train', h= 360, w =480):
    """ resizes image to input size and mask to a downsampled size, 
        applies horizontal flip and color jitter augmentation only to trainsets
        
    """
    if mode == 'train':
        img_transformation = A.Compose([
                        A.Resize(h,w),
                        A.HorizontalFlip(p= 0.5),
                        #A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

                    ])
        #mask size different from image size
        mask_transformation = A.Compose([
                    A.Resize(h//8, w//8),
                    A.HorizontalFlip(p=0.5)
                    ])
        
    else:
        img_transformation =A.Resize(h, w)
        mask_transformation = A.Resize(h//8, w//8)
        
    return img_transformation,  mask_transformation

def get_wts(dataset):
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    hist = torch.zeros(12)
    for batch in tqdm(loader):
        _, mask = batch
        #print(mask.shape)
        mask = mask.squeeze(0).float()
        #print(mask.shape)

        hist += torch.histc(mask, 12, 0,11 )

    norm_hist = hist/torch.sum(hist)

    class_wts = torch.ones(12)
    for idx in range(12):
        if hist[idx]<1 or idx ==11:
            class_wts[idx] = 0
        else:
            class_wts[idx] = 1/torch.log(1.02 + norm_hist[idx])

    return class_wts

class CamvidDataset(Dataset):
    """custom camvid datset that returns images and the corresponding masks after augmentation and normalisation  
        
    """
    def __init__(self, img_path, mask_path, augmentation, norm_transform=True, road_idx=None):
        self.filenames_t = os.listdir(img_path)
        self.img_path = img_path
        self.mask_path = mask_path
        self.norm_transform = norm_transform
        self.augmentation = augmentation
        #for exttracting road mask
        #self.road_idx = road_idx 

    def __len__(self):
        return len(self.filenames_t)

    def __getitem__(self, idx):
        each_img_path = os.path.join(self.img_path, self.filenames_t[idx])
        each_mask_path = os.path.join(self.mask_path, self.filenames_t[idx])
        
#         img = cv2.imread(each_img_path, cv2.COLOR_BGR2RGB)
        img = cv2.imread(each_img_path, cv2.IMREAD_COLOR)  # Load the image in BGR color space
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)         # Convert from BGR to RGB

        label_array = cv2.imread(each_mask_path, cv2.IMREAD_GRAYSCALE)#h,w array datatype
        label = np.expand_dims(label_array, axis = -1)#h,w,c
        
        if self.augmentation:
            img_transforms, mask_transforms = self.augmentation
            seed = 7
            random.seed(seed)
            img = img_transforms(image=img)['image']# albumentations must be passed with named argument, and gets stored with that name as key
            random.seed(seed)
            mask = mask_transforms(image=label)['image']
            
        if self.norm_transform:
            normalize_tensor = transforms.Compose([
                            transforms.ToTensor(),
                                     ])
            img = normalize_tensor(img)
            
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.uint8)
        mask = mask.permute(2, 0, 1)#c,h,w
        #print(mask.shape)
        mask = mask.squeeze()#h,w
        #print(mask.shape)
        return img, mask

class camvid_lite(pl.LightningDataModule):
    def __init__(self, batch_size=10):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        self.class_weights = get_wts(CamvidDataset('/teamspace/uploads/CamVid/train','/teamspace/uploads/CamVid/trainannot',augmentation('train')))

    def setup(self, stage =None):#stage none means 
        if stage =='fit' or stage is None:
            self.train_aug = augmentation('train')
            self.val_aug = augmentation('val')
            self.train_dataset = CamvidDataset('/teamspace/uploads/CamVid/train','/teamspace/uploads/CamVid/trainannot', self.train_aug)
            self.val_dataset = CamvidDataset('/teamspace/uploads/CamVid/val','/teamspace/uploads/CamVid/valannot', self.val_aug)

        if stage =='test' or stage is None:
            self.test_aug = augmentation('val')
            self.test_dataset = CamvidDataset('/teamspace/uploads/CamVid/test','/teamspace/uploads/CamVid/testannot', self.test_aug)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False, drop_last=True)
