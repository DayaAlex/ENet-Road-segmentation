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
#from segmentation_models_pytorch.losses import DiceLoss

import metrics

from torch.optim.lr_scheduler import StepLR
#import torch.optim.lr_scheduler as lr_scheduler
from PIL import Image
import glob

import wandb


def show_images(images, in_row=True):
    '''
    Helper function to show 3 images
    '''
    total_images = len(images)

    rc_tuple = (1, total_images)
    if not in_row:
        rc_tuple = (total_images, 1)
    
    #figure = plt.figure(figsize=(20, 10))
    for ii in range(len(images)):
        plt.subplot(*rc_tuple, ii+1)
        plt.title(images[ii][0])
        plt.axis('off')
        plt.imshow(images[ii][1])
    plt.show()

def decode_segmap(image, threshold=0.5):
    
    #print(image)#RGB
    Sky = [0, 0, 0]
    Building = [0, 0, 153]
    Pole = [0, 0, 255]
    Road = [51, 153, 255]
    Pavement = [0, 255, 255]
    Tree = [128, 255, 0]
    SignSymbol = [255, 255, 0]
    Fence = [64, 64, 128]
    Car = [255, 128, 0]
    Pedestrian = [255, 0, 127]
    Bicyclist = [255, 204, 255]
    Background_scene = [255,255,255]

    label_colours = np.array([Sky, Building, Pole, Road, 
                              Pavement, Tree, SignSymbol, Fence, Car, 
                              Pedestrian, Bicyclist, Background_scene]).astype(np.uint8)
    
    #print(label_colours.shape)
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    
    for l in range(0, 12):
        r[image == l] = label_colours[l, 0]
        g[image == l] = label_colours[l, 1]
        b[image == l] = label_colours[l, 2]

    rgb = np.zeros((image.shape[0], image.shape[1], 3)).astype(np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb


def visualize_sample(dataset, idx):
    
    img, mask = dataset[idx]  # Fetch the image and mask using the dataset's __getitem__ method
    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img.permute(1, 2, 0))  # Assuming img is a PyTorch tensor of shape [C, H, W]
    ax[0].set_title("Image")
    ax[1].imshow(mask, cmap='gray')  # Assuming mask is a PyTorch tensor of shape [C, H, W] and C=1 for grayscale
    ax[1].set_title("Mask")
    plt.show()
