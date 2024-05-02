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


class InitialBlock(nn.Module):
  
  # Initial block of the model:
  #         Input
  #        /     \
  #       /       \
  # maxpool2d    conv2d-3x3
  #       \       /  
  #        \     /
  #      concatenate
  #          |
  #         Batchnorm
 #        PReLU
   
    def __init__ (self,in_channels = 3,out_channels = 13):
        super().__init__()


        self.maxpool = nn.MaxPool2d(kernel_size=2, 
                                      stride = 2, 
                                      padding = 0)

        self.conv = nn.Conv2d(in_channels, 
                                out_channels,
                                kernel_size = 3,
                                stride = 2, 
                                padding = 1)

        self.prelu = nn.PReLU(16)

        self.batchnorm = nn.BatchNorm2d(16)
  
    def forward(self, x):
        
        main = self.conv(x)
        side = self.maxpool(x)
        #print('main size ', main.size)
        #print('side size ', side.size)
        # concatenating on the channels axis
        x = torch.cat((main, side), dim=1)
        x = self.batchnorm(x)
        x = self.prelu(x)
        #print('init block size ',x.shape)
        
        return x
    
class RDDNeck(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, relu=False, projection_ratio=4, p=0.1):
    
  # Regular|Dilated|Downsampling bottlenecks:
  #
  #     Bottleneck Input
  #        /        \
  #     Identity     \  
  #       /          \
  # maxpooling2d   conv2d-1x1(when downsamp flag is ON, otherwise 2x2)
  # (when downsamp)    | BN +PReLU
  # (-flag is ON)    conv2d-3x3
  #      |             | BN +PReLU
  #      |         conv2d-1x1
  #      |             |
  #  Padding2d     Regularizer(BN + dropout)
  #(when i/p ch !=o/p ch) /   
  #        \            /
  #      Summing + PReLU
  #
  # Params: 
  #  dilation (bool) - if True: creating dilation bottleneck
  #  down_flag (bool) - if True: creating downsampling bottleneck
  #  projection_ratio - ratio between input and output channels
  #  relu - if True: relu used as the activation function else: Prelu us used
  #  p - dropout ratio
        
        super().__init__()
        
        # Define class variables
        self.in_channels = in_channels
        self.reduced_depth = int(out_channels // projection_ratio)
        self.out_channels = out_channels
        self.dilation = dilation
        
        # calculating the number of reduced channels

        self.stride = 1
        self.conv1_kernel = 1
        
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        
        self.dropout = nn.Dropout2d(p=p)

        self.prelu1 = activation
        
        self.conv1 = nn.Conv2d(in_channels = self.in_channels,
                               out_channels = self.reduced_depth,
                               kernel_size = self.conv1_kernel,
                               stride = self.stride,
                               padding = 0,
                               bias = False,
                               dilation = 1)
        
        self.conv2 = nn.Conv2d(in_channels = self.reduced_depth,
                                  out_channels = self.reduced_depth,
                                  kernel_size = 3,
                                  stride = 1,
                                  padding = self.dilation,
                                  bias = True,
                                  dilation = self.dilation)
                                  
        self.prelu2 = activation
        
        self.conv3 = nn.Conv2d(in_channels = self.reduced_depth,
                                  out_channels = self.out_channels,
                                  kernel_size = 1,
                                  stride = 1,
                                  padding = 0,
                                  bias = False,
                                  dilation = 1)
        
        self.prelu3 = activation
        
        self.batchnorm = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm2 = nn.BatchNorm2d(self.out_channels)
#         self.identity = nn.Identity()
        
    def forward(self, x):
        
        #print("Input to layer:", type(x), x.shape if not isinstance(x, tuple) else "Tuple")
    
        if not isinstance(x, tuple):
        #print(x.shape)
            bs = x.size()[0]
        else:
            bs = x[0].size()[0]
            
        #print(bs)
        x_copy = x
        
        # Main branch
        x = self.conv1(x)
        #print(" Conv1 called ")
        x = self.batchnorm(x)
        x = self.prelu1(x)
        #print(self.conv1_kernel, self.stride, x.shape)
        
        x = self.conv2(x)
        #print(" Conv2 called ")
        x = self.batchnorm(x)
        x = self.prelu2(x)
        #print(self.conv2.kernel_size, self.stride,self.dilation, x.shape)
        
        x = self.conv3(x)
        #print(" Conv3 called ")
        x = self.batchnorm2(x)     
        x = self.dropout(x)
        #print(self.conv3.kernel_size, self.stride, x.shape)
        
        #other branch
        if self.in_channels != self.out_channels:
            #print('input and output channels diffrence, so padding of side channel being carried out')
            out_shape = self.out_channels - self.in_channels
            #print('extra channels required ', out_shape)
            
            #padding and concatenating in order to match the channels axis of the side and main branches
            extras = torch.zeros((bs, out_shape, x.shape[2], x.shape[3]), device=x.device)
            #print('x copy shape ',x_copy.shape)
            #print('extras shape ', extras.shape)
            x_copy = torch.cat((x_copy, extras), dim = 1)
            #print('final side route shape ,', x_copy.shape)

        # Summing main and side branches
        x = x + x_copy
        x = self.prelu3(x)
        #print('final layer ', x.shape)
        
        
        return x

class DownRDDNeck(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, relu=False, projection_ratio=4, p=0.1):
    
  # Regular|Dilated|Downsampling bottlenecks:
  #
  #     Bottleneck Input
  #        /        \
  #     Identity     \  
  #       /          \
  # maxpooling2d   conv2d-1x1(when downsamp flag is ON, otherwise 2x2)
  # (when downsamp)    | BN +PReLU
  # (-flag is ON)    conv2d-3x3
  #      |             | BN +PReLU
  #      |         conv2d-1x1
  #      |             |
  #  Padding2d     Regularizer(BN + dropout)
  #(when i/p ch !=o/p ch) /   
  #        \            /
  #      Summing + PReLU
  #
  # Params: 
  #  dilation (bool) - if True: creating dilation bottleneck
  #  down_flag (bool) - if True: creating downsampling bottleneck
  #  projection_ratio - ratio between input and output channels
  #  relu - if True: relu used as the activation function else: Prelu us used
  #  p - dropout ratio
        
        super().__init__()
        
        # Define class variables
        self.in_channels = in_channels
        self.reduced_depth = int(out_channels // projection_ratio)
        self.out_channels = out_channels
        self.dilation = dilation
        
        # calculating the number of reduced channels
     
        self.stride = 2
        self.conv1_kernel = 2
       
        
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        
        self.maxpool = nn.MaxPool2d(kernel_size = 2,
                                      stride = 2,
                                      padding = 0, return_indices=True)
        
        self.dropout = nn.Dropout2d(p=p)

        self.prelu1 = activation
        
        self.conv1 = nn.Conv2d(in_channels = self.in_channels,
                               out_channels = self.reduced_depth,
                               kernel_size = self.conv1_kernel,
                               stride = self.stride,
                               padding = 0,
                               bias = False,
                               dilation = 1)
        
        self.conv2 = nn.Conv2d(in_channels = self.reduced_depth,
                                  out_channels = self.reduced_depth,
                                  kernel_size = 3,
                                  stride = 1,
                                  padding = self.dilation,
                                  bias = True,
                                  dilation = self.dilation)
                                  
        self.prelu2 = activation
        
        self.conv3 = nn.Conv2d(in_channels = self.reduced_depth,
                                  out_channels = self.out_channels,
                                  kernel_size = 1,
                                  stride = 1,
                                  padding = 0,
                                  bias = False,
                                  dilation = 1)
        
        self.prelu3 = activation
        
        self.batchnorm = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm2 = nn.BatchNorm2d(self.out_channels)
#         self.identity = nn.Identity()
        
    def forward(self, x):
        
        #print("Input to layer:", type(x), x.shape if not isinstance(x, tuple) else "Tuple")
    
        if not isinstance(x, tuple):
        #print(x.shape)
            bs = x.size()[0]
        else:
            bs = x[0].size()[0]
            
        #print(bs)
        x_copy = x
        
        # Main branch
        x = self.conv1(x)
        #print(" Conv1 called ")
        x = self.batchnorm(x)
        x = self.prelu1(x)
        #print(self.conv1_kernel, self.stride, x.shape)
        
        x = self.conv2(x)
        #print(" Conv2 called ")
        x = self.batchnorm(x)
        x = self.prelu2(x)
        #print(self.conv2.kernel_size, self.stride,self.dilation, x.shape)
        
        x = self.conv3(x)
        #print(" Conv3 called ")
        x = self.batchnorm2(x)     
        x = self.dropout(x)
        #print(self.conv3.kernel_size, self.stride, x.shape)
        
        #other branch
        
        x_copy, indices = self.maxpool(x_copy)
            
        if self.in_channels != self.out_channels:
            #print('input and output channels diffrence, so padding of side channel being carried out')
            out_shape = self.out_channels - self.in_channels
            #print('extra channels required ', out_shape)
            
            #padding and concatenating in order to match the channels axis of the side and main branches
            extras = torch.zeros((bs, out_shape, x.shape[2], x.shape[3]), device=x.device)
            #print('x copy shape ',x_copy.shape)
            #print('extras shape ', extras.shape)
            x_copy = torch.cat((x_copy, extras), dim = 1)
            #print('final side route shape ,', x_copy.shape)

        # Summing main and side branches
        x = x + x_copy
        x = self.prelu3(x)
        #print('final layer ', x.shape)
        
        
        return x, indices

class ASNeck(nn.Module):
    def __init__(self, in_channels, out_channels, projection_ratio=4):
      
  # Asymetric bottleneck:
  #
  #     Bottleneck Input
  #        /        \
  #       /          \
  #      |         conv2d-1x1
  # Identity           | PReLU
  #      |         conv2d-1x5
  #      |             |
  #      |         conv2d-5x1
  #      |             | PReLU
  #      |         conv2d-1x1
  #      |             |
  #       \     Regularizer
  #       \           /  
  #        \         /
  #      Summing + PReLU
  #
  # Params:    
  #  projection_ratio - ratio between input and output channels
        
        super().__init__()
        
        # Define class variables
        self.in_channels = in_channels
        self.reduced_depth = int(out_channels // projection_ratio)
        self.out_channels = out_channels
        
        self.dropout = nn.Dropout2d(p=0.1)
        
        self.conv1 = nn.Conv2d(in_channels = self.in_channels,
                               out_channels = self.reduced_depth,
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               bias = False)
        
        self.prelu1 = nn.PReLU()
        
        self.conv21 = nn.Conv2d(in_channels = self.reduced_depth,
                                  out_channels = self.reduced_depth,
                                  kernel_size = (5, 1),
                                  stride = 1,
                                  padding = (2, 0),
                                  bias = False)
        
        self.conv22 = nn.Conv2d(in_channels = self.reduced_depth,
                                  out_channels = self.reduced_depth,
                                  kernel_size = (1, 5),
                                  stride = 1,
                                  padding = (0, 2),
                                  bias = True)############TRUE bias in original code###############
        
        self.prelu2 = nn.PReLU()
        
        self.conv3 = nn.Conv2d(in_channels = self.reduced_depth,
                                  out_channels = self.out_channels,
                                  kernel_size = 1,
                                  stride = 1,
                                  padding = 0,
                                  bias = False)
        
        self.prelu3 = nn.PReLU()
        
        self.batchnorm = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm2 = nn.BatchNorm2d(self.out_channels)
#         self.identity = nn.Identity()
        
    def forward(self, x):
   
        #main branch
        x_copy = x
        #print('side branch')
        
        # Side Branch
        x = self.conv1(x)
        #print('conv1 called')
        x = self.batchnorm(x)
        x = self.prelu1(x)
        #print(self.conv1.kernel_size,self.conv1.stride, x.shape)
        
        x = self.conv21(x)
        #print('conv21 called')
        #print(self.conv21.kernel_size, self.conv21.stride, x.shape)
        x = self.conv22(x)
        #print('conv22 called')
        #print(self.conv22.kernel_size, self.conv22.stride, x.shape)
        x = self.batchnorm(x)
        x = self.prelu2(x)
        
        x = self.conv3(x)
        #print('conv3 called')   
        x = self.dropout(x)
        x = self.batchnorm2(x)
        #print('final main ',self.conv3.kernel_size,self.conv3.stride, x.shape)

        # Summing main and side branches
        x = x + x_copy
        x = self.prelu3(x)
        #print('final total ', x.shape)
        
        return x
 

class ENetEncoder(pl.LightningModule):
  #to do, write a description 
  
    def __init__(self, C, class_weights, lr=5e-4, weight_decay=2e-4 ):
        super().__init__()
        
        # Define class variables
        self.C = C
        self.class_weights = class_weights
        self.loss = loss_function(self.class_weights)
        self.tp, self.fp, self.fn, self.tn = 0,0,0,0
        self.maxiou = 10e-4#keeping as very small number

        self.save_hyperparameters()#saves the hyperparameters defined in __init__ and can be accessed using self.hparams['key']

        # The initial block
        self.init = InitialBlock()
        
        # The first bottleneck
        self.b10 = DownRDDNeck(dilation=1, 
                           in_channels=16, 
                           out_channels=64, 
                           p=0.01)
        
        self.b11 = RDDNeck(dilation=1, 
                           in_channels=64, 
                           out_channels=64,   
                           p=0.01)
        
        self.b12 = RDDNeck(dilation=1, 
                           in_channels=64, 
                           out_channels=64, 
                           p=0.01)
        
        self.b13 = RDDNeck(dilation=1, 
                           in_channels=64, 
                           out_channels=64, 
                           p=0.01)
        
        self.b14 = RDDNeck(dilation=1,
                           in_channels=64, 
                           out_channels=64, 
                           p=0.01)
        
        
        # The second bottleneck
        self.b20 = DownRDDNeck(dilation=1, 
                           in_channels=64, 
                           out_channels=128, 
                          )
        
        self.b21 = RDDNeck(dilation=1, 
                           in_channels=128, 
                           out_channels=128, 
                           )
        
        self.b22 = RDDNeck(dilation=2, 
                           in_channels=128, 
                           out_channels=128, 
                           )
        
        self.b23 = ASNeck(in_channels=128, 
                          out_channels=128)
        
        self.b24 = RDDNeck(dilation=4, 
                           in_channels=128, 
                           out_channels=128, 
                           )
        
        self.b25 = RDDNeck(dilation=1, 
                           in_channels=128, 
                           out_channels=128, 
                           )
        
        self.b26 = RDDNeck(dilation=8, 
                           in_channels=128, 
                           out_channels=128, 
                           )
        
        self.b27 = ASNeck(in_channels=128, 
                          out_channels=128)
        self.b28 = RDDNeck(dilation=16, 
                           in_channels=128, 
                           out_channels=128, 
                           )
        
        
        # The third bottleneck
        self.b31 = RDDNeck(dilation=1, 
                           in_channels=128, 
                           out_channels=128, 
                           )
        
        self.b32 = RDDNeck(dilation=2, 
                           in_channels=128, 
                           out_channels=128, 
                           )
        
        self.b33 = ASNeck(in_channels=128, 
                          out_channels=128)
        
        self.b34 = RDDNeck(dilation=4, 
                           in_channels=128, 
                           out_channels=128, 
                           )
        
        self.b35 = RDDNeck(dilation=1, 
                           in_channels=128, 
                           out_channels=128, 
                           )
        
        self.b36 = RDDNeck(dilation=8, 
                           in_channels=128, 
                           out_channels=128, 
                           )
        
        self.b37 = ASNeck(in_channels=128, 
                          out_channels=128)
        
        self.b38 = RDDNeck(dilation=16, 
                           in_channels=128, 
                           out_channels=128, 
                           )
        
        #fully convolutional layer to get the encoder output
        self.enc_conv = nn.Conv2d(in_channels=128,
                                 out_channels = 12,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias =False)
    def forward(self, x):
        
        # The initial block
        x = self.init(x)
        #print('^^^^^^^^^^^^^^init block^^^^^^^^^^^^^^^^')
        
        # The first bottleneck
        x, i1 = self.b10(x)
        #print('................bottleneck_10 complete................')
        x = self.b11(x)
        #print('................bottleneck_11 complete...........')
        x = self.b12(x)
        #print('...............bottleneck_12 complete............')
        x = self.b13(x)
        #print('............bottleneck_13 complete............')
        x = self.b14(x)
        #print('.................bottleneck_14 complete.............')
        
        # The second bottleneck
        x, i2 = self.b20(x)
        #print('----------------bottleneck_20 complete-----------')
        x = self.b21(x)
        #print('----------------bottleneck_21 complete-----------')
        x = self.b22(x)
        #print('-----------bottleneck_22 complete--------------')
        x = self.b23(x)
        #print('-----------bottleneck_23 complete---------------')
        x = self.b24(x)
        #print('--------------bottleneck_24 complete------------')
        x = self.b25(x)
        #print('-------------bottleneck_25 complete------------')
        x = self.b26(x)
        #print('-------------bottleneck_26 complete-------------')
        x = self.b27(x)
        #print('-------------bottleneck_27 complete-------------')
        x = self.b28(x)
        #print('------------bottleneck_28 complete----------------')
        
        # The third bottleneck
        x = self.b31(x)
        #print('********bottleneck_31 complete************')
        x = self.b32(x)
        #print('********bottleneck_32 complete************')
        x = self.b33(x)
        #print('********bottleneck_33 complete************')
        x = self.b34(x)
        #print('********bottleneck_34 complete************')
        x = self.b35(x)
        #print('********bottleneck_35 complete************')
        x = self.b36(x)
        #print('********bottleneck_36 complete************')
        x = self.b37(x)
        #print('********bottleneck_37 complete************')
        x = self.b38(x)
        #print('********bottleneck_38 complete************')
        
        x = self.enc_conv(x)
        #print(' encoder indices first :', i1[0],'#'*8, i2[0])
        return x
    
    def training_step(self, batch, batch_idx):
        
        X_batch, mask_batch = batch
        out = self(X_batch.float()) 
        train_loss = self.loss(out, mask_batch.long())

        self.log('train/loss', train_loss, on_step = True, on_epoch = True)
        return train_loss

    def on_validation_epoch_start(self):#hook
        self.val_step_outputs = [] #we will fill this with the logits(prediction)

    def validation_step(self, batch, batch_idx):
        #print(f"Batch index: {batch_idx}, Batch size: {len(batch)}")
        X_batch, mask_batch = batch
        out = self(X_batch.float())
        self.val_step_outputs.append(torch.softmax(out ,dim =1))

        val_loss = self.loss(out, mask_batch.long())

        _, predicted_classes = torch.max(out, dim = 1)# to get the id of the channel having largest prob (argmax of prob) 
        this_tp, this_fp, this_fn, this_tn = metrics.get_stats(
                                                predicted_classes, mask_batch.long(), mode = "multiclass", num_classes= 12
        )

        self.tp += this_tp 
        self.fp += this_fp 
        self.fn += this_fn 
        self.tn += this_tn 
                
        self.log('val/loss', val_loss,  on_step = False, on_epoch = True) 
        return out
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.hparams.lr, weight_decay = self.hparams.weight_decay) 
        return {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler":StepLR(optimizer, step_size = 100, gamma = 0.1 ),
                    "monitor":'val_loss'}
                }

    def on_validation_epoch_end(self):#hook
        miou_score = metrics.iou_score(sum(self.tp), sum(self.fp), sum(self.fn), sum(self.tn), reduction = 'micro')
        self.log('val/val_accuracy', miou_score )
        if miou_score > self.maxiou:
            self.maxiou = miou_score
            checkpoint = {
                'epochs' : self.current_epoch,
                'state_dict': self.state_dict(),
                'miou': self.maxiou
            } 
            torch.save(checkpoint, f'./CNNEncoder_for_ENet_trained_on_Camvid_epoch{self.current_epoch}_acc{self.maxiou:.3f}.pth') #checkpoint,checkpoint path   
            self.log('New best model saved with miou:', self.maxiou)   
        self.tp, self.fp, self.fn, self.tn  = 0,0,0,0#reseting for next epoch calculation

        flattened_prob = torch.flatten(torch.cat(self.val_step_outputs))
        try:
            self.logger.experiment.log({
            'valid/softmax': wandb.Histogram(flattened_prob),
            'epoch': self.current_epoch
            })
        except Exception as e:
            print(f"Error logging to WandB: {e}")

    def test_step(self, batch, batch_idx):
        X_batch, mask_batch = batch
        out = self(X_batch.float())
        test_loss = self.loss(out, mask_batch.long())
        
        _, predicted_classes = torch.max(out, dim = 1)
        this_tp, this_fp, this_fn, this_tn = metrics.get_stats(
                                                predicted_classes, mask_batch.long(), mode = "multiclass", num_classes= 12
        )

        self.tp += this_tp 
        self.fp += this_fp 
        self.fn += this_fn 
        self.tn += this_tn 

        self.log('test/loss', test_loss, on_step = False, on_epoch = True)
        return test_loss

    def on_test_epoch_end(self):#hook
        miou_score = metrics.iou_score(sum(self.tp), sum(self.fp), sum(self.fn), sum(self.tn), reduction = 'micro')
        self.log('test/test_accuracy', miou_score )
        self.tp, self.fp, self.fn, self.tn  = 0, 0, 0, 0

        dummy_input = torch.zeros((1,3,360,480), device=self.device)
        model_filename = f"model_{self.current_epoch}.onnx"
        torch.onnx.export(self, dummy_input, model_filename, opset_version=11)
        artifact = wandb.Artifact(name="test_enet_model", type="model")
        artifact.add_file(model_filename)
        self.logger.experiment.log_artifact(artifact)
        
class loss_function(nn.Module):
    def __init__(self, class_weights):
        super().__init__()

        self.register_buffer('cls_wts', class_weights)
        self.criterion = nn.CrossEntropyLoss(weight = self.cls_wts)

    def forward(self, out, target):
        loss = self.criterion(out, target)
        return loss    