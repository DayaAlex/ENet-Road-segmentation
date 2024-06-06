import torch
import pytorch_lightning as pl
from model import ENetEncoder
from dataset import camvid_lite
import config
from callbacks import ImagePredictionLogger
from pytorch_lightning.loggers import WandbLogger
import wandb
import os 

import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from torchvision import transforms
import math
import random
from torchmetrics import JaccardIndex
import albumentations as A
from torch.utils.data import DataLoader, Dataset
import metrics

from torch.optim.lr_scheduler import StepLR
from PIL import Image
import glob

import wandb

if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['WANDB_API_KEY'] = '022c1b388c10d71b380a05f47bab773ce231dcb5'

    logger = WandbLogger()
    run = wandb.init( project = 'Module wise training script testing')
    wandb.run.name = f'epoch {config.NUM_EPOCHS}, lr is {config.LEARNING_RATE}, weight_decay is {config.WEIGHT_DECAY}'

    datamod = camvid_lite(config.BATCH_SIZE)
    datamod.prepare_data()
    datamod.setup()

    class_weights = datamod.class_weights
    val_samples = next(iter(datamod.val_dataloader()))

    model = ENetEncoder(
        C= config.NUM_CLASSES,
        class_weights= class_weights,
        lr= config.LEARNING_RATE,
        weight_decay= config.WEIGHT_DECAY
    )

    trainer = pl.Trainer(
         strategy = 'ddp',
         accelerator = config.ACCELERATOR,
         gpus = 2,
         sync_batchnorm= True,
        logger= logger,
        max_epochs = config.NUM_EPOCHS,
        deterministic = True, 
        log_every_n_steps = 50,#log how many training steps 
        callbacks = [ImagePredictionLogger(val_samples)]
    )
    wandb.watch(model, model.loss, log= 'all', log_freq = 360 )#log every 10 epoch
    trainer.fit(model, datamod)
    trainer.test(datamodule = datamod, ckpt_path = None)