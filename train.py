import torch
import pytorch_lightning as pl
from model import ENetEncoder
from dataset import camvid_lite
import config
from callbacks import ImagePredictionLogger
from pytorch_lightning.loggers import WandbLogger
import wandb
import os 

if __name__ == "__main__":
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
        # strategy = 'ddp',
        # accelerator = config.ACCELERATOR,
        # gpus = 1,
        logger= logger,
        max_epochs = config.NUM_EPOCHS,
        deterministic = True, 
        log_every_n_steps = 50,#log how many training steps 
        callbacks = [ImagePredictionLogger(val_samples)]
    )
    wandb.watch(model, model.loss, log= 'all', log_freq = 360 )#log every 10 epoch
    trainer.fit(model, datamod)
    trainer.test(datamodule = datamod, ckpt_path = None)