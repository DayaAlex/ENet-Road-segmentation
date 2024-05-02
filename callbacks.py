import wandb
import pytorch_lightning as pl
import torch
import visualize

class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=3):
        super().__init__()
        self.X_img_samples, self.mask_samples = val_samples
        self.X_img_samples= self.X_img_samples[:num_samples]
        self.mask_samples= self.mask_samples[:num_samples] 

    def on_validation_epoch_end(self, trainer, pl_module):#remember model is now pl_module

        self.X_img_samples = self.X_img_samples.to(pl_module.device)
        output_samples = pl_module(self.X_img_samples)

        table = wandb.Table(columns = ["images", "predictions", "targets"] 
            )
        for X_img, output, mask in zip(self.X_img_samples.to("cpu"), output_samples.to("cpu"), self.mask_samples.to("cpu")):
            segmap_pred = visualize.decode_segmap(output.data.max(0)[1].numpy())#pass the max prob channel
            segmap_gt = visualize.decode_segmap(mask.numpy())
        
            table.add_data(wandb.Image(X_img.numpy().transpose(1,2,0)*255), 
                    wandb.Image(segmap_pred), 
                    wandb.Image(segmap_gt)
                    )    

        trainer.logger.experiment.log(
            {'val_images_table': table}
        )

