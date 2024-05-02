import wandb
import pytorch_lightning as pl
import torch

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
            segmap_pred = decode_segmap(output.data.max(0)[1].numpy())#pass the max prob channel
            segmap_gt = decode_segmap(mask.numpy())
        
            table.add_data(wandb.Image(X_img.numpy().transpose(1,2,0)*255), 
                    wandb.Image(segmap_pred), 
                    wandb.Image(segmap_gt)
                    )    

        trainer.logger.experiment.log(
            {'val_images_table': table}
        )

        validation_step_outputs = getattr(pl_module, 'val_step_outputs', None)#returns None when  'val_step_outputs' cannot be found
        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        trainer.logger.experiment.log(
            {
                'valid/logits':flattened_logits,
                'global_step':trainer.global_step
            }
        )
