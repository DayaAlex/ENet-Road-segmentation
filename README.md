This is an implementation of a Real time deep learning model(ENet) for semantic segmentation on the CamVid Dataset. Code is developed using PyTorch Lightning, Wandb logger on lightning.ai/Kaggle platform. The reference used was the original code written by the author in lua language and python implemntations available online. This is an original pytorch lightning implemenatation. It is also strict code implementation with the encoder part of the network being seperately, then copying the weights to the composite network for full training. As the original code was implemented using 4 TitanX GPU's, total replication can be carried out using Pytorch lightning's distributed training option on a cloud GPU. For this implementation to carryout semantic segmentation on a single GPU, additional setting should be done on batchnorm layers(use_running_stats=False).



Output on Jetsontx2
https://drive.google.com/drive/folders/1NNDtiulWwHpfb-ZxJjglUfWZ7_vL_5mC?usp=drive_link
