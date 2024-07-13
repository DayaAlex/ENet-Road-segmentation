# Real-Time Deep Learning Model (ENet) for Semantic Segmentation on the CamVid Dataset and Road Segmentation on IDD-lite using SOTA segmentation models with pretrained MobileNet-v2 backbone.

This repository implements a Real-time deep learning model(ENet) for semantic segmentation on the CamVid Dataset and Road segmentation (binary segmentation: foreground Road) on the IDD-lite dataset using the models found in Segmentations-model-PyTorch Library with pretrained MobileNet-v2 backbone.

Code is developed using PyTorch Lightning, WandB logger, Segmentations-model-PyTorch on the lightning.ai/Kaggle platform. 

There are 3 branches:
1. **Main branch**: Scripts to train only the Encoder.
2. **Decoder attachment**: Scripts for full Network training.
3. **Idd-lite Segmentation**: Road segmentation (binary segmentation: foreground Road) on the IDD-lite dataset, implemented using PyTorch Lightning and the WandB API (for hyperparameter tuning and experiment tracking).


# ENet Development

This is an original PyTorch Lightning implementation for ENet, accounting for the original phased training procedure outlined in the paper(The encoder part of the network is separately trained on the downsampled ground truth, then copying the weights to the composite network for full training on the decoder size ground truth.)

The reference for ENet was the original code written by the author in Lua language and Python implementations available online(links are available in the notebook). 

As the original code was implemented using 4 TitanX GPUs, total replication can be done using Pytorch lightning's distributed training option on a multi-GPU device. For this implementation to carry out semantic segmentation on a single GPU on CamVid, additional settings should be done on Batchnorm layers(use_running_stats=False) which will make the network use current batch statistics instead of precomputed values of the train set.

# Road Segmentation on Indian Driving Dataset lite 

## Problem Statement for Road Segmentation:
Perform pixel-level classification to distinguish drivable roads from images taken from monocular cameras and output binary mask.
<img width="660" alt="prob_statement" src="https://github.com/user-attachments/assets/e55d19d5-e1f0-462b-b9ea-5fcf459f80a3">

## Methodology for Road Segmentation 

<img width="763" alt="methodology 4 50 55 PM" src="https://github.com/user-attachments/assets/e91efe3c-bd08-416b-9adc-00418d8c563a">

## Experiment Tracking using WandB
 ### Hyperparameter tuning
<img width="1358" alt="sweep" src="https://github.com/user-attachments/assets/b001b177-697a-453d-8343-552b1b5730ac">

<img width="911" alt="hparam_result" src="https://github.com/user-attachments/assets/b2a9d114-eafe-407f-8d6c-81eb0136cf8a">

 ### Loss function analysis
Soft Binary Cross Entropy Loss combined with Dice loss was chosen after analysis
![trend_loss_func](https://github.com/user-attachments/assets/29089361-e705-4d8e-a140-68b109bde271)

 ### Trying out different Segmentation Heads
<img width="726" alt="test_per_model" src="https://github.com/user-attachments/assets/4e8dac51-9521-416d-b0a3-edfd85af2428">

 # Output on Jetsontx2
 
ENet output on Jetson TX2 could be achieved without any additional optimization, inference spped was 61.2ms and accuracy for 12 class segmentation using a model trained on current batch statistics of CamVid lite dataset is 66.87 %:
![enet_tx2](https://github.com/user-attachments/assets/5c7944b7-8ffa-4629-9a92-aeb80ddaab7e)

The trained models for Road segmentation were exported to ONNX and optimized using TensorRT.
<img width="732" alt="Screenshot 2024-06-23 at 10 08 48 PM" src="https://github.com/user-attachments/assets/a7a1bbb6-97a7-4eb2-8df1-7ef5a24dca2e">

Inference statistics for the Road Segmentation models deployed on Jetson TX2 is shown below:
![speed_vs_accuracy](https://github.com/user-attachments/assets/0d684a63-494a-4d09-996a-17783fb6cc39)
![inf_speeds_wo_manet](https://github.com/user-attachments/assets/6d72bbde-722b-4761-a6c9-fe3257a24973)

Inference applied to a video file can be viewed [here](https://drive.google.com/drive/folders/1NNDtiulWwHpfb-ZxJjglUfWZ7_vL_5mC?usp=drive_link).

## References
Complete details on my thesis can be found in [here](https://drive.google.com/drive/folders/1hXe-pL0idwwIjPXbGG-_Hze6TPd5pFIe?usp=drive_link)

Paszke, A., Chaurasia, A., Kim, S., & Culurciello, E. (2016). Enet: A deep neural network architecture for real-time semantic segmentation. *arXiv preprint arXiv:1606.02147*.

G. Varma, A. Subramanian, A. Namboodiri, M. Chandraker, and C. Jawahar, “Idd: A dataset for exploring problems of autonomous navigation in unconstrained environments,” in 2019 IEEE winter conference on applications of computer vision (WACV), pp. 1743–1751, IEEE, 2019.

https://github.com/e-lab/ENet-training
https://github.com/iArunava/ENet-Real-Time-Semantic-Segmentation

