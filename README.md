# Official Implementation of ECCV 2024 paper "Multi-modal Crowd Counting via a Broker Modality"

The codes consist of three parts:

1 DDFM Inference: To generate the initial intermediate modality. This part of code comes from 
Zhao Z, Bai H, Zhu Y, et al. DDFM: denoising diffusion model for multi-modality image fusion[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 8082-8093.
https://github.com/Zhaozixiang1228/MMIF-DDFM

2 Distillation stage: BMG pre-trained by distilling DDFM as its counterpart.

3 Fine-tuning stage: BMG tuned together with the feature extractor to suit the counting task and bridge two original modalities.

Update on 25 May, 2025:

Share Link: https://pan.baidu.com/s/1pEW_77fOcbx6icZ3ndr8Pg?pwd=eysu

Containing:
- Density maps after fine-tuning on RGBT-CC;
- RGBT-CC dataset with initial broker modality and .npy files;
- DroneRGBT dataset with initial broker modality and .npy files;
- Broker modality images after fine-tuning on RGBT-CC;
- Weight files:
   - Backbone pre-training weight (need loading in args in Fine-tuning/train.py);
   - Broker Modality Generator (BMG) distillation weight from DDFM on RGBT-CC (need loading in args in Fine-tuning/models/bm.py);
   - Broker Modality Generator (BMG) distillation weight from DDFM on DroneRGBT (need loading in args in Fine-tuning/models/bm.py);
   - Model weight after fine-tuning on RGBT-CC;
   - Model weight after fine-tuning on DroneRGBT.

Note: During fine-tuning, the format of .npy files is consistent with Bayesian Loss (https://github.com/ZhihengCV/Bayesian-Crowd-Counting): The first two columns represent the position of each annotation point. The third column records the distances between each annotation point and its nearest neighbors. The share link contains a script to generate .npy files from the original .npy files and the .npy files after processing.
