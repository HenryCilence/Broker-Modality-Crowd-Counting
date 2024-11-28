# Official Implement of ECCV 2024 paper "Multi-modal Crowd Counting via a Broker Modality"

The codes consist of three parts:

1 DDFM Inference: To generate the initial intermediate modality. This part of code comes from 
Zhao Z, Bai H, Zhu Y, et al. DDFM: denoising diffusion model for multi-modality image fusion[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 8082-8093.
https://github.com/Zhaozixiang1228/MMIF-DDFM

2 Distillation stage: BMG pre-trained by distilling DDFM as its counterpart.

3 Fine-tuning stage: BMG tuned together with the feature extractor to suit the counting task and bridge two original modalities.
