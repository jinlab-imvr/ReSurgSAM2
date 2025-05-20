# ReSurgSAM2: Referring Segment Anything in Surgical Video via Credible Long-term Tracking 

Official implementation for ReSurgSAM2, an innovative model that leverages the power of the Segment Anything Model 2 (SAM2), integrating it with credible long-term tracking for real-time surgical video segmentation.
> [ReSurgSAM2: Referring Segment Anything in Surgical Video via Credible Long-term Tracking ](https://www.arxiv.org/abs/2505.08581)

>Haofeng Liu, Mingqi Gao, Xuxiao Luo, Ziyue Wang, Guanyi Qin, Junde Wu, Yueming Jin
> 
>Early accepted by MICCAI 2025

The source code is coming soon.

## Overview

Surgical scene segmentation is critical in computer-assisted surgery and is vital for enhancing surgical quality and patient outcomes. We introduce ReSurgSAM2, a two-stage surgical referring segmentation framework that:

- Leverages SAM2 to perform text-referred target detection with our Cross-modal Spatial-Temporal Mamba (CSTMamba) for precise detection and segmentation
- Employs a Credible Initial Frame Selection (CIFS) strategy for reliable tracking initialization
- Incorporates a Diversity-driven Long-term Memory (DLM) that maintains a credible and diverse memory bank for consistent long-term tracking
- Operates in real-time at 61.2 FPS, making it practical for clinical applications
Achieves substantial improvements in accuracy and efficiency compared to existing methods

![architecture](./assets/architecture.png)

## Installization
For SAM2 installization, please refer to [INSTALL.md](INSTALL.md). For this project, we need to run for development in project_root:
```
pip install -e ".[dev]"
```
For Mamba installization, please refer to [mamba/README.md](mamba/README.md). For this project, we need to run in project_root/mamba
```
pip install .
```

## Dataset Acquisition and Preprocessing

Please follow the steps written in [datasets/README.md](datasets/README.md)

## Training
Please set the working directory as project_root, and then follow:

For Ref-Endovis17:

```
export CUDA_VISIBLE_DEVICES=0
python training/train.py --config configs/rvos_training/17/sam2.1_s_ref17_resurgsam_pretrained --num-gpus 1
python training/train.py --config configs/rvos_training/17/sam2.1_s_ref17_resurgsam --num-gpus 1
```

For Ref-Endovis18:

```
export CUDA_VISIBLE_DEVICES=0
python training/train.py --config configs/rvos_training/18/sam2.1_s_ref18_resurgsam_pretrained --num-gpus 1
python training/train.py --config configs/rvos_training/18/sam2.1_s_ref18_resurgsam --num-gpus 1
```

## Evaluation

Download the checkpoints from [google drive](https://drive.google.com/file/d/12pbQhWdKFNPAYk9IC33CVNbeBded7_wI/view?usp=sharing). Place the files at `project_root/checkpoints/`.

Please set the working directory as project_root, and then follow:

For Ref-Endovis17:

```
python tools/rvos_inference.py --training_config_file configs/rvos_training/17/sam2.1_s_ref17_resurgsam --sam2_cfg configs/sam2.1/sam2.1_hiera_s_rvos.yaml --sam2_checkpoint checkpoints/sam2.1_hiera_s_ref17.pth --output_mask_dir results/ref-endovis17/hiera_small_long_mem --dataset_root ./data/Ref-Endovis17/valid --gpu_id 0 --apply_long_term_memory --num_cifs_candidate_frame 5
```

For Ref-Endovis18:

```
python tools/rvos_inference.py --training_config_file configs/rvos_training/18/sam2.1_s_ref18_resurgsam --sam2_cfg configs/sam2.1/sam2.1_hiera_s_rvos.yaml --sam2_checkpoint checkpoints/sam2.1_hiera_s_ref18.pth --output_mask_dir results/ref-endovis18/hiera_small_long_mem --dataset_root ./data/Ref-Endovis18/valid --gpu_id 0 --apply_long_term_memory --num_cifs_candidate_frame 5
```

## Acknowledgement

This research utilizes datasets from [Endovis 2017](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/Downloads/) and [Endovis 2018](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Downloads/). If you wish to use these datasets, please request access through their respective official websites.

Our implementation builds upon the [segment anything 2](https://github.com/facebookresearch/segment-anything-2) framework, [mamba](https://github.com/state-spaces/mamba), and [CLIP](https://github.com/openai/CLIP). We extend our sincere appreciation to the authors for their outstanding work and significant contributions to the field of video segmentation.

## Citation

```
@inproceedings{resurgsam2,
  title={ReSurgSAM2: Referring Segment Anything in Surgical Video via Credible Long-term Tracking},
  author={Haofeng Liu and Mingqi Gao and Xuxiao Luo and Ziyue Wang and Guanyi Qin and Junde Wu and Yueming Jin},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2025},
}
```
