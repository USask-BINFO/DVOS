# A Semi-Self-Supervised Approach for Dense-Pattern Video Object Segmentation
---

## Overview

> This repository contains the official implementation for the paper "A Semi-Self-Supervised Approach for Dense-Pattern Video Object Segmentation". We tackle the challenging task of Dense Video Object Segmentation (DVOS) in agricultural settings, particularly wheat head segmentation, where objects are numerous, small, occluded, and move unpredictably. Our approach uses a semi-self-supervised method leveraging synthetic data and pseudo-labels, significantly reducing the need for costly manual video annotations. The core of our method is a multi-task UNet-style architecture enhanced with diffusion and spatiotemporal attention mechanisms.

<div style="text-align: justify;">
  <img src="data/readme/main_figure02.png" alt="" width="500"/>
  <p><em>Figure 1: The proposed UNet-style architecture, highlighting the multi-task heads (segmentation, reconstruction) and the spatiotemporal attention blocks with diffusion integration.</em></p>
</div>

---

## Setup

### 1. Clone Repository
```bash
git clone https://github.com/KeyhanNajafian/DVOS.git
cd DVOS
```

### 2. Create Environment
We recommend using Conda:
```bash
conda env create -f environment.yaml
conda activate DVOSEnv
```
Alternatively, using pip:
```bash
pip install -r requirements.txt
```

### Prepare the data for video synthesis
### Configuration Files

This repository is primarily driven by YAML configuration files:

- **Frame and Object Extraction:** Config files are located in `extraction/configs/`
- **Video Synthesis:** Config files are located in `simulation/configs/`

**Note**: The sample CSV files can be found in `data/` directory. 

<div style="text-align: justify;">
  <img src="data/readme/sub_figure01.png" alt="" width="500"/>
  <p><em>Figure 2: The procedure for extracting video frames from background videos </em></p>
</div>

<div style="text-align: justify;">
  <img src="data/readme/sub_figure02.png" alt="" width="500"/>
  <p><em>Figure 3: This diagram illustrates the process of synthesizing videos.</em></p>
</div>

### 4. Pretrained Models
Pretrained models for both DVOS and XMem are included within their respective code pipelines.

---

## Usage
### Data Organization
- **DVOSCode Pipeline**:  
  For this pipeline, the frames and masks are stored in CSV files. You can find the CSV metadata inside the `data/` folder, which contains all the necessary references for the frames and masks.

- **DVOSXMem Pipeline**:  
  For this pipeline, you need to organize your data in a `root folder` with two subfolders: `frames/` and `masks`. Inside these subfolders, create identical short video clips subfolders, each containing the corresponding frames and masks for each video.


### Training
**Training DVOS model**
Set the config file properly from DVOSCode pipeline.  
```bash
cd DVOSCode/

python3 ddp_experiment.py --config configs/configs.yaml
```
**Note**: Example CSV files required for simulation and training are included in the data directory for reference.

**Training XMem model**
To train XMem model run the following command inside XMem pipeline:

```bash
cd DVOSXMem/

torchrun --master_port 25763 --nproc_per_node=2 train.py \
  --stage 2 \
  --s2_batch_size 16 \
  --s2_iterations 30000 \
  --s2_finetune 10000 \
  --s2_lr 0.0001 \
  --s2_num_ref_frames 4 \
  --save_network_interval 5000 \
  --load_network model_path.pth \ 
  --wheat_root data_root_dir_path \ 
  --exp_id experiment_name
  ```
**Note**: The XMem pipeline used in this project is a slightly modified version of the original [XMem repository](https://github.com/hkchengrex/XMem). You can still access the original version at the provided link.
>  
### Evaluation
**Evaluating DVOSXMem**
To run evaluation, make sure to:

1. Set the configuration file to the `TEST` phase.
2. Specify the path to the best pretrained model you want to evaluate.

```bash
python3 ddp_experiment.py --config configs/configs.yaml
```

**Evaluating DVOSXMem**
To evaluate a trained DVOSXMem model on a test dataset, run the following command:

```bash
python eval.py \
  --model best_model_path.pth \
  --dataset test_set_root_dir_path \
  --split test \
  --size 384 \
  --output prediction_dir_path
```

This will generate the predicted masks and save them inside the prediction_dir_path folder.

Next, run the following command to calculate the scores and overlay the predictions onto the samples in the test set, using the specified overlay_interval:

```bash
python scoring.py --gt_dir base_test_root_dir --pr_dir prediction_dir_path --overlay_interval 1
```

---
## Citation
```bibtex
@inproceedings{najafian2025semi,
  author    = {Keyhan Najafian, Farhad Maleki, Lingling Jin, Ian Stavness},
  title     = {A Semi-Self-Supervised Approach for Dense-Pattern Video Object Segmentation},
  booktitle = {Vision for Agriculture Workshop, {IEEE/CVF} Conference on Computer Vision and Pattern Recognition (CVPR) 2025},
  year      = {2025},
  note      = {To appear}
}
```