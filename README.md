# **FundusSexBias**

## Introduction

This repository evaluates sex bias in both conventional deep learning models (ViT-Large, DenseNet121, ResNet50) and ophthalmic foundation models (RETFound, RETFound-DE, VisionFM) for retinal disease diagnosis. It supports fine-tuning under multi-class and multi-label settings, enabling systematic comparison of model performance across male and female subgroups to assess potential demographic bias.

## Prerequisities & Installation

Create environment with conda
```bash
conda create -n FundusSexBias python==3.8
conda activate FundusSexBias
```

Install dependencies for finetuning
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
git clone https://github.com/NBeye-research/FundusSexBias.git
cd FundusSexBias
pip install -r requirements.txt
```
* The fintuning code is based on [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+.

## Finetuning

Organise your data into this directory structure
```bash
├──img_root_path
    label_info.xlsx
    ├──Images
        ├──image1
        ├──image2
        ├──...
```

Multi-class fine-tuning

```bash

PRETRAIN_PATH=/path/to/pretrain_weights.pt
DATA_ROOT=/path/to/img_root_path

#RETFound_mae, vision_FM, vit_large_patch16_224, densenet121, resnet50
ARCH=vit_large_patch16_224

TAG_PATH=${DATA_ROOT}/label_info.xlsx

 CUDA_VISIBLE_DEVICES=0 python run_class_finetuning.py \
     --batch_size 16 \
     --arch ${ARCH} \
     --fine_tuning ${PRETRAIN_PATH} \
     --data_path ${DATA_ROOT} \
     --epochs 50 \
     --lr 5e-3 \
     --tag_path ${TAG_PATH} \
     --image_size 224 \
     --output_dir ./sex_baias_models \
     --tag multi_class
```


Multi-label fine-tuning

```bash

PRETRAIN_PATH=/path/to/pretrain_weights.pt
DATA_ROOT=/path/to/img_root_path

#RETFound_mae, vision_FM, vit_large_patch16_224, densenet121, resnet50
ARCH=vit_large_patch16_224

TAG_PATH=${DATA_ROOT}/label_info.xlsx
#label example
LABEL=AMD,Cataract,DR,Glaucoma,HR,Myopia,Normal

 CUDA_VISIBLE_DEVICES=0 python run_class_finetuning.py \
     --batch_size 16 \
     --arch ${ARCH} \
     --fine_tuning ${PRETRAIN_PATH} \
     --data_path ${DATA_ROOT} \
     --epochs 50 \
     --lr 5e-3 \
     --tag_path ${TAG_PATH} \
     --image_size 224 \
     --output_dir ./sex_baias_models \
     --tag multi_label \
     --labels ${LABEL} \
     --multi_label
```

**Please feel free to contact us for any questions or comments: Zhongwen Li, E-mail: li.zhw@qq.com or Yangyang Wang, E-mail: youngwang666@hotmail.com.**
