PRETRAIN_PATH=/path/to/pretrain_weights.pt
DATA_ROOT=/path/to/img_root_path

#RETFound_mae, vision_FM, vit_large_patch16_224, densenet121, resnet50
ARCH=vit_large_patch16_224

TAG_PATH=${DATA_ROOT}/label_info.xlsx

#multi-class
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

#label example
LABEL=AMD,Cataract,DR,Glaucoma,HR,Myopia,Normal

#multi-label
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
