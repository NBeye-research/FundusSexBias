
PRETRAIN_ROOT=/data/wyy/Project/pretrain_models
PRETRAIN_PATH=${PRETRAIN_ROOT}/DERETFound_mae_natureCFP.pth
PRETRAIN_PATH=${PRETRAIN_ROOT}/VFM_Fundus_weights.pth
    # --fine_tuning ${PRETRAIN_PATH} \

DATA_ROOT=/data/wyy/data_set/性别差异检测
DATASET_NAME=ODIR_merge
#RETFound_mae, vision_FM, vit_large_patch16_224, densenet121, resnet50
ARCH=vit_large_patch16_224

DATA_PATH=${DATA_ROOT}/${DATASET_NAME}
PARTITIOON_OUTPUT_DIR=./gender_diff_label_files/${DATASET_NAME}
TAG_PATH=${DATA_PATH}/label_info.xlsx
LABEL=AMD,Cataract,DR,Glaucoma,HR,Myopia,Normal

#multi-class
# CUDA_VISIBLE_DEVICES=0 python run_class_finetuning.py \
#     --batch_size 16 \
#     --arch ${ARCH} \
#     --data_path ${DATA_PATH} \
#     --epochs 2 \
#     --lr 5e-3 \
#     --data_path ${DATA_PATH} \
#     --tag_path ${PARTITIOON_OUTPUT_DIR}/partition_multi_class_1.xlsx \
#     --image_size 224 \
#     --output_dir ./sex_differ \
#     --tag ALL_1

#multi-label
CUDA_VISIBLE_DEVICES=0 python run_class_finetuning.py \
    --batch_size 16 \
    --arch ${ARCH} \
    --data_path ${DATA_PATH} \
    --epochs 20 \
    --lr 1e-3 \
    --data_path ${DATA_PATH} \
    --tag_path ${PARTITIOON_OUTPUT_DIR}/partition_all_1.xlsx \
    --image_size 224 \
    --output_dir ./sex_differ \
    --tag multi_label \
    --labels ${LABEL} \
    --multi_label