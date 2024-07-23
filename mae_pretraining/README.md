# Masked Image Pretraining for ViT

## Methods
Three related worsk are included: [MAE](https://github.com/facebookresearch/mae), [MixMAE](https://github.com/Sense-X/MixMIM), [MedMAE](https://github.com/lambert-x/medical_mae).

### Pretrained weights of existing approaches
Use the ViT-base model as backbone, pretrained weights can be downloaded from [ImageNet-1k-pretrain](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and [Mimic-pretrain](https://drive.google.com/file/d/10wqOFCkhyWp6JdSFADrH6Xu9e1am3gXJ/view?usp=share_link)


### Masked Strategy

| Method | Type  | mask_ratio |
|:------:|:-----:|:----------:|
| MAE    | random| 0.75       |
| MedMAE | random| 0.9        |
| MixMAE | mixed | 0.5        |
| MixMAE | dual  | 0.5        |


You need to set the masked strategy and mask_ratio accordingly.


## Pretrain
For MAE：

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
python -m torch.distributed.launch --nproc_per_node=8 \
 --use_env main_pretrain_multi_datasets_xray.py \
 --output_dir ${SAVE_DIR} \
 --log_dir ${SAVE_DIR} \
 --batch_size 256 \
 --model mae_vit_base_patch16_dec512d8b \
 --norm_pix_loss
 --mask_ratio 0.75 \
 --epochs 800 \
 --warmup_epochs 40 \
 --blr 1.5e-4 --weight_decay 0.05 \
 --num_workers 8 \
 --input_size 224 \
 --mask_strategy 'random'\
 --pretrained ${pretrained_path}
```
For MedMAE:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
python -m torch.distributed.launch --nproc_per_node=8 \
 --use_env main_pretrain_multi_datasets_xray.py \
 --output_dir ${SAVE_DIR} \
 --log_dir ${SAVE_DIR} \
 --batch_size 256 \
 --model mae_vit_small_patch16_dec512d2b \
 --mask_ratio 0.90 \
 --epochs 800 \
 --warmup_epochs 40 \
 --blr 1.5e-4 --weight_decay 0.05 \
 --num_workers 8 \
 --input_size 224 \
 --mask_strategy 'random' \
 --random_resize_range 0.5 1.0\
 --pretrained ${pretrained_path}
```

For MixMAE with mixed masking:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
python -m torch.distributed.launch --nproc_per_node=8 \
 --use_env main_pretrain_multi_datasets_xray.py \
 --output_dir ${SAVE_DIR} \
 --log_dir ${SAVE_DIR} \
 --batch_size 128 \
 --model mae_vit_small_patch16_dec512d2b \
 --mask_ratio 0.5 \
 --epochs 800 \
 --warmup_epochs 40 \
 --blr 1.5e-4 --weight_decay 0.05 \
 --num_workers 8 \
 --input_size 224 \
 --mask_strategy 'mixed'\
 --pretrained ${pretrained_path}
```

For MixMAE with dual masking:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
python -m torch.distributed.launch --nproc_per_node=8 \
 --use_env main_pretrain_multi_datasets_xray.py \
 --output_dir ${SAVE_DIR} \
 --log_dir ${SAVE_DIR} \
 --batch_size 128 \
 --model mae_vit_small_patch16_dec512d2b \
 --mask_ratio 0.5 \
 --epochs 800 \
 --warmup_epochs 40 \
 --blr 1.5e-4 --weight_decay 0.05 \
 --num_workers 8 \
 --input_size 224 \
 --mask_strategy 'dual'\
 --pretrained ${pretrained_path}
```

## Feature extraction and save

### Med-SAM

Install [Med-SAM](https://github.com/bowang-lab/MedSAM/) as follows:
```shell
git clone https://github.com/bowang-lab/MedSAM
cd MedSAM
pip install -e.
```

Download SAM weights from [Google Drive](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN)


### Run 
```shell
python extract_features.py --pretrain ${model_weights} --sam_ckpt ${sam_weights}
```


