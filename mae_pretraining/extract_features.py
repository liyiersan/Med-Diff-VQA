import os
import tqdm
import torch
import argparse
import numpy as np
from pathlib import Path
import utils.misc as misc
import models.vit as models_vit
import torch.utils.data as torch_data
from models.tiny_vit_sam import TinyViT
from dataloader.datasets import MimicImageDataset
from segment_anything import sam_model_registry

def get_args_parser():
    parser = argparse.ArgumentParser('Feature Extraction', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for evaluation')
    parser.add_argument('--model', default='vit_base_patch16', type=str, help='Name of model to extract features')
    parser.add_argument('--pretrain', default='./mae_pretraining/pretrained/vit-b_CXR_0.5M_mae.pth', type=str, help='model checkpoint path to load')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool', help='Use class token instead of global pool for classification')
    parser.add_argument('--sam_size', default=1024, type=int, choices=[1024, 256], help='image size for SAM model -- 1024 for vit_b, 256 for vit_tiny')
    parser.add_argument('--sam_ckpt', default='./mae_pretraining/pretrained/sam_vit_b_01ec64.pth', type=str, help='SAM model checkpoint path to load')

    # Dataset parameters
    parser.add_argument('--data_path', default='/data/mimic-cxr-jpg-2.1.0/', type=str, help='dataset path')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--output_dir', default='./pretrained_img_features', type=str, help='path where to save, empty for no saving')

    # dataloader parameters
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', action='store_false', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    
    parser.add_argument('--multi_gpu', action='store_true', help='Use multiple gpus for feature extraction')
    parser.set_defaults(multi_gpu=True)

    return parser

def custom_collate_fn(batch):
    """
    Collate function to handle mixed data types including image tensors and string IDs.
    """
    # Determine if img_ids are included based on the first element's length
    if isinstance(batch[0][-1], str):
        if len(batch[0]) == 6:
            # All elements present including img_ids and SAM features
            ref_img_mae, study_img_mae, ref_img_sam, study_img_sam, ref_img_id, study_img_id = zip(*batch)
            return (
                torch.stack(ref_img_mae),
                torch.stack(study_img_mae),
                torch.stack(ref_img_sam),
                torch.stack(study_img_sam),
                ref_img_id,
                study_img_id
            )
        elif len(batch[0]) == 4:
            # Only MAE features and img_ids present
            ref_img_mae, study_img_mae, ref_img_id, study_img_id = zip(*batch)
            return (
                torch.stack(ref_img_mae),
                torch.stack(study_img_mae),
                ref_img_id,
                study_img_id
            )
    else:
        if len(batch[0]) == 4:
            # Only MAE and SAM features without img_ids
            ref_img_mae, study_img_mae, ref_img_sam, study_img_sam = zip(*batch)
            return (
                torch.stack(ref_img_mae),
                torch.stack(study_img_mae),
                torch.stack(ref_img_sam),
                torch.stack(study_img_sam)
            )
        else:
            # Only MAE features without img_ids
            ref_img_mae, study_img_mae = zip(*batch)
            return torch.stack(ref_img_mae), torch.stack(study_img_mae)


def forward_features(mae_imgs, sam_imgs, mae_model, sam_model):
    with torch.no_grad():
        mae_features = mae_model(mae_imgs)  # (B, 196, 768)
        sam_features = sam_model(sam_imgs)  # (B, 256, 64, 64)
    return mae_features.cpu().numpy(), sam_features.cpu().numpy()

def extract_features(data_loader, mae_model, sam_model, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for (ref_imgs_mae, study_imgs_mae, ref_imgs_sam, study_imgs_same, ref_img_ids, study_img_ids) in tqdm.tqdm(data_loader):
        ref_imgs_mae, study_imgs_mae = ref_imgs_mae.to(device), study_imgs_mae.to(device)
        ref_imgs_sam, study_imgs_same = ref_imgs_sam.to(device), study_imgs_same.to(device)
        
        # Extract features
        ref_mae_feats, ref_sam_feats = forward_features(ref_imgs_mae, ref_imgs_sam, mae_model, sam_model)
        study_mae_feats, study_sam_feats = forward_features(study_imgs_mae, study_imgs_same, mae_model, sam_model)
        
        # Save features to individual compressed npz files based on img_id
        for i, ref_img_id in enumerate(ref_img_ids):
            ref_file = os.path.join(output_dir, f'features_{ref_img_id}.npz')
            np.savez_compressed(ref_file, mae=ref_mae_feats[i], sam=ref_sam_feats[i])

        for i, study_img_id in enumerate(study_img_ids):
            study_file = os.path.join(output_dir, f'features_{study_img_id}.npz')
            np.savez_compressed(study_file, mae=study_mae_feats[i], sam=study_sam_feats[i])

        

def build_sam_encoder(args):
    if args.sam_size == 1024:
        sam_model = sam_model_registry["vit_b"](checkpoint=args.sam_ckpt)
        image_encoder = sam_model.image_encoder
    elif args.sam_size == 256:
        image_encoder = TinyViT(
            img_size=256,
            in_chans=3,
            embed_dims=[
                64, ## (64, 256, 256)
                128, ## (128, 128, 128)
                160, ## (160, 64, 64)
                320 ## (320, 64, 64) 
            ],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8
        )
        args.sam_ckpt = './mae_pretraining/pretrained/lite_medsam.pth'
        misc.load_pretrain(image_encoder, args.sam_ckpt)
    else:
        raise ValueError(f"Invalid sam_size: {args.sam_size}, should be 1024 or 256")
    return image_encoder

def main(args):
    device = torch.device(args.device)

    # Load datasets
    dataset_train = MimicImageDataset(data_dir=args.data_path, flag='train', args=args, is_pretrain=False, sam_size=args.sam_size, return_img_id=True)
    dataset_val = MimicImageDataset(data_dir=args.data_path, flag='val', args=args, is_pretrain=False, sam_size=args.sam_size, return_img_id=True)
    dataset_test = MimicImageDataset(data_dir=args.data_path, flag='test', args=args, is_pretrain=False, sam_size=args.sam_size, return_img_id=True)

    # Scale batch size and workers if using multiple GPUs
    scale = 1
    if args.multi_gpu:
        scale = torch.cuda.device_count()
    batch_size = min(args.batch_size * scale, 128)
    num_workers = min(args.num_workers * scale, 32)
    

    data_loader_train = torch_data.DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, pin_memory=args.pin_mem, drop_last=False, collate_fn=custom_collate_fn)
    data_loader_val = torch_data.DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, pin_memory=args.pin_mem, drop_last=False, collate_fn=custom_collate_fn)
    data_loader_test = torch_data.DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers, pin_memory=args.pin_mem, drop_last=False, collate_fn=custom_collate_fn)

    # Initialize models
    mae_model = models_vit.__dict__[args.model](num_classes=1000, global_pool=args.global_pool)
    misc.load_pretrain(mae_model, args.pretrain)
    mae_model.to(device)
    mae_model.eval()

    medsam_model = build_sam_encoder(args)
    medsam_model.to(device)
    medsam_model.eval()

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1 and args.multi_gpu:
        mae_model = torch.nn.DataParallel(mae_model)
        medsam_model = torch.nn.DataParallel(medsam_model)

    # Extract features
    extract_features(data_loader_train, mae_model, medsam_model, device, args.output_dir)
    extract_features(data_loader_val, mae_model, medsam_model, device, args.output_dir)
    extract_features(data_loader_test, mae_model, medsam_model, device, args.output_dir)

    print("Feature extraction complete!")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, f'mae_{args.input_size}_sam_{args.sam_size}')
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
