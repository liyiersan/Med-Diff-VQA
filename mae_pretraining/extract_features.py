import os
import h5py
import tqdm
import torch
import argparse
import numpy as np
from pathlib import Path
import utils.misc as misc
import models.vit as models_vit
import torch.utils.data as torch_data
from dataloader.datasets import MimicImageDataset
from segment_anything import sam_model_registry

def get_args_parser():
    parser = argparse.ArgumentParser('MAE Feature Extraction', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int, # 256 for medical_mae
                        help='Batch size for evaluation')
    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str,
                        help='Name of model to extract features')
    parser.add_argument('--pretrain', default='./mae_pretraining/pretrained/vit-b_CXR_0.5M_mae.pth', type=str,
                    help='model checkpoint path to load')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                    help='Use class token instead of global pool for classification')
    parser.add_argument('--sam_ckpt', default='./mae_pretraining/pretrained/sam_vit_b_01ec64.pth', type=str,
                        help='SAM model checkpoint path to load')

    # Dataset parameters
    parser.add_argument('--data_path', default='/data/mimic-cxr-jpg-2.1.0/', type=str,
                        help='dataset path')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--output_dir', default='./pretrained_img_features', type=str,
                        help='path where to save, empty for no saving')

    # dataloader parameters
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser

def forward_features(imgs, mae_model, sam_model):
    with torch.no_grad():
        mae_features = mae_model.forward_features(imgs) # (B, 768)
        sam_features = sam_model.image_encoder(imgs) # (B, 256, 14, 14)
        sam_features = torch.mean(sam_features, dim=(2, 3)) # (B, 256)
        output = torch.cat((mae_features, sam_features), dim=1) # (B, 1024)        
    return output.cpu().numpy()

def extract_features(data_loader, mae_model, sam_model, device):
    ref_features = []
    study_features = []
    # add tqdm for progress bar
    for (ref_imgs, study_imgs) in tqdm.tqdm(data_loader):
        ref_imgs = ref_imgs.to(device)
        study_imgs = study_imgs.to(device)
        ref_feats = forward_features(ref_imgs, mae_model, sam_model)
        study_feats = forward_features(study_imgs, mae_model, sam_model)
        ref_features.append(ref_feats)
        study_features.append(study_feats)
            
    return np.vstack(ref_features), np.vstack(study_features) # (N, 1024)

def save_features(ref_features, study_features, output_dir, flag='train'):
    h5_file_path = os.path.join(output_dir, f'{flag}_features.h5')
    
    with h5py.File(h5_file_path, 'w') as f:
        f.create_dataset(f'{flag}_ref_features', data=ref_features, compression='gzip')
        f.create_dataset(f'{flag}_study_features', data=study_features, compression='gzip')
    
    print(f"Features saved to {h5_file_path}")

def main(args):
    device = torch.device(args.device)

    dataset_train = MimicImageDataset(data_dir = args.data_path, flag='train', args=args, is_pretrain=False)
    dataset_val = MimicImageDataset(data_dir = args.data_path, flag='val', args=args, is_pretrain=False)
    dataset_test = MimicImageDataset(data_dir = args.data_path, flag='test', args=args, is_pretrain=False)
    
    # set transform to None
    dataset_train.transform = None
    dataset_val.transform = None
    dataset_test.transform = None

    data_loader_train = torch_data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )

    data_loader_val = torch_data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False, 
        shuffle=False,
    )
    
    data_loader_test = torch_data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )
    
    mae_model = models_vit.__dict__[args.model](
        num_classes=1000,
        global_pool=args.global_pool,
    )

    misc.load_pretrain(mae_model, args.pretrain)

    mae_model.to(device)
    mae_model.eval()
    model_without_ddp = mae_model
    n_parameters = sum(p.numel() for p in mae_model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))


    medsam_model = sam_model_registry["vit_b"](checkpoint=args.sam_ckpt)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    
    # Extract features
    print("Extracting training features...")
    ref_train_feats, study_train_feats = extract_features(data_loader_train, mae_model, medsam_model, device)
    
    print("Extracting validation features...")
    ref_val_feats, study_val_feats = extract_features(data_loader_val, mae_model, medsam_model, device)
    
    print("Extracting testing features...")
    ref_test_feats, study_test_feats = extract_features(data_loader_test, mae_model, medsam_model, device)
    
    # save features to h5 file
    save_features(ref_train_feats, study_train_feats, args.output_dir, flag='train')
    save_features(ref_val_feats, study_val_feats, args.output_dir, flag='val')
    save_features(ref_test_feats, study_test_feats, args.output_dir, flag='test')
    
    print("Done!")
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
