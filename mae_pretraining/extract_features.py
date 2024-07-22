import os
import tqdm
import torch
import argparse
import numpy as np
import medsam
from pathlib import Path
import utils.misc as misc
import models.vit as models_vit
from dataloader.datasets import MimicImageDataset

def get_args_parser():
    parser = argparse.ArgumentParser('MAE Feature Extraction', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int, # 256 for medical_mae
                        help='Batch size for evaluation')
    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str,
                        help='Name of model to extract features')
    parser.add_argument('--pretrain', default=None, type=str,
                    help='model checkpoint path to load')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                    help='Use class token instead of global pool for classification')
    parser.add_argument('--sam-ckpt', default=None, type=str,
                        help='SAM model checkpoint path to load')

    # Dataset parameters
    parser.add_argument('--data_path', default='/data/mimic-cxr-jpg-2.1.0/', type=str,
                        help='dataset path')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--output_dir', default='./img_features', type=str,
                        help='path where to save, empty for no saving')

    # dataloader parameters
    parser.add_argument('--num_workers', default=8, type=int)
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
    np.save(os.path.join(output_dir, f'{flag}_ref_features.npy'), ref_features)
    np.save(os.path.join(output_dir, f'{flag}_study_features.npy'), study_features)
    print(f"Features saved to {output_dir}")

def main(args):
    device = torch.device(args.device)


    dataset_train = MimicImageDataset(data_dir = args.data_path, flag='train', args=args, is_pretrain=False)
    dataset_val = MimicImageDataset(data_dir = args.data_path, flag='false', args=args, is_pretrain=False)
    dataset_test = MimicImageDataset(data_dir = args.data_path, flag='test', args=args, is_pretrain=False)


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False, 
        shuffle=False,
    )
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )
    
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    misc.load_pretrain(model, args.pretrain)

    model.to(device)
    model.eval()
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))


    medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    # Extract features
    
    print("Extracting training features...")
    ref_train_feats, study_train_feats = extract_features(data_loader_train, model, medsam_model, device)
    
    print("Extracting validation features...")
    ref_val_feats, study_val_feats = extract_features(data_loader_val, model, medsam_model, device)
    
    print("Extracting testing features...")
    ref_test_feats, study_test_feats = extract_features(data_loader_test, model, medsam_model, device)
    
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
