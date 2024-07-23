import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

MIMIC_DEFAULT_MEAN = (0.485, 0.456, 0.406)
MIMIC_DEFAULT_STD = (0.229, 0.224, 0.225)


# data augmentation, modified from https://github.com/lambert-x/medical_mae/main_pretrain_multi_datasets_xray.py
def build_transform(is_pretrain, args):
    dataset_mean = MIMIC_DEFAULT_MEAN
    dataset_std = MIMIC_DEFAULT_STD
    if is_pretrain:
        if args.random_resize_range:
            resize_ratio_min, resize_ratio_max = args.random_resize_range
            print(resize_ratio_min, resize_ratio_max)
            transform = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size, scale=(resize_ratio_min, resize_ratio_max),
                                                interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(dataset_mean, dataset_std)])
        else:
            print('Using Directly-Resize Mode. (no RandomResizedCrop)')
            transform = transforms.Compose([
                transforms.Resize((args.input_size, args.input_size), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(dataset_mean, dataset_std)])
    else:
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)])
    return transform

class MimicImageDataset(Dataset):
    def __init__(self, data_dir="/data/mimic-cxr-jpg-2.1.0/", flag='train', args=None, is_pretrain=True):
        csv_name = flag + '_mimic-metadata.csv'
        self.data_frame = pd.read_csv(os.path.join(data_dir, csv_name))
        self.ref_img_path_list = self.data_frame['ref_path']
        self.study_img_path_list = self.data_frame['study_path']
        assert len(self.ref_img_path_list) == len(self.study_img_path_list), 'ref and study image number not equal'
        self.transform = build_transform(is_pretrain, args=args)
    def __len__(self):
        return len(self.ref_img_path_list)

    def __getitem__(self, idx):
        ref_img_path = self.ref_img_path_list[idx]
        # load the png image with size 1024x1024
        ref_img_path = os.path.join('/data/mimic-cxr-jpg-2.1.0/mimic_cxr_png', ref_img_path.split('/')[-1].replace('jpg', 'png'))
        study_img_path = self.study_img_path_list[idx]
        study_img_path = os.path.join('/data/mimic-cxr-jpg-2.1.0/mimic_cxr_png', study_img_path.split('/')[-1].replace('jpg', 'png'))
        
        ref_img = Image.open(ref_img_path).convert('RGB')
        study_img = Image.open(study_img_path).convert('RGB')
        
        if self.transform:
            ref_img = self.transform(ref_img)
            study_img = self.transform(study_img)
        
        return ref_img, study_img

    
def split_train_val_test(data_dir, csv_file):
    file_path = os.path.join(data_dir, csv_file)
    data_frame = pd.read_csv(file_path)
    
    # split data according to the split column
    train_data = data_frame[data_frame['split'] == 'train']
    val_data = data_frame[data_frame['split'] == 'val']
    test_data = data_frame[data_frame['split'] == 'test']
    
    save_name = 'mimic-metadata.csv'
    
    # save the split data to csv files
    train_csv_file = os.path.join(data_dir, 'train_' + save_name)
    val_csv_file = os.path.join(data_dir, 'val_' + save_name)
    test_csv_file = os.path.join(data_dir, 'test_' + save_name)
    
    train_data.to_csv(train_csv_file, index=False)
    val_data.to_csv(val_csv_file, index=False)
    test_data.to_csv(test_csv_file, index=False)
    
    
if __name__ == '__main__':
    data_dir = '/data/mimic-cxr-jpg-2.1.0'
    csv_file = 'diff_info.csv'
    split_train_val_test(data_dir, csv_file)
    
    