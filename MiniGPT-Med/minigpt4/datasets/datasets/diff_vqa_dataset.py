import os
from einops import rearrange
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class DiffVQADataset(Dataset):
    def __init__(self, flag, text_processor, vis_root, ann_path):
        self.text_processor = text_processor
        self.flag = flag
        csv_name = flag + '_mimic-metadata.csv'
        self.data_frame = pd.read_csv(os.path.join(ann_path, csv_name))
        self.questions = self.data_frame['question']
        self.answers = self.data_frame['answer']
        self.feature_dir = vis_root
        self.ref_img_path_list = self.data_frame['ref_path']
        self.study_img_path_list = self.data_frame['study_path']
        
        assert len(self.questions) == len(self.answers), 'Lengths do not match'
        
        self.instruction_pool = [
            'In the first reference image and the second main image, focus on the differences and answer, {}',
            'The first is the reference image and the second is the main image, note any changes and answer, {}',
            'The first image is the reference and the second is the main, identify any differences and answer, {}',
            "Based on the two images, respond to this question with a short answer, {}"
        ]

    def decouple_img_features(self, img_features):
        mae_features = img_features['mae'] # [196, 768]
        mae_features = rearrange(mae_features, '(L n) d -> L (n d)', n=4) # [49, 3072]
        sam_features = img_features['sam'] # [256, 64, 64]
        sam_features = rearrange(sam_features, 'c (h n1) (w n2) -> (h w) (n1 n2 c)', n1=4, n2=4) # [256, 4096]
        return mae_features, sam_features
        # return mae_features # fast debugging

    def __len__(self):
        # # for debug only
        # if self.flag == 'train':
        #     length = 100
        # elif self.flag == 'val':
        #     length = 4
        # elif self.flag == 'test':
        #     length = 8
        # return length

        return len(self.questions)

    def __getitem__(self, index):
        ref_img_id = self.ref_img_path_list[index].split('/')[-1].split('.')[0]
        study_img_id = self.study_img_path_list[index].split('/')[-1].split('.')[0]
        
        # # for debug only
        # ref_img_id = "0a0f60d6-707064a5-828e58b1-f2487506-a63a8869"
        # study_img_id = "0a0f60d6-707064a5-828e58b1-f2487506-a63a8869"
        
        question = self.questions[index]
        answer = self.answers[index]
        
        ref_feature = np.load(os.path.join(self.feature_dir, f'features_{ref_img_id}.npz'))
        ref_feature = self.decouple_img_features(ref_feature)
        study_feature = np.load(os.path.join(self.feature_dir, f'features_{study_img_id}.npz'))
        study_feature = self.decouple_img_features(study_feature)
        
        instruction = random.choice(self.instruction_pool).format(question)
        instruction = f'<Img1><ImageHere></Img1> <Img2><ImageHere><Img2> [vqa] {self.text_processor(instruction)}'


        return {
            "ref_feature": ref_feature,
            "study_feature": study_feature,
            "instruction_input": instruction,
            "answer": answer,
            "ref_image_id": ref_img_id,
            "study_image_id": study_img_id
        }


class EvalDiffVQADataset(DiffVQADataset):
    def __init__(self, flag, text_processor, vis_root, ann_path):
        super().__init__(flag, text_processor, vis_root, ann_path)
        self.study_ids = self.data_frame['study_id']

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        ref_feature = sample['ref_feature']
        study_feature = sample['study_feature']
        instruction = sample['instruction_input']
        study_id = self.study_ids[index]
        answer = sample['answer']
        question = self.questions[index]
        
        return ref_feature, study_feature, instruction, question, answer, study_id