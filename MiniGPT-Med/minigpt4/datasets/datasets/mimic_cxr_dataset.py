import os
import json
import h5py
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class MimicCxrVQADataset(Dataset):
    def __init__(self, flag, text_processor, vis_root, ann_path):
        self.text_processor = text_processor
        
        csv_name = flag + '_mimic-metadata.csv'
        self.data_frame = pd.read_csv(os.path.join(ann_path, csv_name))
        self.questions = self.data_frame['question']
        self.answers = self.data_frame['answer']
        self.ref_ids = self.data_frame['ref_path']
        self.study_ids = self.data_frame['study_path']
        
        features_name = flag + '_features.h5'
        with h5py.File(os.path.join(vis_root, features_name), 'r') as f:
            self.ref_features = f[f'{flag}_ref_features'][:]
            self.study_features = f[f'{flag}_study_features'][:]
            
        assert len(self.questions) == len(self.answers) == len(self.ref_features) == len(self.study_features), 'Lengths do not match'
        
        self.instruction_pool = [
            'Describe this image in detail',
            'Take a look at this image and describe what you notice',
            'Please provide a detailed description of the picture',
            'Could you describe the contents of this image for me?'
        ]

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, index):
        ref_feature = self.ref_features[index]
        study_feature = self.study_features[index]
        question = self.questions[index]
        answer = self.answers[index]
        instruction = random.choice(self.instruction_pool) + question
        instruction = f'<Img><ImageHere></Img> {self.text_processor(instruction)}'
        ref_id = self.ref_ids[index]
        study_id = self.study_ids[index]

        return {
            "ref_feature": ref_feature,
            "study_feature": study_feature,
            "instruction_input": instruction,
            "answer": answer,
            "ref_image_id": ref_id,
            "study_image_id": study_id,
        }

class MimicCxrDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        # with open(ann_path, 'r') as f:
        #     self.ann = json.load(f)
        self.vis_root = '/data/mimic-cxr-jpg-2.1.0/mimic_cxr_png/'
        self.ann = ["1"] * 1024
    
        self.instruction_pool = [
            'Describe this image in detail',
            'Take a look at this image and describe what you notice',
            'Please provide a detailed description of the picture',
            'Could you describe the contents of this image for me?'
        ]

    def load_image(self, image_id):
        # image_file = f'{image_id}.jpg'
        image_file = f'{image_id}.png'
        image_path = os.path.join(self.vis_root, image_file)
        grayscale_image = Image.open(image_path).convert("L")
        image = Image.new("RGB", grayscale_image.size)
        image.paste(grayscale_image)
        image = self.vis_processor(image)
        return image

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # info = self.ann[index]
        # image = self.load_image(info['image_id'])
        # instruction = random.choice(self.instruction_pool)
        # instruction = f'<Img><ImageHere></Img> {self.text_processor(instruction)}'

        # return {
        #     "image": image,
        #     "instruction_input": instruction,
        #     "answer": info['caption'],
        #     "image_id": info['image_id'],
        # }
        img_id = '5da6598b-f5a17c40-e9bf1a88-b9747692-9f721582'
        image = self.load_image(img_id)
        instruction = random.choice(self.instruction_pool)
        instruction = f'<Img><ImageHere></Img> {self.text_processor(instruction)}'
        image = image.unsqueeze(0)
        
        demo_sample = {
            "image": image,
            "instruction_input": instruction,
            "answer": "Bilateral nodular opacities, which most likely represent nipple shadows, are observed. There is no focal consolidation, pleural effusion, or pneumothorax. Cardiomediastinal silhouette is normal, and there is no acute cardiopulmonary process. Clips project over the left lung, potentially within the breast, and the imaged upper abdomen is unremarkable. Chronic deformity of the posterior left sixth and seventh ribs is noted.",
            "image_id": img_id,
        }
        return demo_sample

#####Eval Classes#####

class evalMIMICDataset(Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

        self.instruction_pool = [
            'Describe this image in detail',
            'Take a look at this image and describe what you notice',
            'Please provide a detailed description of the picture',
            'Could you describe the contents of this image for me?'
        ]

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        info = self.loaded_data[idx]
        img_id = '{}.jpg'.format(info['image_id'])
        image_path = os.path.join(self.root_path, img_id)
        grayscale_image = Image.open(image_path).convert("L")
        image = Image.new("RGB", grayscale_image.size)
        image.paste(grayscale_image)
        image = self.vis_processor(image)

        answer = info['caption']
        question = random.choice(self.instruction_pool)

        return image, question, img_id
    
    
class evalDetectMimicDataset(Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_id = data['key']
        sent = data['objects']
        image_path = os.path.join(self.root_path, img_id)
        grayscale_image = Image.open(image_path).convert("L")
        image = Image.new("RGB", grayscale_image.size)
        image.paste(grayscale_image)
        image = self.vis_processor(image)
        question = f"[detection] {sent}"

        return image, question, img_id