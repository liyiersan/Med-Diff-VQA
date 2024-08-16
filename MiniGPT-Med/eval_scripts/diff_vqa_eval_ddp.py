import os
import sys
sys.path.append('./MiniGPT-Med')
import json
import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from minigpt4.common.config import Config
from minigpt4.conversation.conversation import CONV_VISION_minigptv2

from minigpt4.datasets.datasets.diff_vqa_dataset import EvalDiffVQADataset
from minigpt4.common.registry import registry
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

def list_of_str(arg):
    return list(map(str, arg.split(',')))

def eval_parser():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument('--local_rank', type=int, default=0,
                    help="passed by torch.distributed.launch module")
    parser.add_argument('--local-rank', type=int, default=0,
                    help="passed by torch.distributed.launch module, for pytorch >=2.0")
    parser.add_argument("--cfg-path", type=str, default="./MiniGPT-Med/eval_configs/diff_vqa_eval.yaml", help="path to config file")
    parser.add_argument("--name", type=str, default='A2', help="evaluation name")
    parser.add_argument("--ckpt", type=str, help="path to configuration file.")
    parser.add_argument("--eval_opt", type=str, default='all', help="path to configuration file.")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="max number of generated tokens")
    parser.add_argument("--batch_size", type=int, default=16)  # Reduce batch size
    parser.add_argument("--lora_r", type=int, default=64, help="lora rank of the model")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    return parser

def init_model(args, rank):
    print(f'Initialization Model on rank {rank}')
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(rank)
    model = DDP(model, device_ids=[rank])

    key = 'diff_vqa'
    diff_vqa_cfg = cfg.evaluation_datasets_cfg.get(key)
    text_processor_cfg = diff_vqa_cfg.text_processor
    text_processor = registry.get_processor_class(text_processor_cfg.name).from_config(text_processor_cfg)
    print('Initialization Finished')
    return model, text_processor, diff_vqa_cfg

def prepare_texts(texts, conv_temp):
    convs = [conv_temp.copy() for _ in range(len(texts))]
    [conv.append_message(
        conv.roles[0], text) for conv, text in zip(convs, texts)]
    [conv.append_message(conv.roles[1], None) for conv in convs]
    texts = [conv.get_prompt() for conv in convs]
    return texts

def main(rank, world_size, args):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    model, text_processor, dataset_cfg = init_model(args, rank)
    model.eval()
    CONV_VISION = CONV_VISION_minigptv2
    conv_temp = CONV_VISION.copy()
    conv_temp.system = ""
    model.eval()
    save_path = Config(args).run_cfg.save_path

    vis_root = dataset_cfg['vis_root']
    ann_path = dataset_cfg['ann_path']
    max_new_tokens = dataset_cfg['max_new_tokens']
    batch_size = dataset_cfg['batch_size']

    gts = {}
    preds = {}
    save_datas = {}

    with torch.no_grad():
        eval_dataset = EvalDiffVQADataset("test", text_processor, vis_root, ann_path)
        sampler = DistributedSampler(eval_dataset)
        data_loader = DataLoader(eval_dataset, batch_size=batch_size, sampler=sampler)
        for i, data in enumerate(data_loader):
            print(f'Rank {rank} processing {i}/{len(data_loader)} batch...')
            ref_features, study_features, instructions, questions, gt_answers, study_ids = data
            images = (ref_features, study_features)
            texts = prepare_texts(instructions, conv_temp)
            answers = model.module.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
            for answer, img_id, question, gt_answer in zip(answers, study_ids, questions, gt_answers):
                imgId = str(img_id.item())
                gt_answer = gt_answer.lstrip().rstrip() # remove leading and trailing whitespaces
                gt_data = [{'study_id': imgId, 'caption': f'{gt_answer}'}]
                gts[imgId] = gt_data
                pred_data = [{'study_id': imgId, 'caption': f'{answer}'}]
                preds[imgId] = pred_data
                save_data = [{'id': imgId, 'answer': f'{answer}', 'question': f'{question}', 'gt_answer': f'{gt_answer}'}]
                save_datas[imgId] = save_data 
                
    dist.barrier()
    # Aggregate results from all processes
    gathered_gts = [None for _ in range(world_size)]
    gathered_preds = [None for _ in range(world_size)]
    gathered_save_datas = [None for _ in range(world_size)]

    dist.all_gather_object(gathered_gts, gts)
    dist.all_gather_object(gathered_preds, preds)
    dist.all_gather_object(gathered_save_datas, save_datas)

    if rank == 0:
        # Merge results
        merged_gts = {}
        merged_preds = {}
        merged_save_datas = {}

        for d in gathered_gts:
            merged_gts.update(d)

        for d in gathered_preds:
            merged_preds.update(d)

        for d in gathered_save_datas:
            merged_save_datas.update(d)

        # Save the results
        print('saving the results...')
        with open(f'{save_path}/preds.json', 'w+') as f:
            json.dump(merged_save_datas, f)
        
        # Tokenize and score
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(merged_gts)
        res = tokenizer.tokenize(merged_preds)

        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")
        ]

        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.3f"%(m, sc))
            else:
                print("%s: %0.3f"%(method, score))

if __name__ == "__main__":
    parser = eval_parser()
    parser.add_argument("--gt_file", type=str, default='vqa_prepare/data/mimic_gt_captions_test.json', help="ground truth files")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    rank = int(os.environ['RANK'])
    main(rank, world_size, args)
