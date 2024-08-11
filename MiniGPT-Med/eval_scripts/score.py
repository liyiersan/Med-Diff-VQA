

import sys
sys.path.append('./MiniGPT-Med')
import json
import argparse


from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


pred_path = './MiniGPT-Med/experiments/preds_38.json'

data = json.load(open(pred_path))


gts = {}
preds = {}

for key in data.keys():
    imgId = key
    answer = data[key][0]['answer']
    gt_answer = data[key][0]['gt_answer'].lstrip().rstrip()
    gt_data = [{'study_id': imgId, 'caption': f'{gt_answer}'}]
    gts[imgId] = gt_data
    pred_data = [{'study_id': imgId, 'caption': f'{answer}'}]
    preds[imgId] = pred_data
    

# =================================================
# Set up scorers
# =================================================
print('tokenization...')
tokenizer = PTBTokenizer()
gts  = tokenizer.tokenize(gts)
res = tokenizer.tokenize(preds)

# =================================================
# Set up scorers
# =================================================
print('setting up scorers...')
scorers = [
    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    (Meteor(),"METEOR"),
    (Rouge(), "ROUGE_L"),
    (Cider(), "CIDEr"),
    # (Spice(), "SPICE")
]

# =================================================
# Compute scores
# =================================================
for scorer, method in scorers:
    print('computing %s score...'%(scorer.method()))
    score, scores = scorer.compute_score(gts, res)
    if type(method) == list:
        for sc, scs, m in zip(score, scores, method):
            print("%s: %0.3f"%(m, sc))
    else:
        print("%s: %0.3f"%(method, score))