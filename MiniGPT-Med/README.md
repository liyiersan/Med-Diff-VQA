# [MiniGPT-Med](https://github.com/Vision-CAIR/MiniGPT-Med) Finetuning with Pretrained Image Features for [Med-Diff-VQA](https://github.com/Holipori/MIMIC-Diff-VQA)

## Installation
```bash
git clone https://github.com/Vision-CAIR/MiniGPT-Med
cd MiniGPT-Med
conda env create -f environment.yml
conda activate miniGPT-Med
```

## Download miniGPT-Med trained model weights

* miniGPT-Med's weights [miniGPT-Med Model](https://drive.google.com/file/d/1kjGLk6s9LsBmXfLWQFCdlwF3aul08Cl8/view?usp=sharing)

* Then modify line 9 at MiniGPT-Med/train_configs/minigptv2_finetune.yaml to be the path of miniGPT-Med weight.

## Prepare weight for LLMs

### Llama2

```shell
git clone https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
```

Then modify line 14 at MiniGPT-Med/minigpt4/configs/models/minigpt_v2.yaml and line 8 at MiniGPT-Med/train_configs/minigptv2_finetune.yaml to be the path of Llama-2-13b-chat-hf.

### Better Download of Llama2

##### Prepare hfd
```shell
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
```

##### Install aria2 and git-lfs
```shell
sudo apt install aria2
sudo apt install git-lfs
```

##### Download Llama2
```shell
export HF_ENDPOINT="https://hf-mirror.com"
./hfd.sh NousResearch/Llama-2-7b-chat-hf --tool aria2c -x 16
```

## Requirements
```txt
perf==0.2.0
transformers==4.30.0
sentence-transformers==2.2.2
```
For other packages, download the latest as you need via `pip install`.

## Training 

```shell
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigptv2_finetune.yaml
```

## Eval

Modify line 8 and line 9 at MiniGPT-Med/eval_configs/diff_vqa_eval.yaml to be the path of Llama-2-13b-chat-hf and trained model weights.

```shell
python MiniGPT-Med/eval_scripts/diff_vqa_eval.py
```

Currently, the eval script works only when batch_size=1, to speed up evaluation, you can use the ddp script.
```shell
torchrun --nproc_per_node=8 --master_port=8888 MiniGPT-Med/eval_scripts/diff_vqa_eval_ddp.py
```
