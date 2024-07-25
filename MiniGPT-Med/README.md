# MiniGPT-Med: Large Language Model as a General Interface for Radiology Diagnosis
Asma Alkhaldi, Raneem Alnajim, Layan Alabdullatef, Rawan Alyahya, Jun Chen, Deyao Zhu, Ahmed Alsinan, Mohamed Elhoseiny

*Saudi Data and Artificial Intelligence Authority (SDAIA) and King Abdullah University of Science and Technology (KAUST)*

## Installation
```bash
git clone https://github.com/Vision-CAIR/MiniGPT-Med
cd MiniGPT-Med
conda env create -f environment.yml
conda activate miniGPT-Med
```

## Download miniGPT-Med trained model weights

* miniGPT-Med's weights [miniGPT-Med Model](https://drive.google.com/file/d/1kjGLk6s9LsBmXfLWQFCdlwF3aul08Cl8/view?usp=sharing)

* Then modify line 8 at miniGPT-Med/eval_configs/minigptv2_eval.yaml to be the path of miniGPT-Med weight.

## Prepare weight for LLMs

### Llama2 Version

```shell
git clone https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
```

Then modify line 14 at miniGPT-Med/minigpt4/configs/models/minigpt_v2.yaml to be the path of Llama-2-13b-chat-hf.

### Better Download of Llama2

##### Prepare hfd
```shell
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
```

##### Install aria2 and git-lfs
```shell
sudo apt install aira2
sudo apt install git-lfs
```

##### Download Llama2
```shell
export HF_ENDPOINT="https://hf-mirror.com"
./hfd.sh NousResearch/Llama-2-7b-chat-hf --tool aria2c -x 16
```
