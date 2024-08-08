## Prepare the vqa dataset

### Dataset files

Download the necessary files from [data](https://github.com/Holipori/EKAID/tree/main/model/data).

Download the VQA dataset from [Physionet](https://physionet.org/content/medical-diff-vqa/1.0.0/).

Put them in the data folder, and do not forget the `dicom2id.pkl` obtained when [convert_jpg_to_png](mae_pretraining/dataloader/convert.py).

### nltk packages

```shell
pip install nltk
```
Download [punkt.zip](https://github.com/nltk/nltk_data/blob/gh-pages/packages/tokenizers/punkt.zip) and [averaged_perceptron_tagger.zip](https://github.com/nltk/nltk_data/blob/gh-pages/packages/taggers/averaged_perceptron_tagger.zip).

Unzip them and place them in the `./vqa_prepare/nltk_data` folder. The file structure should be follows:
```txt
nltk_data/
├── taggers/
│   └── averaged_perceptron_tagger/
│       └── averaged_perceptron_tagger.pickle
└── tokenizers/
    └── punkt/
```

### Run

```shell
python vqa_prepare/dataset_preparation.py
```

