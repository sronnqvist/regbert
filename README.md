# Fine-tune BERT for register classification

Usage: `python train_classifier.py`

## Configure data source

In python files below specify `DATA_PATH` (e.g., /scratch/project_2002026/multilabel_bert/neuro_classifier/multilabel/CORE-final). `read_tsv()` parameter `sentence_index` specifies column number of text content (e.g., 1).


# Fine-tune Transformer-XL for register classification (under construction)

Usage: `python train_xl.py`

## Configuration

Initialize configuration file `config.ini` with `sh init_config.sh`. Configure `data` path and `sentence_index` according to your data.
