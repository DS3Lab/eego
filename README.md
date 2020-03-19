# eego
NLP experiments with ZuCo for COLING 2020

Usage on spaceml:  
`$ conda activate /mnt/ds3lab-scratch/noraho/anaconda3/envs/env-eego`  
`$ CUDA_VISIBLE_DEVICES=7 python tune_model.py`

Set configuration to train or tune in  `config.py`.

## Feature extraction
ZuCo datasets available at: `/mnt/ds3lab-scratch/noraho/coling2020`

Word embeddings available at: `/mnt/ds3lab-scratch/noraho/embeddings/glove-6B/` and `/mnt/ds3lab-scratch/noraho/embeddings/bert/`

## NER

sequence to sequence word-level classification

## Relation detection

multi-label sentence-level classification

## Sentiment analysis

binary or ternary sentence-level classification
only SR sentences from ZuCo 1

