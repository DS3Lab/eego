# extract EEG features
from . import data_loading_helpers as dh
import config
import nltk
import numpy as np


# get raw EEG features (mean word level activity)
def extract_raw_eeg(sentence_data, sentence_dict):
    """extract tokens of all sentences."""

    for tup in sentence_data:

        # read Matlab v7.3 structures
        f = tup[0]
        s_data = tup[1]
        rawData = s_data['rawData']
        contentData = s_data['content']

        for idx in range(len(rawData)):

            raw_sent_eeg_ref = rawData[idx][0]
            raw_sent_eeg = f[raw_sent_eeg_ref]
            #print(type(raw_sent_eeg))
            print(raw_sent_eeg.shape)
            mean_raw_sent_eeg = np.nanmean(raw_sent_eeg, axis=1)
            print(mean_raw_sent_eeg.shape)

            obj_reference_content = contentData[idx][0]
            sent = dh.load_matlab_string(f[obj_reference_content])
            # whitespace tokenization
            split_tokens = sent.split()
            # linguistic tokenization
            spacy_tokens = nltk.word_tokenize(sent)

            # for sentiment and relation detection
            if config.class_task.startswith('sentiment') or config.class_task == "reldetect":
                if sent not in sentence_dict:
                    sentence_dict[sent] = split_tokens
                else:
                    print('duplicate!')

            # for ner (different tokenization needed for NER)
            if config.class_task == "ner":
                if sent not in sentence_dict:
                    sentence_dict[sent] = spacy_tokens

                else:
                    print('duplicate!')


# get raw EEG features for each power spectrum