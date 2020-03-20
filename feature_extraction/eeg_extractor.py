# extract EEG features
from . import data_loading_helpers as dh
import config
import nltk
import numpy as np


# get raw EEG features (mean sentence level activity)
def extract_raw_eeg(sentence_data, eeg_dict):
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
            mean_raw_sent_eeg = np.nanmean(raw_sent_eeg, axis=0)

            obj_reference_content = contentData[idx][0]
            sent = dh.load_matlab_string(f[obj_reference_content])

            # for sentiment and relation detection
            if config.class_task.startswith('sentiment') or config.class_task == "reldetect":
                if sent not in eeg_dict:
                    eeg_dict[sent] = {'mean_raw_sent_eeg': [mean_raw_sent_eeg]}
                else:
                    eeg_dict[sent]['mean_raw_sent_eeg'].append(mean_raw_sent_eeg)
                    #print('duplicate!')

            # for ner (different tokenization needed for NER)
            #if config.class_task == "ner":
            # todo: how to handle for word-level models?

