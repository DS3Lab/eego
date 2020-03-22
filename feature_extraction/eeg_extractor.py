# extract EEG features
from . import data_loading_helpers as dh
import config
import nltk
import numpy as np


# get raw EEG features (mean sentence level activity)
def extract_sent_raw_eeg(sentence_data, eeg_dict):
    """extract sentence-level raw EEG data of all sentences."""

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
            print(mean_raw_sent_eeg)

            obj_reference_content = contentData[idx][0]
            sent = dh.load_matlab_string(f[obj_reference_content])

            # for sentiment and relation detection
            if config.class_task.startswith('sentiment') or config.class_task == "reldetect":
                if sent not in eeg_dict:
                    eeg_dict[sent] = {'mean_raw_sent_eeg': [mean_raw_sent_eeg]}
                else:
                    eeg_dict[sent]['mean_raw_sent_eeg'].append(mean_raw_sent_eeg)

            # for ner (different tokenization needed for NER)
            #if config.class_task == "ner":
            # todo: how to handle for word-level models? of fixation level? or timestamp level?


def get_freq_band_data():

    if 'eeg_theta' in config.feature_set:
        band1 = 'mean_t1'
        band2 = 'mean_t2'

    if 'eeg_alpha' in config.feature_set:
        band1 = 'mean_a1'
        band2 = 'mean_a2'

    if 'eeg_beta' in config.feature_set:
        band1 = 'mean_b1'
        band2 = 'mean_b2'

    if 'eeg_gamma' in config.feature_set:
        band1 = 'mean_g1'
        band2 = 'mean_g2'

    return band1, band2


def extract_sent_freq_eeg(sentence_data, eeg_dict):
    """extract sentence-level frequency band EEG of all sentences."""

    for tup in sentence_data:

        # read Matlab v7.3 structures
        f = tup[0]
        s_data = tup[1]

        band1, band2 = get_freq_band_data()

        meanB1data = s_data[band1]
        meanB2data = s_data[band2]
        contentData = s_data['content']

        for idx in range(len(meanB1data)):

            sent_t1_ref = meanB1data[idx][0]
            sent_t1 = f[sent_t1_ref]

            sent_t2_ref = meanB2data[idx][0]
            sent_t2 = f[sent_t2_ref]

            mean_sent_t = (np.array(sent_t1) + np.array(sent_t2)) / 2.0
            print(mean_sent_t.shape)
            mean_sent_t = mean_sent_t[:, :, 0]
            print(mean_sent_t.shape)

            obj_reference_content = contentData[idx][0]
            sent = dh.load_matlab_string(f[obj_reference_content])

            # for sentiment and relation detection
            if config.class_task.startswith('sentiment') or config.class_task == "reldetect":
                if sent not in eeg_dict:
                    eeg_dict[sent] = {'mean_raw_sent_eeg': [mean_sent_t]}
                else:
                    eeg_dict[sent]['mean_raw_sent_eeg'].append(mean_sent_t)