from . import data_loading_helpers as dh
import config
import numpy as np


def extract_word_raw_eeg(sentence_data, eeg_dict):
    """extract word-level raw EEG data of all sentences.
    word-level EEG data = mean activity over all fixations of a word"""

    for tup in sentence_data:

        # read Matlab v7.3 structures
        f = tup[0]
        s_data = tup[1]
        rawData = s_data['rawData']
        contentData = s_data['content']
        wordData = s_data['word']

        for idx in range(len(rawData)):

            obj_reference_content = contentData[idx][0]
            sent = dh.load_matlab_string(f[obj_reference_content])

            sent_features = {}
            # get word level data

            #try:
            word_data = dh.extract_word_level_data(f, f[wordData[idx][0]],
                                               eeg_float_resolution=dh.eeg_float_resolution)

            #if word_data:
            for widx in range(len(word_data)):
                word = word_data[widx]['content']
                fixations_eeg = word_data[widx]["RAW_EEG"]

                word_eeg = []
                if len(fixations_eeg) > 0:
                    print(len(fixations_eeg))
                    for fixation in fixations_eeg:
                        fix = np.nanmean(fixation, axis=0)
                        word_eeg.append(fix)

                    print(len(word_eeg))
                    # average over multiple fixations
                    print(word_eeg)
                    print(len(word_eeg[0]))
                    word_eeg = np.nanmean(word_eeg, axis=0)
                    print(len(word_eeg))
                    sent_features[widx] = word_eeg
                else:
                    nan_array = np.empty((105,))
                    nan_array[:] = np.NaN
                    sent_features[widx] = nan_array
                #else:
                    #print("NO word data available!")
            #except ValueError:
             #   print("NO sentence data available!")

            #if sent_features:
            # for sentiment and relation detection
            if config.class_task.startswith('sentiment') or config.class_task == "reldetect":
                if sent not in eeg_dict:
                    eeg_dict[sent] = {}
                    for widx, fts in sent_features.items():
                        eeg_dict[sent][widx] = [fts]
                else:
                    for widx, fts in sent_features.items():
                        if not widx in eeg_dict[sent]:
                            eeg_dict[sent][widx] = [fts]
                        else:
                            eeg_dict[sent][widx].append(sent_features[widx])




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
            #print(mean_raw_sent_eeg)

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
            mean_sent_t = mean_sent_t[:, 0]
            print(mean_sent_t.shape)

            obj_reference_content = contentData[idx][0]
            sent = dh.load_matlab_string(f[obj_reference_content])

            # for sentiment and relation detection
            if config.class_task.startswith('sentiment') or config.class_task == "reldetect":
                if sent not in eeg_dict:
                    eeg_dict[sent] = {config.feature_set[0]+'_sent_eeg': [mean_sent_t]}
                else:
                    eeg_dict[sent][config.feature_set[0]+'_sent_eeg'].append(mean_sent_t)
