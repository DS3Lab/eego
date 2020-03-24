from . import data_loading_helpers as dh
import nltk
import numpy as np
import config

# extract eye-tracking features: nFix, FFD, TRT, GD, GPT


def word_level_et_features(sentence_data, gaze_dict):
    """extract word level eye-tracking features from Matlab files"""

    for tup in sentence_data:

        # read Matlab v7.3 structures
        f = tup[0]
        s_data = tup[1]
        rawData = s_data['rawData']
        contentData = s_data['content']
        wordData = s_data['word']

        gaze_features = ['nFix', 'FFD', 'TRT', 'GD', 'GPT']

        for idx in range(len(rawData)):

            #raw_sent_eeg_ref = rawData[idx][0]
            #raw_sent_eeg = f[raw_sent_eeg_ref]
            #mean_raw_sent_eeg = np.nanmean(raw_sent_eeg, axis=0)
            # print(mean_raw_sent_eeg)

            obj_reference_content = contentData[idx][0]
            sent = dh.load_matlab_string(f[obj_reference_content])
            # whitespace tokenization
            split_tokens = sent.split()
            # linguistic tokenization
            spacy_tokens = nltk.word_tokenize(sent)

            sent_features = {}
            # get word level data
            try:
                word_data = dh.extract_word_level_data(f, f[wordData[idx][0]],
                                                   eeg_float_resolution=dh.eeg_float_resolution)

                if word_data:
                    for widx in range(len(word_data)):
                        word = word_data[widx]['content']
                        word_feats = []
                        for feature in gaze_features:
                            if word_data[widx][feature] is not None:
                                word_feats.append(float(word_data[widx][feature]))
                            else:
                                word_feats.append(0.0)
                        sent_features[widx] = [word_feats]
                else:
                    print("NO word data available!")
            except ValueError:
                print("NO sentence data available!")

            # for sentiment and relation detection
            if config.class_task.startswith('sentiment') or config.class_task == "reldetect":
                if sent not in gaze_dict:
                    gaze_dict[sent] = sent_features
                    #print(gaze_dict[sent])
                else:
                    for widx, fts in gaze_dict[sent].items():
                        gaze_dict[sent][widx].append(sent_features[widx])

