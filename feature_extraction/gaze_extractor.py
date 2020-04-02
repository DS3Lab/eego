from . import data_loading_helpers as dh
import numpy as np
import config


def word_level_et_features(sentence_data, gaze_dict):
    """extract word level eye-tracking features from Matlab files"""

    for tup in sentence_data:

        # read Matlab v7.3 structures
        f = tup[0]
        s_data = tup[1]
        rawData = s_data['rawData']
        contentData = s_data['content']
        wordData = s_data['word']

        # nFix: number of fixations
        # FFD: first fixation duration
        # TRT: total reading time
        # GD: gaze duration
        # GPT: go-past time
        gaze_features = ['nFix', 'FFD', 'TRT', 'GD', 'GPT']

        for idx in range(len(rawData)):

            obj_reference_content = contentData[idx][0]
            sent = dh.load_matlab_string(f[obj_reference_content])

            sent_features = {}
            # get word level data
            try:
                word_data = dh.extract_word_level_data(f, f[wordData[idx][0]],
                                                   eeg_float_resolution=dh.eeg_float_resolution)
                #if word_data:
                for widx in range(len(word_data)):
                    word = word_data[widx]['content']
                    word_feats = []
                    for feature in gaze_features:
                        if word_data[widx][feature] is not None:
                            word_feats.append(float(word_data[widx][feature]))
                        else:
                            word_feats.append(np.nan)
                    #if word_feats:
                    sent_features[widx] = word_feats
            #else:
             #   print("NO word data available!")
            except ValueError:
                print("NO sentence data available!")

            # for sentiment and relation detection
            #if sent_features:
             #   print(sent_features)
            #if config.class_task.startswith('sentiment') or config.class_task == "reldetect":
            if sent not in gaze_dict:
                gaze_dict[sent] = {}
                for widx, fts in sent_features.items():
                    gaze_dict[sent][widx] = [fts]
            else:
                for widx, fts in sent_features.items():
                    if not widx in gaze_dict[sent]:
                        gaze_dict[sent][widx] = [fts]
                    else:
                        gaze_dict[sent][widx].append(sent_features[widx])

