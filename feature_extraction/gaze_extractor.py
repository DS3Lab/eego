from . import data_loading_helpers as dh

# extract eye-tracking features: nFix, FFD, TRT, GD, GPT


def word_level_et_features(f, sentence_data):
    """extract word level eye-tracking features from Matlab struct"""

    gaze_features = ['nFix', 'FFD', 'TRT', 'GD', 'GPT']

    rawData = sentence_data['rawData']
    contentData = sentence_data['content']
    wordData = sentence_data['word']

    all_sents_features = {}

    for idx in range(len(rawData)):
        obj_reference_content = contentData[idx][0]
        sent = dh.load_matlab_string(f[obj_reference_content])
        #print(sent)
        tokens = sent.split()

        # get word level data
        word_data = dh.extract_word_level_data(f, f[wordData[idx][0]],
                                               eeg_float_resolution=dh.eeg_float_resolution)

        # todo: check that no. of tokens and feature values are the same
        sentence_features = {'tokens': tokens, 'nFix': [], 'FFD': [], 'TRT': [], 'GD': [], 'GPT': []}
        if word_data:
            for widx in range(len(word_data)):
                for feature in gaze_features:
                    if word_data[widx][feature] is not None:
                        sentence_features[feature].append(float(word_data[widx][feature]))
                    else:
                        sentence_features[feature].append(0.0)

            #print(sentence_features)

        else:
            print("NO word data available!")

        all_sents_features[idx] = sentence_features

    print(len(all_sents_features))
