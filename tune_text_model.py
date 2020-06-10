import config
from feature_extraction import zuco_reader
from reldetect import reldetect_text_model
from ner import ner_text_model
from sentiment import sentiment_text_model
from data_helpers import save_results, load_matlab_files, drop_sentiment_sents
import collections
import numpy as np

# Usage on spaceml:
# $ conda activate env-eego
# $ CUDA_VISIBLE_DEVICES=7 python tune_text_model.py


def main():
    feature_dict = {}
    label_dict = {}
    eeg_dict = {}
    gaze_dict = {}
    print("TASK: ", config.class_task)
    print("extracting", config.feature_set, "features....")
    for subject in config.subjects:
        print(subject)

        loaded_data = load_matlab_files(config.class_task, subject)

        zuco_reader.extract_features(loaded_data, config.feature_set, feature_dict, eeg_dict, gaze_dict)
        zuco_reader.extract_labels(feature_dict, label_dict, config.class_task, subject)

    feature_dict = collections.OrderedDict(sorted(feature_dict.items()))
    label_dict = collections.OrderedDict(sorted(label_dict.items()))
    print(len(feature_dict.keys()), len(label_dict))

    if len(feature_dict) != len(label_dict):
        print("WARNING: Not an equal number of sentences in features and labels!")

    # test with less data
    print(len(eeg_dict), len(label_dict), len(feature_dict))
    drop_sentiment_sents(label_dict, feature_dict, eeg_dict)
    print(len(eeg_dict), len(label_dict), len(feature_dict))

    for rand in config.random_seed_values:
        np.random.seed(rand)
        for lstmDim in config.lstm_dim:
            for lstmLayers in config.lstm_layers:
                for denseDim in config.dense_dim:
                    for drop in config.dropout:
                        for bs in config.batch_size:
                            for lr_val in config.lr:
                                for e_val in config.epochs:
                                    parameter_dict = {"lr": lr_val, "lstm_dim": lstmDim, "lstm_layers": lstmLayers,
                                                      "dense_dim": denseDim, "dropout": drop, "batch_size": bs,
                                                      "epochs": e_val, "random_seed": rand}

                                    if config.class_task == 'reldetect':
                                        for threshold in config.rel_thresholds:
                                            fold_results = reldetect_text_model.lstm_classifier(feature_dict, label_dict, config.embeddings, parameter_dict, rand, threshold)
                                            save_results(fold_results, config.class_task)

                                    elif config.class_task == 'ner':
                                        fold_results = ner_text_model.lstm_classifier(feature_dict, label_dict, config.embeddings, parameter_dict, rand)
                                        save_results(fold_results, config.class_task)

                                    elif config.class_task == 'sentiment-tri':
                                        fold_results = sentiment_text_model.lstm_classifier(feature_dict, label_dict, config.embeddings, parameter_dict, rand)
                                        save_results(fold_results, config.class_task)

                                    elif config.class_task == 'sentiment-bin':
                                        for s, label in list(label_dict.items()):
                                            # drop neutral sentences for binary sentiment classification
                                            if label == 2:
                                                    del label_dict[s]
                                                    del feature_dict[s]
                                        print(len(feature_dict), len(label_dict))
                                        fold_results = sentiment_text_model.lstm_classifier(feature_dict, label_dict, config.embeddings, parameter_dict, rand)
                                        save_results(fold_results, config.class_task)


if __name__ == '__main__':
    main()
