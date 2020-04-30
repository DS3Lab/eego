import config
from feature_extraction import zuco_reader
from reldetect import reldetect_eeg_model, reldetect_text_eeg_model
from ner import ner_text_model
from sentiment import sentiment_text_eeg_model_best
from data_helpers import save_results, load_matlab_files
import collections
import json


# Usage on spaceml:
# $ conda activate env-eego
# $ CUDA_VISIBLE_DEVICES=7 python tune_model.py


def main():
    feature_dict = {}
    label_dict = {}
    eeg_dict = {}
    gaze_dict = {}
    print("TASK: ", config.class_task)
    print("Extracting", config.feature_set, "features....")
    for subject in config.subjects:
        print(subject)

        loaded_data = load_matlab_files(config.class_task, subject)

        zuco_reader.extract_features(loaded_data, config.feature_set, feature_dict, eeg_dict, gaze_dict)
        zuco_reader.extract_labels(feature_dict, label_dict, config.class_task, subject)

    print(len(feature_dict), len(label_dict), len(eeg_dict))

    print("Reading EEG features from file!!")
    eeg_dict = json.load(open("../eeg_features/"+config.feature_set[0] + "_feats_file_" + config.class_task + ".json"))
    print("done, ", len(eeg_dict), " sentences with EEG features.")

    feature_dict = collections.OrderedDict(sorted(feature_dict.items()))
    label_dict = collections.OrderedDict(sorted(label_dict.items()))
    eeg_dict = collections.OrderedDict(sorted(eeg_dict.items()))

    if len(feature_dict) != len(label_dict) != len(eeg_dict):
        print("WARNING: Not an equal number of sentences in features and labels!")

    for rand in config.random_seed_values:
        parameter_dict_text = {"lr": config.lr[0], "lstm_dim": config.lstm_dim[0],
                          "dense_dim": config.dense_dim[0], "dropout": config.dropout[0], "batch_size": config.batch_size[0],
                          "epochs": config.epochs[0], "random_seed": rand}
        parameter_dict_eeg = {"lstm_dim": config.eeg_lstm_dim[0],
                               "dense_dim": config.eeg_dense_dim[0], "dropout": config.eeg_dropout[0]}

        if config.class_task == 'reldetect':
            for threshold in config.rel_thresholds:
                if 'eeg_raw' in config.feature_set:
                    fold_results = reldetect_eeg_model.lstm_classifier(label_dict, eeg_dict,
                                                               config.embeddings, parameter_dict_text,
                                                               rand, threshold)
                elif 'combi_eeg_raw' in config.feature_set or 'eeg_theta' in config.feature_set or 'eeg_alpha' in config.feature_set or 'eeg_beta' in config.feature_set or 'eeg_gamma' in config.feature_set:
                    print("this model....")
                    fold_results = reldetect_text_eeg_model.lstm_classifier(feature_dict, label_dict,
                                                                        eeg_dict,
                                                                        emb,
                                                                        parameter_dict,
                                                                        rand, threshold)
                save_results(fold_results, config.class_task)

        elif config.class_task == 'ner':
            fold_results = ner_text_model.lstm_classifier(feature_dict, label_dict, emb,
                                                     parameter_dict, rand)
            save_results(fold_results, config.class_task)

        elif config.class_task == 'sentiment-tri':
            if 'combi_concat' in config.feature_set:
                print("Starting EEG + text combi model")
                fold_results = sentiment_text_eeg_model.lstm_classifier(feature_dict,
                                                                   label_dict, eeg_dict,
                                                                   emb, parameter_dict,
                                                                   rand)
            elif 'eeg_raw' in config.feature_set:
                fold_results = sentiment_eeg_model.lstm_classifier(label_dict,
                                                                     eeg_dict,
                                                                     emb,
                                                                     parameter_dict,
                                                                     rand)
            elif 'combi_eeg_raw' in config.feature_set or 'eeg_theta' in config.feature_set or 'eeg_alpha' in config.feature_set or 'eeg_beta' in config.feature_set or 'eeg_gamma' in config.feature_set:
                fold_results = sentiment_text_eeg_model.lstm_classifier(feature_dict, label_dict,
                                                                        eeg_dict,
                                                                        emb,
                                                                        parameter_dict,
                                                                        rand)
            save_results(fold_results, config.class_task)
        elif config.class_task == 'sentiment-bin':
            for s, label in list(label_dict.items()):
                # drop neutral sentences for binary sentiment classification
                if label == 2:
                    del label_dict[s]
                    del feature_dict[s]
                    del eeg_dict[s]
            if 'combi_concat' in config.feature_set:
                print("Starting EEG + text combi model")
                fold_results = sentiment_text_eeg_model.lstm_classifier(feature_dict,
                                                                   label_dict, eeg_dict,
                                                                   emb, parameter_dict,
                                                                   rand)
            elif 'eeg_raw' in config.feature_set:
                fold_results = sentiment_eeg_model.lstm_classifier(label_dict,
                                                                        eeg_dict,
                                                                        emb,
                                                                        parameter_dict,
                                                                        rand)

            elif 'combi_eeg_raw' in config.feature_set or 'eeg_theta' in config.feature_set or 'eeg_alpha' in config.feature_set or 'eeg_beta' in config.feature_set or 'eeg_gamma' in config.feature_set:
                fold_results = sentiment_text_eeg_model_best.lstm_classifier(feature_dict, label_dict,
                                                                        eeg_dict,
                                                                        config.embeddings,
                                                                        parameter_dict_text, parameter_dict_eeg,
                                                                        rand)

            save_results(fold_results, config.class_task)


if __name__ == '__main__':
    main()
