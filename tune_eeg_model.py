import config
from feature_extraction import zuco_reader
from reldetect import reldetect_model
from ner import ner_model
from sentiment import sentiment_model, sentiment_eeg_model, sentiment_combi_model
from data_helpers import save_results, load_matlab_files


# Usage on spaceml:
# $ conda activate env-eego
# $ CUDA_VISIBLE_DEVICES=7 python tune_model.py


def main():
    feature_dict = {}
    label_dict = {}
    eeg_dict = {}
    print("TASK: ", config.class_task)
    print("extracting", config.feature_set, "features....")
    for subject in config.subjects:
        print(subject)

        loaded_data = load_matlab_files(config.class_task, subject)

        zuco_reader.extract_features(loaded_data, config.feature_set, feature_dict, eeg_dict)
        zuco_reader.extract_labels(feature_dict, label_dict, config.class_task, subject)

    print(len(feature_dict), len(label_dict), len(eeg_dict))
    if len(feature_dict) != len(label_dict) != len(eeg_dict):
        print("WARNING: Not an equal number of sentences in features and labels!")

    for rand in config.random_seed_values:
        for emb in config.embeddings:
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
                                                fold_results = reldetect_model.lstm_classifier(feature_dict, label_dict,
                                                                                               emb, parameter_dict,
                                                                                               rand, threshold)
                                                save_results(fold_results, config.class_task)

                                        elif config.class_task == 'ner':
                                            fold_results = ner_model.lstm_classifier(feature_dict, label_dict, emb,
                                                                                     parameter_dict, rand)
                                            save_results(fold_results, config.class_task)

                                        elif config.class_task == 'sentiment-tri':
                                            fold_results = sentiment_model.lstm_classifier(feature_dict, label_dict,
                                                                                           emb, parameter_dict, rand)
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
                                                fold_results = sentiment_combi_model.lstm_classifier(feature_dict,
                                                                                                   label_dict, eeg_dict,
                                                                                                   emb, parameter_dict,
                                                                                                   rand)
                                            else:
                                                fold_results = sentiment_eeg_model.lstm_classifier(feature_dict,
                                                                                                   label_dict, eeg_dict,
                                                                                                   emb, parameter_dict,
                                                                                                   rand)
                                            save_results(fold_results, config.class_task)


if __name__ == '__main__':
    main()
