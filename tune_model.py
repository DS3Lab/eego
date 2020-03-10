import config
from feature_extraction import zuco_reader
from reldetect import reldetect_model
from sentiment import sentiment_model, sentiment_model_bert
from data_helpers import save_results, load_matlab_files


def main():
    feature_dict = {}
    label_dict = {}
    print("TASK: ", config.class_task)
    print("extracting", config.feature_set, "features....")
    for subject in config.subjects:
        print(subject)

        loaded_data = load_matlab_files(config.class_task, subject)

        zuco_reader.extract_features(loaded_data, config.feature_set, feature_dict)
        # todo: fix label extraction for relation detection
        zuco_reader.extract_labels(feature_dict, label_dict, config.class_task, subject)

    print(len(feature_dict), len(label_dict))
    if len(feature_dict) != len(label_dict):
        print("WARNING: Not an equal number of sentences in features and labels!")

    for lr_val in config.lr:
        for e_val in config.epochs:
            parameter_dict = {"lr": lr_val, "lstm_dim": 128, "dense_dim": 64, "dropout": 0.5, "batch_size": 20, "epochs": e_val}

            if config.class_task == 'reldetect':
                reldetect_model.lstm_classfier(feature_dict, label_dict)
            elif config.class_task == 'sentiment-tri':
                fold_results = sentiment_model.lstm_classifier(feature_dict, label_dict, config.embeddings, parameter_dict)
                print(fold_results)
                save_results(fold_results, config.class_task)
            elif config.class_task == 'sentiment-bin':
                print(len(feature_dict), len(label_dict))
                for s, label in list(label_dict.items()):
                    # delete neutral sentences
                    if label == 2:
                            del label_dict[s]
                            del feature_dict[s]
                print(len(feature_dict), len(label_dict))
                fold_results = sentiment_model_bert.lstm_classifier(feature_dict, label_dict, config.embeddings, parameter_dict)
                print(fold_results)
                save_results(fold_results, config.class_task)





if __name__ == '__main__':
    main()