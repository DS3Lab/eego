import config
from feature_extraction import zuco_reader
from sentiment import sentiment_gaze_model, sentiment_text_gaze_model
from data_helpers import save_results, load_matlab_files
from ner import ner_text_gaze_model
from reldetect import reldetect_text_gaze_model, reldetect_gaze_model
import json
import collections
import numpy as np
import os
import tensorflow as tf

from datetime import timedelta
import time

# Usage on spaceml:
# $ conda activate env-eego
# $ CUDA_VISIBLE_DEVICES=7 python tune_model.py


def main():
    start = time.time()
    feature_dict = {}
    label_dict = {}
    eeg_dict = {}
    gaze_dict = {}
    print("TASK: ", config.class_task)
    print("extracting", config.feature_set, "features....")
    for subject in config.subjects:
        loaded_data = load_matlab_files(config.class_task, subject)

        zuco_reader.extract_features(loaded_data, config.feature_set, feature_dict, eeg_dict, gaze_dict)
        zuco_reader.extract_labels(feature_dict, label_dict, config.class_task, subject)

        elapsed = (time.time() - start)
        print('{}: {}'.format(subject, timedelta(seconds=int(elapsed))))

    if config.run_eeg_extraction:
        print(len(gaze_dict))
        with open('gaze_feats_file_ner.json', 'w') as fp:
            json.dump(gaze_dict, fp)
    else:
        print("Reading gaze features from file!!")
        gaze_dict = json.load(open("feature_extraction/features/gaze_feats_file_" + config.class_task + ".json"))


    feature_dict = collections.OrderedDict(sorted(feature_dict.items()))
    label_dict = collections.OrderedDict(sorted(label_dict.items()))
    gaze_dict = collections.OrderedDict(sorted(gaze_dict.items()))

    print('len(feature_dict): {}\nlen(label_dict): {}\nlen(eeg_dict): {}'.format(len(feature_dict), len(label_dict), len(eeg_dict)))

    if len(feature_dict) != len(label_dict) or len(feature_dict) != len(eeg_dict) or len(label_dict) != len(eeg_dict):
        print("WARNING: Not an equal number of sentences in features and labels!")

    print('Starting Loop')
    start = time.time()
    count = 0

    for rand in config.random_seed_values:
        np.random.seed(rand)
        tf.random.set_seed(rand)
        os.environ['PYTHONHASHSEED'] = str(rand)
        for lstmDim in config.lstm_dim:
            for lstmLayers in config.lstm_layers:
                for denseDim in config.dense_dim:
                    for inception_filters in config.inception_filters:
                        for inception_kernel_sizes in config.inception_kernel_sizes:
                            for inception_pool_size in config.inception_pool_size:
                                for inception_dense_dim in config.inception_dense_dim:
                                    for drop in config.dropout:
                                        for bs in config.batch_size:
                                            for lr_val in config.lr:
                                                for e_val in config.epochs:
                                                    parameter_dict = {"lr": lr_val, "lstm_dim": lstmDim, "lstm_layers": lstmLayers,
                                                                    "dense_dim": denseDim, "dropout": drop, "batch_size": bs,
                                                                    "epochs": e_val, "random_seed": rand, "inception_filters": inception_filters, 
                                                                    "inception_kernel_sizes": inception_kernel_sizes, "inception_pool_size": inception_pool_size, 
                                                                    "inception_dense_dim": inception_dense_dim}

                                                    if config.class_task == 'reldetect':
                                                        for threshold in config.rel_thresholds:
                                                            if 'combi_eye_tracking' in config.feature_set:
                                                                fold_results = reldetect_text_gaze_model.classifier(feature_dict,
                                                                                                                        label_dict,
                                                                                                                        gaze_dict,
                                                                                                                        config.embeddings,
                                                                                                                        parameter_dict,
                                                                                                                        rand,
                                                                                                                        threshold)

                                                            elif 'eye_tracking' in config.feature_set:
                                                                fold_results = reldetect_gaze_model.classifier(label_dict,
                                                                                                                    gaze_dict,
                                                                                                                    config.embeddings,
                                                                                                                    parameter_dict,
                                                                                                                    rand, threshold)

                                                            save_results(fold_results, config.class_task)

                                                    elif config.class_task == 'ner':
                                                        if 'combi_eye_tracking' in config.feature_set:
                                                            fold_results = ner_text_gaze_model.classifier(feature_dict, label_dict,
                                                                                                            gaze_dict,
                                                                                                            config.embeddings,
                                                                                                            parameter_dict,
                                                                                                            rand)
                                                        save_results(fold_results, config.class_task)

                                                    elif config.class_task == 'sentiment-tri':
                                                        if 'eye_tracking' in config.feature_set:
                                                            fold_results = sentiment_gaze_model.classifier(label_dict, gaze_dict,
                                                                                                                config.embeddings,
                                                                                                                parameter_dict,
                                                                                                                rand)
                                                        elif 'combi_eye_tracking' in config.feature_set:
                                                            fold_results = sentiment_text_gaze_model.classifier(feature_dict,
                                                                                                                    label_dict,
                                                                                                                    gaze_dict,
                                                                                                                    config.embeddings,
                                                                                                                    parameter_dict,
                                                                                                                    rand)

                                                        save_results(fold_results, config.class_task)
                                                    elif config.class_task == 'sentiment-bin':
                                                        for s, label in list(label_dict.items()):
                                                            # drop neutral sentences for binary sentiment classification
                                                            if label == 2:
                                                                del label_dict[s]
                                                                del feature_dict[s]
                                                                # del gaze_dict[s]

                                                        if 'eye_tracking' in config.feature_set:
                                                            fold_results = sentiment_gaze_model.classifier(label_dict, gaze_dict,
                                                                                                                config.embeddings,
                                                                                                                parameter_dict,
                                                                                                                rand)
                                                        elif 'combi_eye_tracking' in config.feature_set:
                                                            fold_results = sentiment_text_gaze_model.classifier(feature_dict,
                                                                                                                    label_dict,
                                                                                                                    gaze_dict,
                                                                                                                    config.embeddings,
                                                                                                                    parameter_dict,
                                                                                                                    rand)
                                                        save_results(fold_results, config.class_task)

                                                    elapsed = (time.time() - start)
                                                    print('iteration {} done'.format(count))
                                                    print('Time since starting the loop: {}'.format(timedelta(seconds=int(elapsed))))
                                                    count += 1


if __name__ == '__main__':
    main()
