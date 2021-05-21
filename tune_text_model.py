import config
from feature_extraction import zuco_reader
from reldetect import reldetect_text_model, reldetect_text_model_binary
from ner import ner_text_model
from sentiment import sentiment_text_model
from data_helpers import save_results, load_matlab_files
import collections
import numpy as np
import os
import tensorflow as tf
import random
from datetime import timedelta
import time

# Usage on spaceml:
# $ conda activate env-eego
# $ CUDA_VISIBLE_DEVICES=7 python tune_text_model.py


def main():
    start = time.time()
    feature_dict = {}
    label_dict = {}
    eeg_dict = {}
    gaze_dict = {}
    print("TASK: ", config.class_task)
    print("extracting", config.feature_set[0], "features....")
    for subject in config.subjects:
        loaded_data = load_matlab_files(config.class_task, subject)

        zuco_reader.extract_features(loaded_data, config.feature_set, feature_dict, eeg_dict, gaze_dict)
        zuco_reader.extract_labels(feature_dict, label_dict, config.class_task, subject)

        elapsed = (time.time() - start)
        print('{}: {}'.format(subject, timedelta(seconds=int(elapsed))))

    feature_dict = collections.OrderedDict(sorted(feature_dict.items()))
    label_dict = collections.OrderedDict(sorted(label_dict.items()))
    print(len(feature_dict.keys()), len(label_dict))

    # test with less data
    #drop_first_sents(label_dict, feature_dict, eeg_dict)
    
    print('len(feature_dict): {}\nlen(label_dict): {}'.format(len(feature_dict), len(label_dict)))

    if len(feature_dict) != len(label_dict):
        print("WARNING: Not an equal number of sentences in features and labels!")

    print('starting loop')
    start = time.time()
    count = 0

    for rand in config.random_seed_values:
        np.random.seed(rand)
        tf.random.set_seed(rand)
        os.environ['PYTHONHASHSEED'] = str(rand)
        random.seed(rand)
        for lstmDim in config.lstm_dim:
            for lstmLayers in config.lstm_layers:
                for denseDim in config.dense_dim:
                    for drop in config.dropout:
                        for bs in config.batch_size:
                            for lr_val in config.lr:
                                for e_val in config.epochs:
                                    for inception_filters in config.inception_filters:
                                        for inception_kernel_sizes in config.inception_kernel_sizes:
                                            for inception_pool_size in config.inception_pool_size:
                                                for inception_dense_dim in config.inception_dense_dim:
                                                    parameter_dict = {"lr": lr_val, "lstm_dim": lstmDim, "lstm_layers": lstmLayers,
                                                                    "dense_dim": denseDim, "dropout": drop, "batch_size": bs,
                                                                    "epochs": e_val, "random_seed": rand, "inception_filters": inception_filters,
                                                                    "inception_dense_dim": inception_dense_dim, "inception_kernel_sizes": inception_kernel_sizes,
                                                                    "inception_pool_size": inception_pool_size}

                                                    if config.class_task == 'reldetect':
                                                        for threshold in config.rel_thresholds:
                                                            if 'binary' in config.feature_set:
                                                                fold_results = reldetect_text_model_binary.classifier(feature_dict,
                                                                                                                    label_dict,
                                                                                                                    config.embeddings,
                                                                                                                    parameter_dict,
                                                                                                                    rand)
                                                                save_results(fold_results, config.class_task)
                                                            else:
                                                                fold_results = reldetect_text_model.classifier(feature_dict, label_dict, config.embeddings, parameter_dict, rand, threshold)
                                                                save_results(fold_results, config.class_task)


                                                    elif config.class_task == 'ner':
                                                        fold_results = ner_text_model.classifier(feature_dict, label_dict, config.embeddings, parameter_dict, rand)
                                                        save_results(fold_results, config.class_task)

                                                    elif config.class_task == 'sentiment-tri':
                                                        fold_results = sentiment_text_model.classifier(feature_dict, label_dict, config.embeddings, parameter_dict, rand)
                                                        save_results(fold_results, config.class_task)

                                                    elif config.class_task == 'sentiment-bin':
                                                        for s, label in list(label_dict.items()):
                                                            # drop neutral sentences for binary sentiment classification
                                                            if label == 2:
                                                                    del label_dict[s]
                                                                    del feature_dict[s]
                                                        print(len(feature_dict), len(label_dict))
                                                        fold_results = sentiment_text_model.classifier(feature_dict, label_dict, config.embeddings, parameter_dict, rand)
                                                    
                                                        save_results(fold_results, config.class_task)
                                                        
                                                    elapsed = (time.time() - start)
                                                    print('iteration {} done'.format(count))
                                                    print('Time since starting the loop: {}'.format(timedelta(seconds=int(elapsed))))
                                                    count += 1


if __name__ == '__main__':
    main()
