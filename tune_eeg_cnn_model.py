import config
from feature_extraction import zuco_reader
from reldetect import reldetect_eeg_model, reldetect_text_eeg_model, reldetect_eeg_gaze_model, reldetect_text_eeg4_model
from ner import ner_text_model
from sentiment import sentiment_eeg_model, sentiment_eeg_gaze_model, sentiment_text_eeg_gaze_model, sentiment_text_eeg4_model, sentiment_text_eeg_model
from data_helpers import save_results, load_matlab_files
import numpy as np
import collections
import json

from datetime import timedelta
import time

# Usage on spaceml:
# $ conda activate env-eego
# $ CUDA_VISIBLE_DEVICES=7 python tune_model.py

def main():
    start = time.time() # remove later
    feature_dict = {}
    label_dict = {}
    eeg_dict = {}
    gaze_dict = {}
    print("TASK: ", config.class_task)
    print("Extracting", config.feature_set, "features....")
    
    for subject in config.subjects:
        #print(subject)
        loaded_data = load_matlab_files(config.class_task, subject)

        zuco_reader.extract_features(loaded_data, config.feature_set, feature_dict, eeg_dict, gaze_dict)
        zuco_reader.extract_labels(feature_dict, label_dict, config.class_task, subject)

        elapsed = (time.time() - start)
        print('{}: {}'.format(subject, timedelta(seconds=int(elapsed))))

    print('len(feature_dict): {}\nlen(label_dict): {}\nlen(eeg_dict): {}'.format(len(feature_dict), len(label_dict), len(eeg_dict)))

    #print("Reading EEG features from file!!")
    #eeg_dict = json.load(
        #open("../eeg_features/" + config.feature_set[0] + "_feats_file_" + config.class_task + ".json"))
    print("done, ", len(eeg_dict), " sentences with EEG features.")
 
    # save EEG features
    with open('../eeg_features/' + config.feature_set[0] + '_feats_file_' + config.class_task + '.json', 'w') as fp:
       json.dump(eeg_dict, fp)
    print("saved to ../eeg_features/{}_feats_file_{}.json".format(config.feature_set[0], config.class_task))

    feature_dict = collections.OrderedDict(sorted(feature_dict.items()))
    label_dict = collections.OrderedDict(sorted(label_dict.items()))
    eeg_dict = collections.OrderedDict(sorted(eeg_dict.items()))

    if len(feature_dict) != len(label_dict) or len(feature_dict) != len(eeg_dict) or len(label_dict) != len(eeg_dict):
        print("WARNING: Not an equal number of sentences in features and labels!")

    if config.model is not 'cnn':
        print("WARNING: Not running CNN model!")

    print('Starting loop')
    start = time.time()

    for rand in config.random_seed_values:
        np.random.seed(rand)
        for lstmDim in config.lstm_dim: # needed for text model
            for lstmLayers in config.lstm_layers: 
                for denseDim in config.dense_dim: # needed for text model
                    for cnn_filter in config.eeg_cnn_filters:
                        for cnn_model in config.eeg_cnn_network:
                            for drop in config.dropout:
                                for bs in config.batch_size:
                                    for lr_val in config.lr:
                                        for e_val in config.epochs:
                                            parameter_dict = {"lr": lr_val, "lstm_dim": lstmDim, "lstm_layers": lstmLayers,
                                                            "dense_dim": denseDim, "dropout": drop, "batch_size": bs,
                                                            "epochs": e_val, "random_seed": rand, "cnn_model": cnn_model,
                                                            "cnn_filter": cnn_filter}

                                            if config.class_task == "reldetect":
                                                #TODO
                                                continue

                                            elif config.class_task == "ner":
                                                #TODO
                                                continue

                                            elif config.class_task == "sentiment-tri":
                                                #TODO
                                                continue
                                            
                                            elif config.class_task == "sentiment-bin":
                                                for s, label in list(label_dict.items()):
                                                        # drop neutral sentences for binary sentiment classification
                                                        if label == 2:
                                                            del label_dict[s]
                                                            del feature_dict[s]
                                                            del eeg_dict[s]

                                                if 'combi_concat' in config.feature_set:
                                                    print("Starting EEG + text combi model")
                                                    #fold_results =
                                                    continue
                                                
                                                elif 'eeg_raw' in config.feature_set:
                                                        fold_results = sentiment_eeg_model.classifier(label_dict,
                                                                                                        eeg_dict,
                                                                                                        config.embeddings,
                                                                                                        parameter_dict,
                                                                                                        rand)
                                                        

                                                elif 'random' in config.feature_set and 'eeg_alpha' in config.feature_set:
                                                        #fold_results =
                                                        continue

                                                elif 'combi_eeg_raw' in config.feature_set or 'eeg_theta' in config.feature_set or 'eeg_alpha' in config.feature_set or 'eeg_beta' in config.feature_set or 'eeg_gamma' in config.feature_set:
                                                        fold_results = sentiment_text_eeg_model.classifier(feature_dict,
                                                                                                                label_dict,
                                                                                                                eeg_dict,
                                                                                                                config.embeddings,
                                                                                                                parameter_dict,
                                                                                                                rand)

                                                save_results(fold_results, config.class_task)
                                                elapsed = (time.time() - start)
                                                print('\nTIME since starting loop: {}\n'.format(timedelta(seconds=int(elapsed))))

if __name__ == '__main__':
    main()
