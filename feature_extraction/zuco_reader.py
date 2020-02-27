import csv

import gaze_extractor
import text_extractor

# wrapper script to read matlab files and extract gaze and/or EEG features


def extract_features(sent_data, feature_set, feature_dict):
    """"""

    if "gaze" in feature_set:
        gaze_extractor.word_level_et_features(sent_data)

    # extract only text for baseline models
    if feature_set == 'text_only':
        text_extractor.extract_sentences(sent_data, feature_dict)


def extract_labels(feature_dict, label_dict, task, subject):
    """"""
    if task.startswith("sentiment"):

        count = 0
        #label_names = {};
        label_names = {'0': 2, '1': 1, '-1': 0}
        i = 0

        if subject.startswith('Z'):  # subjects from ZuCo 1
            with open('/Users/norahollenstein/Desktop/PhD/projects/eego/feature_extraction/labels/sentiment_sents_labels-corrected.txt', 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=';')
                for row in csv_reader:
                    #print(row)
                    sent = row[1]
                    label = row[-1]

                    if label not in label_names:
                        label_names[label] = i
                        i += 1

                    if sent in feature_dict:
                        label_dict[sent] = label_names[label]
                    else:
                        print("Sentence not found in feature dict!")
                        print(sent)
                        count += 1
                print('ZuCo 1 sentences not found:', count)

        else:
            print("Sentiment analysis only possible for ZuCo 1!!!")

        print(label_names)

    if task == 'reldetect':

        count = 0
        label_names = {}; i = 0

        # todo: update this to take labels from brat!!! original labels are not complete

        if subject.startswith('Z'):  # subjects from ZuCo 1
            with open('/Users/norahollenstein/Desktop/PhD/projects/eego/feature_extraction/labels/zuco1_relations_nr_labels_cleaned.csv', 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                next(csv_reader, None)
                for row in csv_reader:
                    sent = row[2]
                    # todo: what to do with sentences with multiple labels?!
                    # one approach: evaluate on single classes, i.e. change test set?
                    # another approach: randomly assign one relation and train multiple runs
                    label = row[4].split(";")[0]

                    if label not in label_names:
                        label_names[label] = i
                        i += 1

                    if sent in feature_dict:
                        #print(zuco2_relations_normal_reading_labels.csv[sent])
                        label_dict[sent] = label_names[label]
                    else:
                        print("Sentence not found in feature dict!")
                        print(sent)
                        count += 1
            print('ZuCo 1 sentences not found:', count)

        elif subject.startswith('Y'):  # subjects from ZuCo 2
            count = 0
            with open('/Users/norahollenstein/Desktop/PhD/projects/eego/feature_extraction/labels/zuco2_relations_nr_labels_cleaned.csv', 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                next(csv_reader, None)
                for row in csv_reader:
                    sent = row[2]
                    # todo: what to do with sentences with multiple labels?!
                    # one approach: evaluate on single classes, i.e. change test set?
                    # another approach: randomly assign one relation and train multiple runs
                    label = row[4].split(";")[0]

                    if label not in label_names:
                        label_names[label] = i
                        i += 1

                    if sent in feature_dict:
                        #print(zuco2_relations_normal_reading_labels.csv[sent])
                        label_dict[sent] = label_names[label]
                    else:
                        print("Sentence not found in feature dict!")
                        print(sent)
                        count += 1
            print('ZuCo 2 sentences not found:', count)



