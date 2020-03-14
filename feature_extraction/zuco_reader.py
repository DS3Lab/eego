import csv
import config
from . import gaze_extractor
from . import text_extractor

# wrapper script to read matlab files and extract gaze and/or EEG features


def extract_features(sent_data, feature_set, feature_dict):
    """"""

    # extract only text for baseline models
    if feature_set == 'text_only':
        text_extractor.extract_sentences(sent_data, feature_dict)

    if "gaze" in feature_set:
        gaze_extractor.word_level_et_features(sent_data, feature_dict)


def extract_labels(feature_dict, label_dict, task, subject):
    """"""
    if task.startswith("sentiment"):

        count = 0
        label_names = {'0': 2, '1': 1, '-1': 0}
        i = 0

        if subject.startswith('Z'):  # subjects from ZuCo 1
            with open(config.base_dir+'eego/feature_extraction/labels/sentiment_sents_labels-corrected.txt', 'r') as csv_file:
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


    elif task == 'ner':

        count = 0
        #label_names = {'0': 2, '1': 1, '-1': 0}
        i = 0

        if subject.startswith('Z'):  # subjects from ZuCo 1
            # use NR + sentiment task from ZuCo 1
            ner_ground_truth = open(config.base_dir+'eego/feature_extraction/labels/zuco1_nr_ner.bio', 'r').readlines() + open(config.base_dir+'eego/feature_extraction/labels/zuco1_nr_sentiment_ner.bio', 'r').readlines()
            for line in ner_ground_truth:
                sent_tokens = []
                sent_labels = []
                
                # start of new sentence
                if line == '\n':
                    print(sent_tokens)
                    print(sent_labels)
                    if sent_tokens in feature_dict:

                        label_dict[sent_tokens] = sent_labels
                    else:
                        print("Sentence not found in feature dict!")
                        print(sent_tokens)
                        count += 1

                    sent_tokens = []
                    sent_labels = []
                else:
                    line = line.split('\t')
                    sent_tokens.append(line[0])
                    sent_labels.append(line[1])

                print('ZuCo 1 sentences not found:', count)


    elif task == 'reldetect':

        count = 0
        label_names = {}; i = 0

        # todo: update this to take labels from brat!!! original labels are not complete

        if subject.startswith('Z'):  # subjects from ZuCo 1
            with open(config.base_dir+'eego/feature_extraction/labels/zuco1_relations_nr_labels_cleaned.csv', 'r') as csv_file:
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
            with open(config.basedir+'eego/feature_extraction/labels/zuco2_relations_nr_labels_cleaned.csv', 'r') as csv_file:
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



