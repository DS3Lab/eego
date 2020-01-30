import config
import h5py
from feature_extraction import zuco_reader
from reldetect import reldetect_model

# calls zuco_reader --> define which feature to extract


def load_matlab_files(task, subject):
    """loads matlab files depending on which files are required for the chosen classification task"""

    if task.startswith("sentiment"):
        filename_sr = config.rootdir + "results" + subject + "_SR.mat"
        f_sr = h5py.File(filename_sr, 'r')
        sentence_data_sr = f_sr['sentenceData']

        return [(f_sr, sentence_data_sr)]

    elif task.startswith('reldetect'):
        if subject.startswith('Z'):  # subjects from ZuCo 1
            filename_nr = config.rootdir_zuco1 + "results" + subject + "_NR.mat"
            f_nr = h5py.File(filename_nr, 'r')
            sentence_data_nr = f_nr['sentenceData']
        elif subject.startswith('Y'):  # subjects from ZuCo 1
            filename_nr = config.rootdir_zuco2 + "results" + subject + "_NR.mat"
            f_nr = h5py.File(filename_nr, 'r')
            sentence_data_nr = f_nr['sentenceData']
        else:
            print("UNDEFINED SUBJECT NAME")

        return [(f_nr, sentence_data_nr)]


def main():
    feature_dict = {}
    label_dict = {}
    print("TASK: ", config.class_task)
    print("extracting", config.feature_set, "features....")
    for subject in config.subjects:
        print(subject)

        loaded_data = load_matlab_files(config.class_task, subject)

        zuco_reader.extract_features(loaded_data, config.feature_set, feature_dict)
        # todo: fix label extraction
        zuco_reader.extract_labels(feature_dict, label_dict, config.class_task, subject)

    print(len(feature_dict), len(label_dict))

    #reldetect_model.lstm_classfier(feature_dict, label_dict)



    # convert features to sequences for ML model
    #def convert_to_sequences():
        # reshape for LSTM

    # train model --> define which model to train

    # report results


if __name__ == '__main__':
    main()