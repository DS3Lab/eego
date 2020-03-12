from datetime import date
import numpy as np
import config
import h5py


def load_matlab_files(task, subject):
    """loads matlab files depending on which files are required for the chosen classification task"""

    if task.startswith("sentiment"):
        filename_sr = config.rootdir_zuco1 + "results" + subject + "_SR.mat"
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


def save_results(fold_results_dict, task):
    """aggegates the results in fold_results_dict over all folds and
    saves hyper-parameters and results to a result file"""

    result_file = open('sentiment/results/'+str(date.today()) + "_results_" + task + ".txt", 'a')

    # print header
    print("lstm_dim", "lstm_layers", "dense_dim", "dropout", "batch_size", "epochs", "lr", "embedding_type",
          "random_seed", "train_acc", "val_acc", "test_acc", "test_std", "avg_precision", "std_precision",
          "avg_recall", "std_recall", "avg_fscore", "std_fscore", file=result_file)


    # training scores
    train_acc = np.mean([ep[-1] for ep in fold_results_dict['train-accuracy']])

    # validation scores
    val_acc = np.mean([ep[-1] for ep in fold_results_dict['val-accuracy']])

    # test scores
    avg_accuracy = np.mean(fold_results_dict['test-accuracy'])
    std_accuracy = np.std(fold_results_dict['test-accuracy'])
    avg_precision = np.mean(fold_results_dict['precision'])
    avg_recall = np.mean(fold_results_dict['recall'])
    avg_fscore = np.mean(fold_results_dict['fscore'])
    std_precision = np.std(fold_results_dict['precision'])
    std_recall = np.std(fold_results_dict['recall'])
    std_fscore = np.std(fold_results_dict['fscore'])

    print(" ".join(map(str, fold_results_dict['params'])),train_acc, val_acc, avg_accuracy, std_accuracy, avg_precision,
          std_precision, avg_recall, std_recall, avg_fscore, std_fscore, file=result_file)



