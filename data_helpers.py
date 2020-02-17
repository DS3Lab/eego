from datetime import date
import numpy as np


def save_results(fold_results_dict, task):

    result_file = open('sentiment/results/3classes/'+str(date.today()) + "_results_" + task + ".txt", 'a')

    avg_precision = np.mean(fold_results_dict['precision'])
    avg_recall = np.mean(fold_results_dict['recall'])
    avg_fscore = np.mean(fold_results_dict['fscore'])
    std_precision = np.std(fold_results_dict['precision'])
    std_recall = np.std(fold_results_dict['recall'])
    std_fscore = np.std(fold_results_dict['fscore'])

    print(" ".join(map(str, fold_results_dict['params'])),avg_precision, std_precision, avg_recall, std_recall, avg_fscore, std_fscore, file=result_file)



