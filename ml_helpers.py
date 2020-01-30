import matplotlib. pyplot as plt
import numpy as np



def plot_label_distribution(y):

    plt.hist(y, bins=len(set(y)), alpha=0.5)
    plt.xticks(rotation=90, fontsize=7)
    plt.show()
