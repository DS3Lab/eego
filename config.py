# dataset {zuco1, zuco2, zuco1+2}
dataset = 'zuco1'

# dataset directories
#rootdir_zuco1 = "/Volumes/methlab/NLP/Ce_ETH/OSF-ZuCo1.0-200107/mat7.3/"
#rootdir_zuco2 = "/Volumes/methlab/NLP/Ce_ETH/2019/FirstLevel_V2/"
base_dir = "/mnt/ds3lab-scratch/noraho/coling2020/"
rootdir_zuco1 = base_dir+"zuco1/"
rootdir_zuco2 = base_dir+"zuco2/"

# subjects
#subjects = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YMS', 'YRH', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']  # exclude YMH
subjects = ["ZKW"]#,"ZJS", "ZDN", "ZJN"]#, "ZPH", "ZAB", "ZJM", "ZKB", "ZKH", "ZMG", "ZGW", "ZKW", "ZDM"]


# ML task {sentiment-bin, sentiment-tri, ner, reldetect}
class_task = 'sentiment-bin' #'sentiment-bin'

feature_set = 'text_only' #'gaze_features'

# word embeddings {none, glove (300d), bert}
embeddings = 'glove'

# hyper-parameters
lstm_dim = [128]
dense_dim = [64]
dropout = [0.5]
batch_size = [20]
epochs = [3]
lr = [0.001]

# other parameters
folds = 3
