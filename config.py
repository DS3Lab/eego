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

# Features sets {'text_only', 'gaze_feats', 'eeg_feats'}
feature_set = 'text_only' #'gaze_features'

# word embeddings {none, glove (300d), bert}
embeddings = ['glove']
# specify directory with bert model, if using
modelBertDir = "/mnt/ds3lab-scratch/noraho/embeddings/bert"

# hyper-parameters
lstm_dim = [64, 128, 256, 512]
dense_dim = [32, 64, 128, 256]
dropout = [0.1, 0.3, 0.5]
batch_size = [20, 30, 40, 60]
epochs = [3, 5, 10, 20, 50]
lr = [0.001, 0.01, 0.0001]

# other parameters
folds = 5
