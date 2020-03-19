# dataset {zuco1, zuco2, zuco1+2}
# todo: do we need this one?
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
embeddings = ['none', 'glove']
# specify directory with bert model, if using
modelBertDir = "/mnt/ds3lab-scratch/noraho/embeddings/bert"

# hyper-parameters to test - general
lstm_dim = [64, 128, 256, 512]
lstm_layers = [1, 2, 3, 4]
dense_dim = [32, 64, 128, 256]
dropout = [0.1, 0.3, 0.5]
batch_size = [20, 30, 40, 60]
epochs = [5, 10, 20, 50, 100]
lr = [0.001, 0.01, 0.0001]

# other parameters
folds = 5
random_seed_values = [13, 78, 22, 66, 42]

# only for Relation Detection:
rel_thresholds = [0.3, 0.5, 0.7]

# NONE sentiment bin
"""
lstm_dim = [64, 128, 256]  # 512
lstm_layers = [1]  # 2, 3, 4
dense_dim = [32, 64, 128]  # 256
dropout = [0.1, 0.3, 0.5]
batch_size = [40, 60]  # 20, 30
epochs = [50, 100]  # 5, 10, 20
lr = [0.001, 0.0001]  # 0.01
"""

# GLOVE sentiment bin
"""
lstm_dim = [256, 512]  # 64, 128
lstm_layers = [1, 2]  #  3, 4
dense_dim = [64, 128, 256]  # 32
dropout = [0.1, 0.3, 0.5]
batch_size = [20, 30, 40, 60]
epochs = [50, 100]  # 5, 10, 20
lr = [0.001, 0.01, 0.0001]  
"""

# BERT sentiment bin
"""
lstm_dim = [128, 256, 512]  # 64
lstm_layers = [1, 2, 3, 4]  # 1
dense_dim = [64, 128]  # 32, 256
dropout = [0.1, 0.3]  # 0.5
batch_size = [20, 30, 40]  # 60
epochs = [10, 20, 50, 100]  # 5
lr = [0.001, 0.0001]  # 0.01
"""


# NONE sentiment tri
"""
lstm_dim = [256, 512]  # 64, 128
lstm_layers = [1, 2]  # 3, 4
dense_dim = [64, 128, 256]  # 32
dropout = [0.1, 0.3, 0.5]
batch_size = [20, 30, 40, 60]
epochs = [20, 50, 100]  # 5, 10
lr = [0.001, 0.0001]  # 0.01
"""

# GLOVE sentiment tri
"""
lstm_dim = [64, 128, 256, 512]
lstm_layers = [1, 2, 3, 4]
dense_dim = [32, 64, 128, 256]
dropout = [0.1, 0.3, 0.5]
batch_size = [20, 30, 40, 60]
epochs = [5, 10, 20, 50, 100]
lr = [0.001, 0.01, 0.0001]
"""

# BERT sentiment tri
"""
lstm_dim = [64, 128, 256, 512]  
lstm_layers = [1, 2, 3, 4]  
dense_dim = [32, 64, 128, 256]  
dropout = [0.1, 0.3, 0.5]
batch_size = [20, 30, 40, 60]  
epochs = [5, 10, 20, 50, 100]  
lr = [0.001, 0.01, 0.0001]  
"""