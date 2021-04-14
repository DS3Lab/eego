# dataset directories
#rootdir_zuco1 = "/Volumes/methlab/NLP/Ce_ETH/OSF-ZuCo1.0-200107/mat7.3/"
#rootdir_zuco2 = "/Volumes/methlab/NLP/Ce_ETH/2019/FirstLevel_V2/"

#base_dir = "/mnt/ds3lab-scratch/noraho/coling2020/"
base_dir = "/mnt/ds3lab-scratch/noraho/datasets/zuco/" # WITH UNFOLD!
rootdir_zuco1 = base_dir+"zuco1_preprocessed_sep2020/"
rootdir_zuco2 = base_dir+"zuco2_preprocessed_sep2020/"

# subjects (subejcts starting with "Z" are from ZuCo 1, subjects starting with "Y" are from ZuCo 2)
#subjects = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YMS', 'YRH', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']  # exclude YMH
subjects = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL',
            'YTL', "ZDN", "ZPH", "ZJN", "ZAB", "ZJM", "ZKB", "ZKH", "ZMG", "ZGW", "ZKW", "ZDM"]  # 'YMS', 'YRH', #'ZJS

# for running the experiments with previously extracted feature only one subject (from each dataset) is necessary
run_eeg_extraction = True
feature_dir = "../eeg_features/"

# ML task {sentiment-bin, sentiment-tri, reldetect}
class_task = 'reldetect'
# ML model {lstm, cnn}
model = 'cnn'
# word embeddings {none, glove (300d), bert}
embeddings = 'none'

# features sets {'text_only' , 'eeg_raw', 'eeg_theta', 'eeg_alpha', 'eeg_beta', 'eeg_gamma', 'combi_eeg_raw', 'eye_tracking', 'combi_eye_tracking'}
# sentence level features: {'combi_concat', 'sent_eeg_theta'}
# combined models: {'eeg_eye_tracking', 'eeg4'} 'binary' ?
feature_set = ['eeg_theta']

# hyper-parameters to test - general
lstm_dim = [64]
lstm_layers = [1]
dense_dim = [256]
dropout = [0.1]
batch_size = [60]
epochs = [200]
lr = [0.01]

# hyper-parameters for the convolutional EEG-decoding component, only apply if model = 'cnn' is selected
inception_filters = [14]
inception_kernel_sizes = [[1,4,7]]
inception_pool_size = [3]
inception_dense_dim = [(128,16)]

# best params raw eeg:
eeg_lstm_dim = [64]
eeg_dense_dim = [64]
eeg_dropout = [0.1]

# other parameters
folds = 5
random_seed_values = [13, 78, 22, 66, 42]  
validation_split = 0.1
patience = 80
min_delta = 0.0000001
data_percentage = 0#0.75
drop_classes = []#, 1, 4, 6, 3, 9, 2]

# only for Relation Detection:
rel_thresholds = [0.3]#, 0.5, 0.7]
