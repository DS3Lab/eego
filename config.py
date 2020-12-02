# dataset directories
#rootdir_zuco1 = "/Volumes/methlab/NLP/Ce_ETH/OSF-ZuCo1.0-200107/mat7.3/"
#rootdir_zuco2 = "/Volumes/methlab/NLP/Ce_ETH/2019/FirstLevel_V2/"
base_dir = "/mnt/ds3lab-scratch/noraho/coling2020/"
#base_dir = '/Users/benjaminglaus/Desktop/BA_code/'
rootdir_zuco1 = base_dir+"zuco1/"
rootdir_zuco2 = base_dir+"zuco2/"

# subjects (subejcts starting with "Z" are from ZuCo 1, subjects starting with "Y" are from ZuCo 2)
#subjects = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YMS', 'YRH', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']  # exclude YMH
subjects = ["ZKW", "ZJS", "ZDN", "ZJN", "ZPH", "ZAB", "ZJM", "ZKB", "ZKH", "ZMG", "ZGW", "ZKW", "ZDM"]
#subjects = ["ZKW", "ZJS", "ZDN", "ZJN", "ZPH", "ZAB", "ZJM", "ZKB", "ZKH", "ZMG", "ZGW", "ZKW", "ZDM", 'YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YMS', 'YRH', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

# for running the experiments with previously extracted feature only one subject (from each dataset) is necessary


# ML task {sentiment-bin, sentiment-tri, ner, reldetect}
class_task = 'sentiment-bin'
# ML model {lstm, cnn}
model = 'cnn'

# features sets {'text_only' , 'eeg_raw', 'eeg_theta', 'eeg_alpha', 'eeg_beta', 'eeg_gamma', 'combi_eeg_raw', 'eye_tracking', 'combi_eye_tracking'}
# sentence level features: {'combi_concat', 'sent_eeg_theta'}
# combined models: {'eeg_eye_tracking', 'eeg4'}

feature_set = ['eeg_alpha']

# word embeddings {none, glove (300d), bert}
embeddings = 'bert'

# hyper-parameters to test - general
lstm_dim = [64]
lstm_layers = [1]
dense_dim = [128]
dropout = [0.3, 0.5]
batch_size = [40, 60]
epochs = [200]
lr = [0.01, 0.0001, 0.00001]

# params for CNN model
# eeg_cnn_filters = [[64,32,32,32]]
# eeg_cnn_kernel_size = [[4,4,2,2]]
# cnn_pool_size = [[2,4]]

# list of list of (layer_type, params)
# layer types:
# Conv, [filters, kernel_size, activation]
# Pooling, [pool_size]
# GlobalPooling, []
# Flatten, []
# Dense, [units, activation]
# Droupout, [rate]
cnn_network =  [#[('Conv', [64,8,'relu']), ('Flatten',[]), ('Dense', [512, 'relu'])],
                #[('Conv', [64,8,'relu']), ('Pooling', [4]), ('Flatten',[]), ('Dense', [256, 'relu'])],
                #[('Conv', [64,4,'relu']), ('Conv', [32,4,'relu']), ('Pooling', [2]), ('Flatten',[]), ('Dense', [128,'relu'])],
                #[('Conv', [64,4,'elu']), ('Conv', [32,4,'elu']), ('Pooling', [2]), ('Flatten',[]), ('Dense', [128,'relu'])],
                #[('Conv', [64,4,'relu']), ('Conv', [32,4,'relu']), ('Conv', [20,4,'relu']), ('Conv', [12,4,'relu']), ('Pooling', [4]), ('Flatten',[]), ('Dense', [64,'relu'])],
                [('Conv', [64,4,'elu']), ('Conv', [32,4,'elu']), ('Conv', [20,4,'relu']), ('Conv', [12,4,'relu']), ('Pooling', [4]), ('Flatten',[]), ('Dense', [64,'relu'])],
                [('Conv', [64,4,'elu']), ('Conv', [32,4,'elu']), ('Conv', [20,4,'relu']), ('Conv', [12,4,'relu']), ('GlobalPooling',[]), ('Dense', [64,'relu'])],
                #[('Conv', [64,4,'relu']), ('Conv', [32,4,'relu']), ('Pooling', [2]), ('Conv', [32,2,'relu']), ('Conv', [32,2,'relu']), ('Pooling', [4]), ('Flatten',[]), ('Dense', [64,'relu'])],
                [('Conv', [64,4,'elu']), ('Conv', [32,4,'elu']), ('Pooling', [2]), ('Conv', [32,2,'elu']), ('Conv', [32,2,'elu']), ('Pooling', [4]), ('Flatten',[]), ('Dense', [64,'relu'])],
                [('Conv', [64,4,'elu']), ('Conv', [32,4,'elu']), ('Pooling', [2]), ('Conv', [32,2,'elu']), ('Conv', [32,2,'elu']), ('GlobalPooling',[]), ('Dense', [64,'relu'])]]
                # always dropout and dense at end: control dropout

inception_filters = [16]#,16]
inception_kernel_sizes = [[1,4,7]]
inception_pool_size = [5]
#inception_activation_function=['elu', 'relu']

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
data_percentage = 0
drop_classes = [10, 8]#, 1, 4, 6, 3, 9, 2]

# only for Relation Detection:
rel_thresholds = [0.3, 0.5, 0.7]

