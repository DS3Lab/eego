# dataset {zuco1, zuco2, zuco1+2}
dataset = 'zuco1'

# dataset directories
rootdir_zuco1 = "/Volumes/methlab/NLP/Ce_ETH/OSF-ZuCo1.0-200107/mat7.3/"
rootdir_zuco2 = "/Volumes/methlab/NLP/Ce_ETH/2019/FirstLevel_V2/"

# subjects
#subjects = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YMS', 'YRH', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']  # exclude YMH
subjects = ['YAG']#,"ZJS", "ZDN" "ZJN", "ZPH", "ZAB", "ZJM", "ZKB", "ZKH", "ZMG", "ZGW", "ZKW", "ZDM"]


# ML task {sentiment-bin, sentiment-tri, ner, reldetect}
class_task = 'reldetect' #'sentiment-bin'

feature_set = 'text_only'#'gaze_features'

