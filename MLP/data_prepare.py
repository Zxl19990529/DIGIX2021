import pandas as pd
import numpy as np
from utils import reduce_mem
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path',type = str,default='/media/zxl/数据/DIGIX比赛/Dataset')
args = parser.parse_args()

INPUT_PATH = args.input_path
#读取测试集和训练集文件
test_data = pd.read_csv(f'{INPUT_PATH}/testdata/test.csv', sep=',')
user_features = pd.read_csv(f'{INPUT_PATH}/traindata/user_features_data/user_features_data.csv', sep='\t')
video_features = pd.read_csv(f'{INPUT_PATH}/traindata/video_features_data/video_features_data.csv', sep='\t')
history_behavior = pd.concat([
    reduce_mem(pd.read_csv(x, sep='\t')) for x in glob.glob(f'{INPUT_PATH}/traindata/history_behavior_data/*/*')
])
# 日期和用户id
history_behavior = history_behavior.sort_values(by=['pt_d', 'user_id'])

#将数据压缩到16bit
test_data = reduce_mem(test_data)
user_features = reduce_mem(user_features)
video_features = reduce_mem(video_features)
history_behavior = reduce_mem(history_behavior)

#将数据写入二进制文件,以.hdf格式储存
test_data.to_hdf('digix-data.hdf', 'test_data')
user_features.to_hdf('digix-data.hdf', 'user_features')
video_features.to_hdf('digix-data.hdf', 'video_features')
history_behavior.to_hdf('digix-data.hdf', 'history_behavior')
