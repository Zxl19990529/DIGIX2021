import os,shutil,glob,gc
import numpy as np
from model import DeepFM,MLPDataset
from utils import to_text_vector
import pandas as pd
import torch,datetime
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from gensim.models import Word2Vec
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path',type = str,default='/media/zxl/数据/DIGIX比赛/Dataset')
parser.add_argument('--chk',type=str,default='checkpoint/2021-08-31_09_47_26/epoch8_iter1000.pth',help='checkpoint path')
parser.add_argument('--num_workers',type = int,default=10,help='The number of the core for data reading')
parser.add_argument('--w2v',type = str,default= 'word2vec.w2v',help='The weight of Word2Vec model')
parser.add_argument('--output',type=str,default='submission.csv')
args = parser.parse_args()

INPUT_PATH = args.input_path

# word2vec model
text_model = Word2Vec.load(args.w2v)

history_behavior = pd.read_hdf('digix-data.hdf', 'history_behavior')
test_data = pd.read_hdf('digix-data.hdf', 'test_data')
user_features = pd.read_hdf('digix-data.hdf', 'user_features')
video_features = pd.read_hdf('digix-data.hdf', 'video_features')

# history_behavior

history_behavior = history_behavior[history_behavior['user_id'].isin(test_data['user_id'].unique())]#这一步需要至少14G内存
val_behavior = history_behavior[history_behavior['pt_d'] == 20210502]
train_behavior = history_behavior[history_behavior['pt_d'] != 20210502]

test_loader = torch.utils.data.DataLoader(
    dataset = MLPDataset(test_data,user_features,video_features,text_model, train=False),
    batch_size=20, shuffle=False, num_workers=args.num_workers,
)


test_watch = []
test_share = []

model = DeepFM(cate_fea_nuniqs=[5910798+1, 50355+1,34, 340, 1927],
              nume_fea_size=100)
model = model.cuda()
print(model)


with torch.no_grad():
    for idx,data in tqdm(enumerate(test_loader),total=len(test_loader),desc='Test'):

        sparse_feat = torch.stack([data[x].long()
            for x in ['user_id', 'video_id', 'user_province', 
                    'user_city', 'user_device']]).T.cuda()
        dense_feat = data['video_feature_vec'].cuda()

        watch_pred, share_pred = model(sparse_feat,dense_feat)
        
        test_watch += list(watch_pred.argmax(1).cpu().data.numpy())
        test_share += list((share_pred.sigmoid() > 0.5).int().cpu().data.numpy().flatten())


test_data['watch_label'] = test_watch
test_data['is_share'] = test_share

test_data.to_csv('submission.csv', index=None)
