import torch
import torch.nn as nn
import os,argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from model import MLP,MLPDataset

parser = argparse.ArgumentParser()
parser.add_argument('--input_path',type = str,default='/media/zxl/数据/DIGIX比赛/Dataset')
parser.add_argument('--chk',type=str,default='checkpoint/2021-08-21_17_52_06/epoch_9.pth',help='checkpoint path')
parser.add_argument('--num_workers',type = int,default=10,help='The number of the core for data reading')
parser.add_argument('--output',type=str,default='submission.csv')
args = parser.parse_args()

INPUT_PATH = args.input_path
user_features = pd.read_hdf('digix-data.hdf', 'user_features')
test_data = pd.read_hdf('digix-data.hdf', 'test_data')
test_loader = torch.utils.data.DataLoader(
    dataset = MLPDataset(test_data,user_features, train=False),
    batch_size=20, shuffle=False, num_workers=args.num_workers,
)
test_watch = []
test_share = []

model = MLP()
model.predict({'user_id':np.array([10, 10]), 
               'video_id':np.array([10, 10]),
              'user_city': np.array([10, 10]), 
               'user_province': np.array([10, 10]), 
               'user_device':np.array([10, 10])})
model = model.cuda()
checkpoint = torch.load(args.chk)
model.load_state_dict(checkpoint)
pbar = tqdm(test_loader)
with torch.no_grad():
    for data in test_loader:
        feed_dict_cuda = {
            'user_id': data['user_id'].long().cuda(),
            'video_id': data['video_id'].long().cuda(),
            'user_province': data['user_province'].long().cuda(),
            'user_city': data['user_city'].long().cuda(),
            'user_device': data['user_device'].long().cuda(),
        }
        wathch_pred, share_pred = model(feed_dict_cuda)
        
        test_watch += list(wathch_pred.argmax(1).cpu().data.numpy())
        test_share += list((share_pred.sigmoid() > 0.5).int().cpu().data.numpy().flatten())
        pbar.update(1)


test_data['watch_label'] = test_watch
test_data['is_share'] = test_share

test_data.to_csv('submission.csv', index=None)
