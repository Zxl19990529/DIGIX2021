import pandas as pd
import glob, gc,os
import numpy as np
import seaborn as sns
from model import MLP,MLPDataset
import torch,datetime
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import reduce_mem
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path',type = str,default='/media/zxl/数据/DIGIX比赛/Dataset')
parser.add_argument('--bts_train',type = int,default=100,help = 'The batchsize of train')
parser.add_argument('--bts_val',type = int,default = 100,help = 'The batchsize of val')
parser.add_argument('--num_workers',type = int,default=10,help='The number of the core for data reading')
parser.add_argument('--log_root',type = str,default='./log',help='The root folder for saving training logs')
parser.add_argument('--chk_root',type = str,default='./checkpoint',help='The root folder for saving checkpoints')
args = parser.parse_args()

INPUT_PATH = args.input_path

history_behavior = pd.read_hdf('digix-data.hdf', 'history_behavior')
test_data = pd.read_hdf('digix-data.hdf', 'test_data')
user_features = pd.read_hdf('digix-data.hdf', 'user_features')
video_features = pd.read_hdf('digix-data.hdf', 'video_features')


# history_behavior

history_behavior = history_behavior[history_behavior['user_id'].isin(test_data['user_id'].unique())]#这一步需要至少14G内存
val_behavior = history_behavior[history_behavior['pt_d'] == 20210502]
train_behavior = history_behavior[history_behavior['pt_d'] != 20210502]

train_behavior = pd.concat([
    train_behavior[train_behavior['watch_label'] == 0].sample(1000000),
    train_behavior[train_behavior['watch_label'] != 0],
])
val_behavior = pd.concat([
    val_behavior[val_behavior['watch_label'] == 0].sample(1000000),
    val_behavior[val_behavior['watch_label'] != 0],
])
val_behavior['watch_label'].value_counts()

model = MLP()
model.predict({'user_id':np.array([10, 10]), 
               'video_id':np.array([10, 10]),
              'user_city': np.array([10, 10]), 
               'user_province': np.array([10, 10]), 
               'user_device':np.array([10, 10])})
model = model.cuda()
train_loader = torch.utils.data.DataLoader(
    dataset = MLPDataset(train_behavior,user_features),
    batch_size=args.bts_train, shuffle=True, num_workers=args.num_workers,
)

# batch_size 1000

val_loader = torch.utils.data.DataLoader(
    dataset = MLPDataset(val_behavior,user_features, train=True),
    batch_size=args.bts_val, shuffle=False, num_workers=args.num_workers,)

wathch_loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,2,2,2,2,2,2,2,2,2]).cuda())
shaere_loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#训练日志以时间命名
log_root = args.log_root
date = datetime.datetime.now().strftime('%F_%T').replace(':','_')
log_dir = os.path.join(log_root,date)
writer = SummaryWriter(log_dir=log_dir)
#checkpoints 保存路径
chk_folder = os.path.join(args.root,date)

step = 0
for epoch in range(10):
    for idx, data in tqdm(enumerate(train_loader),total=len(train_loader), desc=f'Epoch {epoch}'):
        feed_dict_cuda = {
            'user_id': data['user_id'].long().cuda(),
            'video_id': data['video_id'].long().cuda(),
            'user_province': data['user_province'].long().cuda(),
            'user_city': data['user_city'].long().cuda(),
            'user_device': data['user_device'].long().cuda(),
        }
        
        watch_label = data['watch_label'].long().cuda()
        share_label = data['share_label'].float().cuda().view(-1, 1)

        optimizer.zero_grad()
        wathch_pred, share_pred = model(feed_dict_cuda)
        loss = wathch_loss_fn(wathch_pred, watch_label) # + shaere_loss_fn(share_pred, share_label)

        loss.backward()
        optimizer.step()

        total_acc = (wathch_pred.argmax(1) == watch_label).float().mean().item()
        true_acc = ((wathch_pred.argmax(1) == watch_label) & (watch_label != 0)).float().sum()
        true_acc /= (watch_label != 0).float().sum()
        
        writer.add_scalar('Train/Total-ACC', total_acc, step)
        writer.add_scalar('Train/True-ACC', true_acc, step)
        writer.add_scalar('Train/Loss', loss.item(), step)
        writer.flush()
        step += 1
                
        if idx!=0 and idx % 1000 == 0:
            val_acc = []
            val_pred, val_label = [], []
            with torch.no_grad():
                for data in tqdm(val_loader, desc=f'Val'):
                    val_label += list(data['watch_label'].data.numpy())

                    feed_dict_cuda = {
                        'user_id': data['user_id'].long().cuda(),
                        'video_id': data['video_id'].long().cuda(),
                        'user_province': data['user_province'].long().cuda(),
                        'user_city': data['user_city'].long().cuda(),
                        'user_device': data['user_device'].long().cuda(),
                    }

                    watch_label = data['watch_label'].long().cuda()
                    share_label = data['share_label'].float().cuda().view(-1, 1)

                    wathch_pred, share_pred = model(feed_dict_cuda)
                    
                    val_pred += list(wathch_pred.argmax(1).data.cpu().numpy())
                    true_acc = ((wathch_pred.argmax(1) == watch_label) & (watch_label != 0)).float().sum()
                    true_acc /= (watch_label != 0).float().sum()
                    val_acc.append(true_acc)
            
            val_label = np.array(val_label)
            val_pred = np.array(val_pred)
            from sklearn.metrics import roc_auc_score
            score = 0 
            for aucflag, lbl in zip(np.linspace(0.1, 0.9, 9), range(1, 10)):
                # [0, 1, 2, 3] -> [0, 0, 1, 0]
                # [1, 1, 2, 3] -> [0, 0, 1, 0]
                pred = (val_pred == lbl).astype(int)
                label = (val_label == lbl).astype(int)
                try:
                    score += aucflag * roc_auc_score(label, pred)
                except:
                    pass
            
            val_acc = pd.DataFrame([x.item() for x in val_acc]).fillna(0).mean()[0]
            writer.add_scalar('Val/True-ACC-step', val_acc, step)
            writer.add_scalar('Val/AUC-step', score, step)
            writer.flush()
        writer.add_scalar('Val/True-ACC-epoch', val_acc, epoch)
        writer.add_scalar('Val/AUC-epoch', score, epoch)
        writer.flush()
        # save the checkpoint
        chk_path = os.path.join(chk_folder,'epoch_%d.pth'%epoch)
        torch.save(model.state_dict(),chk_path)

test_loader = torch.utils.data.DataLoader(
    dataset = MLPDataset(test_data,user_features, train=False),
    batch_size=20, shuffle=False, num_workers=args.num_wrokers,
)

test_watch = []
test_share = []
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


test_data['watch_label'] = test_watch
test_data['is_share'] = test_share

test_data.to_csv('submission.csv', index=None)