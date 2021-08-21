import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(0)

class MLP(nn.Module):
    # 行为数据
    def __init__(self, n_users=5910798+1, n_items=50355+1, 
                 n_provinces=33+1, n_citys=339+1, n_devices=1826+1,
                 layers=[70, 32], dropout=False):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(n_users, 50)
        self.video_embedding = torch.nn.Embedding(n_items, 20)
        
        # self.user_province_embedding = torch.nn.Embedding(n_provinces, 5)
        # self.user_city_embedding = torch.nn.Embedding(n_citys, 5)
        # self.user_device_embedding = torch.nn.Embedding(n_devices, 5)

        # list of weight matrices
        self.fc_layers = torch.nn.ModuleList()
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        self.output_layer1 = torch.nn.Linear(layers[-1], 10)
        self.output_layer2 = torch.nn.Linear(layers[-1], 1)
        
        print(self.fc_layers)
        
    def forward(self, feed_dict):
        users = feed_dict['user_id']
        items = feed_dict['video_id']        
        
        user_embedding = self.user_embedding(users)
        video_embedding = self.video_embedding(items)
        
        # user_province = self.user_province_embedding(feed_dict['user_province'])
        # user_city = self.user_city_embedding(feed_dict['user_city'])
        # user_device = self.user_device_embedding(feed_dict['user_device'])
        
        x = torch.cat([user_embedding, video_embedding], 1)
        # x = torch.cat([user_embedding, video_embedding, user_province, user_city, user_device], 1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            x = self.fc_layers[idx](x)
            x = F.relu(x)
            x = F.dropout(x)
        logit1 = self.output_layer1(x)
        logit2 = self.output_layer2(x)
        return logit1, logit2

    def predict(self, feed_dict):
        for key in feed_dict:
            if type(feed_dict[key]) != type(None):
                feed_dict[key] = torch.from_numpy(
                    feed_dict[key]).to(dtype=torch.long, device='cpu')
        output_scores = self.forward(feed_dict)
        return output_scores

# model = model.cuda()

class MLPDataset(Dataset):
    def __init__(self, history_behavior,user_features, train=True):
        self.history_behavior = history_behavior
        self.train = train
        self.user_features = user_features
    def __getitem__(self, index):
        user_id = self.history_behavior.iloc[index]['user_id']
        video_id = self.history_behavior.iloc[index]['video_id']
        
        # 从行为日志 去 索引用户的特征
        user_province = self.user_features.loc[user_id]['province']
        user_city = self.user_features.loc[user_id]['city']
        user_device = self.user_features.loc[user_id]['device_name']
        
        feed_dict = {
            'user_id': user_id,
            'video_id': video_id,
            'user_province': user_province,
            'user_city': user_city,
            'user_device': user_device,
        }
        
        if self.train:
            watch_label = self.history_behavior.iloc[index]['watch_label']
            share_label = self.history_behavior.iloc[index]['is_share']

            feed_dict['watch_label'] = watch_label
            feed_dict['share_label'] = share_label
            
            return feed_dict
            
#             return user_id, video_id, \
#                 torch.from_numpy(np.array(watch_label)), \
#                 torch.from_numpy(np.array([share_label]))
        else:
            return feed_dict
        
    def __len__(self):
        return len(self.history_behavior)
    