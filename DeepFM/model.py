import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils import to_text_vector
torch.manual_seed(0)

class DeepFM(nn.Module):
    def __init__(self, cate_fea_nuniqs, nume_fea_size=0, emb_size=8, 
                 hid_dims=[256, 128], num_classes=1, dropout=[0.2, 0.2]): 
        """
        cate_fea_nuniqs: 类别特征的唯一值个数列表，也就是每个类别特征的vocab_size所组成的列表
        nume_fea_size: 数值特征的个数，该模型会考虑到输入全为类别型，即没有数值特征的情况 
        """
        super().__init__()
        self.cate_fea_size = len(cate_fea_nuniqs)
        self.nume_fea_size = nume_fea_size
        
        """FM部分"""
        # 一阶
        if self.nume_fea_size != 0:
            self.fm_1st_order_dense = nn.Linear(self.nume_fea_size, 1)  # 数值特征的一阶表示
        self.fm_1st_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, 1) for voc_size in cate_fea_nuniqs])  # 类别特征的一阶表示
        
        # 二阶
        self.fm_2nd_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, emb_size) for voc_size in cate_fea_nuniqs])  # 类别特征的二阶表示
        
        """DNN部分"""
        self.all_dims = [self.cate_fea_size * emb_size] + hid_dims
        self.dense_linear = nn.Linear(self.nume_fea_size, self.cate_fea_size * emb_size)  # 数值特征的维度变换到FM输出维度一致
        self.relu = nn.ReLU()
        # for DNN 
        for i in range(1, len(self.all_dims)):
            setattr(self, 'linear_'+str(i), nn.Linear(self.all_dims[i-1], self.all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dims[i]))
            setattr(self, 'activation_' + str(i), nn.ReLU())
            setattr(self, 'dropout_'+str(i), nn.Dropout(dropout[i-1]))
        # for output 
        self.dnn_linear1 = nn.Linear(hid_dims[-1]+2, 10)
        self.dnn_linear2 = nn.Linear(hid_dims[-1], 1)
        
    def forward(self, X_sparse, X_dense=None):
        """
        X_sparse: 类别型特征输入  [bs, cate_fea_size]
        X_dense: 数值型特征输入（可能没有）  [bs, dense_fea_size]
        """
        
        """FM 一阶部分"""
        
        # print('self.fm_1st_order_sparse_emb', len(self.fm_1st_order_sparse_emb))
        
        fm_1st_sparse_res = [emb(X_sparse[:, i].unsqueeze(1)).view(-1, 1) 
                             for i, emb in enumerate(self.fm_1st_order_sparse_emb)]
        fm_1st_sparse_res = torch.cat(fm_1st_sparse_res, dim=1)  # [bs, cate_fea_size]
        fm_1st_sparse_res = torch.sum(fm_1st_sparse_res, 1,  keepdim=True)  # [bs, 1]
        
        if X_dense is not None:
            fm_1st_dense_res = self.fm_1st_order_dense(X_dense) 
            fm_1st_part = fm_1st_sparse_res + fm_1st_dense_res
        else:
            fm_1st_part = fm_1st_sparse_res   # [bs, 1]
        
        """FM 二阶部分"""
        fm_2nd_order_res = [emb(X_sparse[:, i].unsqueeze(1)) for i, emb in enumerate(self.fm_2nd_order_sparse_emb)]
        fm_2nd_concat_1d = torch.cat(fm_2nd_order_res, dim=1)  # [bs, n, emb_size]  n为类别型特征个数(cate_fea_size)
        
        # 先求和再平方
        sum_embed = torch.sum(fm_2nd_concat_1d, 1)  # [bs, emb_size]
        square_sum_embed = sum_embed * sum_embed    # [bs, emb_size]
        # 先平方再求和
        square_embed = fm_2nd_concat_1d * fm_2nd_concat_1d  # [bs, n, emb_size]
        sum_square_embed = torch.sum(square_embed, 1)  # [bs, emb_size]
        # 相减除以2 
        sub = square_sum_embed - sum_square_embed  
        sub = sub * 0.5   # [bs, emb_size]
        
        fm_2nd_part = torch.sum(sub, 1, keepdim=True)   # [bs, 1]
        
        """DNN部分"""
        dnn_out = torch.flatten(fm_2nd_concat_1d, 1)   # [bs, n * emb_size]
        
        if X_dense is not None:
            dense_out = self.relu(self.dense_linear(X_dense))   # [bs, n * emb_size]
            dnn_out = dnn_out + dense_out   # [bs, n * emb_size]
        
        for i in range(1, len(self.all_dims)):
            dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)
        
        out1 = self.dnn_linear1(torch.cat([dnn_out, fm_1st_part, fm_2nd_part],1))   # [bs, N]
        # out1 = fm_1st_part + fm_2nd_part + dnn_out   # [bs, 1]
         
        dnn_out2 = self.dnn_linear2(dnn_out)   # [bs, 1]
        out2 = fm_1st_part + fm_2nd_part + dnn_out2   # [bs, 1]
        
        return out1, out2
# model = DeepFM(cate_fea_nuniqs=[5910798+1, 50355+1,34, 340, 1927],
            #   nume_fea_size=2)

class MLPDataset(Dataset):
    def __init__(self, history_behavior,user_features,video_features,w2v_model, train=True):
        self.history_behavior = history_behavior
        self.train = train
        self.user_features = user_features
        self.video_features = video_features
        self.w2v = w2v_model
    def __getitem__(self, index):
        user_id = self.history_behavior.iloc[index]['user_id']
        video_id = self.history_behavior.iloc[index]['video_id']

        # 从行为日志 去 索引用户的特征
        user_province = self.user_features.loc[user_id]['province']
        user_city = self.user_features.loc[user_id]['city']
        user_device = self.user_features.loc[user_id]['device_name']

        # 从行为日志索引视频特征
        current_feature = self.video_features.loc[self.video_features['video_id']==video_id]
        together_feature = None
        if len(current_feature)>0:
            tmp_feature = []
            video_tags = current_feature['video_tags'].values[0]
            video_director = current_feature['video_director_list'].values[0]
            video_actor = current_feature['video_actor_list'].values[0]
            for x in [video_tags,video_director,video_actor]:
                if type(x) is not float:
                    tmp_feature.append(x)
            together_feature = ','.join(tmp_feature)
        else:
            together_feature = 'None'

        video_feature_vec = to_text_vector(together_feature, self.w2v)
        feed_dict = {
            'user_id': user_id,
            'video_id': video_id,
            'user_province': user_province,
            'user_city': user_city,
            'user_device': user_device,
            'video_feature_vec':video_feature_vec
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
    
