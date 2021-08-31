import numpy as np
import pandas as pd 
from gensim import models
import  argparse,math
from tqdm import  tqdm
from gensim.models import Word2Vec

parser = argparse.ArgumentParser()
parser.add_argument('--save_path',type = str,default='word2vec.w2v',help='the path to save Word2Vec weights')
args = parser.parse_args()

video_features = pd.read_hdf('digix-data.hdf', 'video_features')
sentences = []
print('making sentences....')
pbar = tqdm(range(len(video_features)))
for index,data in video_features.iterrows():
    video_tag = data['video_tags']
    video_director = data['video_director_list']
    video_actor = data['video_actor_list']
    tmp_list = []
    for x in [video_tag,video_director,video_actor]:
        if type(x) is not float:
            tmp_list.append(x)
            
    together = ','.join(tmp_list).split(',')
    sentences.append(together)
    pbar.update(1)
    # print(together)# 犯罪片,真事改编,史蒂夫·詹姆斯,Cobe Williams,Jeff Fort,Ameena Matthews,Eric Holder Jr.
sentences.append(['None'])
print('training models')
model = models.Word2Vec(sentences, workers=8, vector_size=100, min_count = 1, window = 3)
print('saving model to %s'%(args.save_path))
model.save(args.save_path)

print('test the result...')
test_res = model.wv.most_similar(u'家庭关系')
print(test_res)