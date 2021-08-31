import numpy as np
import pandas as pd 
from gensim import models
from gensim.models import Word2Vec
from utils import to_text_vector
model = Word2Vec.load('word2vec.w2v')
# test_res = model.wv.most_similar(u'家庭关系')
f1 = '古装喜剧,剧情片,喜剧片,内地电影'
f2 = '犯罪片,真事改编'

res1 = to_text_vector(f1,model)
res2 = to_text_vector(f2,model)

print(len(res1))
print(len(res2))