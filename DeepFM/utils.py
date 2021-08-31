import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd



def to_text_vector(txt, model):
    '''
        将文本txt转化为文本向量
    '''
    words = txt.split(',') 
    array = np.asarray([model.wv[w] for w in words if w in words],dtype='float32') 
    return array.mean(axis=0)