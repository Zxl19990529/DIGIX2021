# DeepFM

环境准备:
- pytorch
- gensim
- pandas
- numpy

## 数据集准备

1. 参考MLP文档中生成```digix-data.hdf```的步骤,生成```digix-data.hdf```
2. ```python prepare_text_vector.py``` 训练 ```WordVec``` 模型,并生成该模型的权重文件,包括以下三个文件:
  - word2vec.w2v
  - word2vec.w2v.syn1neg.npy
  - word2vec.w2v.wv.vectors.npy

## 训练步骤

```python train.py```即可