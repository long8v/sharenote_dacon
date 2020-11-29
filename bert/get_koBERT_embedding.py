#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook


# In[2]:


from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model


# In[3]:


from transformers import AdamW
from transformers.optimization import WarmupLinearSchedule


# In[4]:


import sys
sys.path.append('kobert/')


# In[5]:


import os
import sys
import requests
import hashlib

import torch

from transformers import BertModel, BertConfig
import gluonnlp as nlp

from utils import download as _download
from utils import tokenizer

pytorch_kobert = {
    'url':
    'https://kobert.blob.core.windows.net/models/kobert/pytorch/pytorch_kobert_2439f391a6.params',
    'fname': 'pytorch_kobert_2439f391a6.params',
    'chksum': '2439f391a6'
}

bert_config = {
    'attention_probs_dropout_prob': 0.1,
    'hidden_act': 'gelu',
    'hidden_dropout_prob': 0.1,
    'hidden_size': 768,
    'initializer_range': 0.02,
    'intermediate_size': 3072,
    'max_position_embeddings': 512,
    'num_attention_heads': 12,
    'num_hidden_layers': 12,
    'type_vocab_size': 2,
    'vocab_size': 8002
}


def get_pytorch_kobert_model(ctx='cpu', cachedir='~/kobert/'):
    # download model
    model_info = pytorch_kobert
    model_path = _download(model_info['url'],
                           model_info['fname'],
                           model_info['chksum'],
                           cachedir=cachedir)
    # download vocab
    vocab_info = tokenizer
    vocab_path = _download(vocab_info['url'],
                           vocab_info['fname'],
                           vocab_info['chksum'],
                           cachedir=cachedir)
    return get_kobert_model(model_path, vocab_path, ctx)


def get_kobert_model(model_file, vocab_file, ctx="cpu"):
    bertmodel = BertModel(config=BertConfig.from_dict(bert_config))
    bertmodel.load_state_dict(torch.load(model_file))
    device = torch.device(ctx)
    bertmodel.to(device)
    bertmodel.eval()
    vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_file,
                                                         padding_token='[PAD]')
    return bertmodel, vocab_b_obj


# In[6]:


##GPU 사용 시
device = torch.device("cuda:0")


# In[7]:


bertmodel, vocab = get_pytorch_kobert_model()


# In[8]:


tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


# In[9]:


import json


# In[10]:


data = []
with open('dacon_ext_summary/data/train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(line)
train_dict = {}

for data_str in data:
    item = eval(data_str)
    train_dict[item["id"]] = {"media" : item["media"], 
                             "article_original" : item["article_original"],
                             "abs":item["abstractive"],
                             "ext":item["extractive"]}


# In[11]:


bert_data = []


# In[12]:


for key, value in train_dict.items():
    bert_data.append([value['article_original'], [1 if idx in value['ext'] 
                                                 else 0 for idx, sen in enumerate(value['article_original'])]])


<<<<<<< HEAD
np.array(list(zip(*bert_data))[0]).save('label.npy')
daf# In[13]:
=======
>>>>>>> 2c4a34268b8c09a317a7e2ece7f31f2fcee5ca16


bert_data[0]


# In[14]:


transform = nlp.data.BERTSentenceTransform(
            tok, max_seq_length=64, pad=True, pair=False)


# In[15]:


transform(['안녕하세요?ㄹㄹㄹ방가방가'])


# In[16]:


transform(tok.sentencepiece(bert_data[0][0][0]))


# In[17]:


# for sen, idx in bert_data:
#     for s in sen:
#         print(s)
#         print(transform([s]))
#     break


# In[18]:


bert_data_transform = [([transform([s]) for s in sen], idx) for sen, idx in bert_data]


# In[19]:


# bert_data_transform[0]


# In[20]:


import torch
from kobert.pytorch_kobert import get_pytorch_kobert_model


# In[21]:


input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
model, vocab  = get_pytorch_kobert_model()
sequence_output, pooled_output = model(input_ids, input_mask, token_type_ids)
pooled_output.shape
torch.Size([2, 768])
vocab
# Last Encoding Layer
sequence_output[0]


# In[24]:


import pickle


# In[27]:




# In[25]:

i = 0
documents = torch.Tensor([]) 
for x, label in bert_data_transform:
<<<<<<< HEAD
    document = torch.Tensor([])
    for input_tuple in x:
        input_ids, valid_len, token_type_ids = map(lambda e: torch.LongTensor(e).unsqueeze(0), input_tuple)
        sequence_output, pooled_output = model(input_ids, token_type_ids, token_type_ids)
        document = torch.cat([document, pooled_output])    

    torch.save(document, 'file/doc_{}.pt'.format(i))
=======
    if i > 11926:
        print(i)
        document = torch.Tensor([])
        for input_tuple in x:
            input_ids, valid_len, token_type_ids = map(lambda e: torch.LongTensor(e).unsqueeze(0), input_tuple)
            sequence_output, pooled_output = model(input_ids, token_type_ids, token_type_ids)
            document = torch.cat([document, pooled_output])    

    torch.save(document.to('cpu'), 'file/doc_{:05d}.pt'.format(i))
>>>>>>> 2c4a34268b8c09a317a7e2ece7f31f2fcee5ca16
    i += 1






