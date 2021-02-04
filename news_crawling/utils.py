
keyword_group="보험"


keyword_dict={
    "보험" : ["삼성생명", "생명보험", "생보사", "금융위원회", "실손보험", "손해보험", "손보사", "변액보험", "연금보험", " 종신보험", "금융+보험", "약관대출", "금융당국+보험", "금감원+보험", "금융감독원+보험", "GA+보험", "보험업계", "보험+디지털", "보험+핀테크"],
    "디지털" : ["빅데이터", "AI -조류", "딥러닝", "인공지능", "핀테크", "테크핀", "빅데이터+보험", "AI+보험", "인공지능+보험", "삼성생명+디지털"]
}


######################################
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import WarmupLinearSchedule

import pickle

from nltk.tokenize import sent_tokenize
import pandas as pd

from glob import glob 
import datetime
from get_news_data import get_news_data

## Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

#####################################
news_dic = get_news_data(keyword_group,keyword_dict[keyword_group])

def get_dataloder(news_dic):
    
    dataset =[]
    index_to_cont = {}
    for item in news_dic.items():

        for idx, cont in enumerate(item[1]["content"]):

            cont_index = item[0] + str(idx).zfill(3)

            dataset.append([cont, cont_index])
            index_to_cont[cont_index]={"cont":cont}


    _, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    dataset = BERTDataset(dataset, 0, 1, tok, max_len, True, False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=5, shuffle=False)
    
    return dataloader

def get_index_to_cont(news_dic):
    dataset =[]
    index_to_cont = {}
    for item in news_dic.items():

        for idx, cont in enumerate(item[1]["content"]):

            cont_index = item[0] + str(idx).zfill(3)

            dataset.append([cont, cont_index])
            index_to_cont[cont_index]={"cont":cont}
            
    return index_to_cont

def get_model():
    model_name = "./model/[SOTA]kobert_downstream_3epoch.sav"
    model = pickle.load(open(model_name, 'rb'))

    return model


def inference(dataloader, model):
    result_dict = {}

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        #label = label.long().to(device)
        #order_ratios=torch.from_numpy(np.asarray(np.float32(order_ratios))).long().to(device)
        score = model(token_ids, valid_length, segment_ids)
        score = score.to("cpu")
        index_to_cont[label[0]]["score"]=score.detach().numpy()[0][1]
        
def get_ext_summary(news_dic, index_to_cont):
    for item in tqdm(news_dic.items()):
        news_index = item[0]
        cont_index = [x[0] for x in index_to_cont.items() if x[0][:10]==news_index]
        output_sorted = sorted({k: index_to_cont[k] for k in index_to_cont.keys() if k in cont_index}.items(), key=lambda x: x[1]["score"], reverse=True)

        news_dic[item[0]]["ext_summary"] = [int(x[0][-3:]) for x in output_sorted][:3]
        
        return news_dic
    
def t
    


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))
 
    def __len__(self):
        return (len(self.labels))



from get_news_data import get_news_data
    
dataset =[]
index_to_cont = {}
for item in news_dic.items():
    
    for idx, cont in enumerate(item[1]["content"]):
        
        cont_index = item[0] + str(idx).zfill(3)
                
        dataset.append([cont, cont_index])
        index_to_cont[cont_index]={"cont":cont}


_, vocab = get_pytorch_kobert_model()
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

dataset = BERTDataset(dataset, 0, 1, tok, max_len, True, False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=5, shuffle=False)




##GPU 사용 시
device = torch.device("cuda:0")