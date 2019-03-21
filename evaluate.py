#流れ
#0.データの処理->prepro.shで実行、dataから必要なデータを取り出しpickle化、word2id,id2vecの処理
#1.contexts,questionsを取り出しid化
#2.dataloaderからbatchを取り出し(ただのshuffleされたid列)、それに従いbatchを作成してtorch化
#3.モデルに入れてp1,p2(スタート位置、エンド位置を出力)
#4.predictはp1,p2それぞれのargmaxを取り、それと正解の位置を比較して出力する

import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append("../")
from tqdm import tqdm
import nltk
import pickle
import json

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import time
from model.seq2seq import Seq2Seq
from model.seq2seq2 import Seq2Seq2
from func.utils import Word2Id,BatchMaker,make_vec,make_vec_c,to_var,logger,get_args,data_loader,loss_calc,predict_calc,predict_sentence
from func import constants
import nltk


#epochあたりの学習
def model_handler(args,data,train=True):
    start=time.time()
    sentences=data["sentences"]
    questions=data["questions"]
    id2word=data["id2word"]
    data_size=len(questions)


    batch_size=args.test_batch_size
    model.eval()
    #batchをランダムな配列で指定する
    batchmaker=BatchMaker(data_size,batch_size,train)
    batches=batchmaker()
    predict_rate=0
    loss_sum=0
    #生成した文を保存するリスト
    predicts=[]
    for i_batch,batch in tqdm(enumerate(batches)):
        #これからそれぞれを取り出し処理してモデルへ
        input_words=make_vec([sentences[i] for i in batch])
        output_words=make_vec([questions[i] for i in batch])#(batch,seq_len)
        #modelにデータを渡してpredictする
        predict=model(input_words,output_words,train)#(batch,seq_len,vocab_size)
        predict=predict_sentence(predict,output_words[:,1:],id2word)#(batch,seq_len)
        predicts.extend(predict)

    sentences=[" ".join([id2word[id] for id in sentence]) for sentence in sentences]#idから単語へ戻す
    questions=[" ".join([id2word[id] for id in sentence[1:-1]]) for sentence in questions]#idから単語へ戻す

    with open("data/predict_sentences.json","w")as f:
        data={"sentences":sentences,
                "questions":questions,
                "predicts":predicts}
        json.dump(data,f)

##start main
args=get_args()
test_data=data_loader(args,"data/test_data.json",first=True) if args.use_train_data==False else \
            data_loader(args,"data/train_data.json",first=True)
model=Seq2Seq(args) if args.model_version==1 else \
        Seq2Seq2(args)



if args.model_name!="":
    param = torch.load("model_data/{}".format(args.model_name))
    model.load_state_dict(param)
#start_epochが0なら最初から、指定されていたら学習済みのものをロードする
elif args.start_epoch>=1:
    param = torch.load("model_data/epoch_{}_model.pth".format(args.start_epoch-1))
    model.load_state_dict(param)
else:
    args.start_epoch=0


#pytorch0.4より、OpenNMT参考
device=torch.device("cuda:{}".format(args.cuda_number) if torch.cuda.is_available() else "cpu")
model.to(device)

model_handler(args,test_data,train=False)
