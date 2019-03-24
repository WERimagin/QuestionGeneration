import random
import numpy as np
import torch
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from func import constants

#lossの計算
def loss_calc(predict,target):
    #criterion = nn.CrossEntropyLoss(ignore_index=constants.PAD)#<pad>=0を無視
    criterion = nn.CrossEntropyLoss()#<pad>=0を無視
    batch=predict.size(0)
    seq_len=predict.size(1)
    #batchとseq_lenを掛けて一次元にしてentropyを計算
    predict=predict.contiguous().view(batch*seq_len,-1)#(batch*seq_len,vocab_size)
    target=target.contiguous().view(-1)#(batch*seq_len)
    loss=criterion(predict,target)
    return loss

#一つの文につき単語の正解率を計算
#これをbatchにつき計算してsumを返す
def predict_calc(predict,target):
    #predict:(batch,seq_len,embed_size)
    #target:(batch,seq_len)
    type="normal"
    if type=="normal":
        batch=predict.size(0)
        seq_len=predict.size(1)
        predict=predict.contiguous().view(batch*seq_len,-1)
        target=target.contiguous().view(-1)
        predict_rate=(torch.argmax(predict,dim=-1)==target).sum().item()/seq_len
        return predict_rate
    elif type=="bleu":
        predict=torch.argmax(predict,dim=-1).tolist()#(batch,seq_len,embed_size)
        target=target.tolist()#(batch,seq_len)
        predict_sum=0
        for p,t in zip(predict,target):#batchごと
            predict_sum+=nltk.bleu_score.sentence_bleu([p],t)
        return predict_sum

#idからid2wordを使ってwordに戻して返す
def predict_sentence(args,predict,target,id2word):
    if args.beam==False:
        predict=torch.argmax(predict,dim=-1) if predict.dim()==3 else predict
        predict=predict.tolist()#(batch,len)
    target=target.tolist()#(batch,len)
    #EOSの前まで
    #batchの中の一つずつ
    predict_list=[" ".join([id2word[w] for w in sentence[0:index_remake(sentence,constants.EOS)]]) \
                    for sentence in predict] if args.include_pad==False else \
                [" ".join([id2word[w] for w in sentence])\
                    for sentence in predict]
    target_list=[" ".join([id2word[w] for w in sentence[0:index_remake(sentence,constants.EOS)]]) \
                    for sentence in target] if args.include_pad==False else \
                [" ".join([id2word[w] for w in sentence])\
                    for sentence in target]
    #predict_list=[" ".join([id2word[w] for w in sentence[0:index_ramake(sentence,constants.EOS)]])\
    #                                    for sentence in predict]
    return predict_list,target_list

#indexの改造,要素がない場合はリストの長さを返す
def index_remake(sentence_list,word):
    if word in sentence_list:
        return sentence_list.index(word)
    else:
        return len(sentence_list)

"""
def predict_sentence(predict,target,id2word):
    #predict:(batch,beam_width,seq_len)
    #target:(batch,seq_len)
    predict=torch.argmax(predict,dim=-1).tolist()#(batch,seq_len)
    #EOSの前まで
    predict_list=[]
    #batchの中の一つずつ
    predict_list=[[" ".join([id2word[w] for w in sentence[0:index_remake(sentence,constants.EOS)]])\
                                        for sentence in sentences]\
                                        for sentences in predict]
    for sentence in predict:
        sentence=[id2word[w] for w in sentence[0:index_remake(sentence,constants.EOS)]]
        sentence=" ".join(sentence)
        predict_list.append(sentence)
    return predict_list
"""
