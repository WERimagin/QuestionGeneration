import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from tqdm import tqdm
from collections import defaultdict
import collections
import math
from statistics import mean, median,variance,stdev
import random
import json


def compute_score(t,p):
    t_dict=collections.Counter(t)
    p_dict=collections.Counter(p)
    common=sum((t_dict & p_dict).values())
    return common

def compute_score_refs(ts,p):
    sum_dict=collections.Counter()
    p_dict=collections.Counter(p)
    for t in ts:
        t_dict=collections.Counter(t)
        sum_dict=(t_dict & p_dict) | sum_dict
    common=sum(sum_dict.values())
    return common

def n_gram(words, n):
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

def head_find(tokens):
    q_head=["what","how","who","when","which","where","why","whose","whom","is","are","was","were","do","did","does"]
    #q_head=["what","how","who","when","which","where","why","whose","whom"]
    for h in q_head:
        if h in tokens:
            return h
    return "<none>"

def head_compare(tgt,pred):
    t=head_find(tgt)
    p=head_find(pred)
    return t==p

random.seed(0)

path="data/predict_sentences.json"
with open(path,"r")as f:
    t=json.load(f)
    targets=t["questions"]
    src=t["sentences"]
    predicts=t["predicts"]


for i in range(100):
    print(src[i])
    print(targets[i])
    print(predicts[i])
    print()

target=[t.split() for t in targets]
predict=[p.split() for p in predicts]

#一文ずつ評価,corpusのサイズ考慮
if True:
    target_dict=defaultdict(lambda: [])
    predict_dict=defaultdict(str)
    src_set=set(src)
    for s,t,p in zip(src,target,predict):
        target_dict[s].append(t)
        predict_dict[s]=p

    target=[target_dict[s] for s in src_set]
    predict=[predict_dict[s] for s in src_set]

    print(len(target),len(predict))


    print(corpus_bleu(target,predict,weights=(1,0,0,0)))
    print(corpus_bleu(target,predict,weights=(0.5,0.5,0,0)))
    print(corpus_bleu(target,predict,weights=(0.333,0.333,0.333,0)))
    print(corpus_bleu(target,predict,weights=(0.25,0.25,0.25,0.25)))


########################

#同じ文はまとめてtargetとして扱う。
#この手法は同じpredictについてもそれぞれ計算。元のはまとめて計算
#shortのoptionについてはほとんど一致
if True:
    print(len(src),len(target),len(predict))
    #src=src[0:len(predict)]
    #target=target[0:len(predict)]
    #predict=predict[0:len(predict)]

    target_dict=defaultdict(lambda:[])
    predict_dict=defaultdict(str)
    src_set=set(src)

    #srcの文が同じものをまとめる。
    for s,t,p in zip(src,target,predict):
        target_dict[s].append(t)
        predict_dict[s]=p

    print("size:{}\n".format(len(target_dict)))

    #ペナルティの計算
    #単語の数が小さければペナルティ
    bleu_score=[]
    count_target=0
    count_predict=0
    for i,s in enumerate(src_set):
        t=target_dict[s]
        p=predict_dict[s]
        c_t=min([len(t2) for t2 in t])
        c_p=len(p)
        count_target+=c_t
        count_predict+=c_p

    penalty=math.exp(1-count_target/count_predict) if count_target>count_predict else 1
    #print(penalty)
    #print(count_target,count_predict)

    #n-gramごとにbleuを計算して平均を取る。
    #本家のbleuの計算はよくわからないので要検証
    for n in range(1,5):
        score_sum=0
        correct_count=0
        total_count=0
        for i,s in enumerate(src_set):
            t=[n_gram(sent,n) for sent in target_dict[s]]
            p=n_gram(predict_dict[s],n)
            correct_num=compute_score_refs(t,p)
            correct_count+=correct_num
            total_count+=len(p)
        print(correct_count,total_count)
        score=correct_count/total_count
        bleu_score.append(score)
        score=penalty*math.exp(mean(map(math.log,bleu_score[0:n])))
        print("{}gram score is {}".format(n,score))
        print()

######################################
