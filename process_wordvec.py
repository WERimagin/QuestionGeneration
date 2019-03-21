#SQuADのデータ処理
#必要条件:CoreNLP
#Tools/core...で
#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

import os
import sys
sys.path.append("../")
import json
import gzip
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize,sent_tokenize
import pickle
import collections
import random

from func.corenlp import CoreNLP

def answer_find(context_text,answer_start,answer_end):
    context=sent_tokenize(context_text)
    start_p=0

    #start_p:対象となる文の文字レベルでの始まりの位置
    #end_p:対象となる文の文字レベルでの終端の位置
    #answer_startがstart_pからend_pの間にあるかを確認。answer_endも同様
    for i,sentence in enumerate(context):
        end_p=start_p+len(sentence)
        if start_p<=answer_start and answer_start<=end_p:
            sentence_start_id=i
        if start_p<=answer_end and answer_end<=end_p:
            sentence_end_id=i
        #スペースが消えている分の追加、end_pの計算のところでするべきかは不明
        start_p+=len(sentence)+1

    #得られた文を結合する（大抵は文は一つ）
    answer_sentence=" ".join(context[sentence_start_id:sentence_end_id+1])

    return answer_sentence

#sentenceを受け取り、tokenizeして返す
def tokenize(sent):
    return [token.replace('``','"').replace("''",'"') for token in word_tokenize(sent)]

#単語のリストとgloveのword2vecからword2idとid2vecを生成し、保存
#wordcount:dict
def vec_process(word_count):
    vec_size=300
    path="../data/glove.840B.300d.txt" if vec_size==300 else "data/glove.6B.{}d.txt".format(vec_size)

    word_count=sorted(list(word_count.items()),key=lambda x:-x[1])
    word2id={w:i for i,(w,count) in enumerate(word_count,6)}
    word2id["<PAD>"]=0
    word2id["<UNK>"]=1
    word2id["<SOS>"]=2
    word2id["<EOS>"]=3
    word2id["<SEP>"]=4
    word2id["<SEP2>"]=5

    w2vec={}#wordとvecの対応辞書
    id2vec=np.zeros((len(list(word2id.items())),vec_size))#wordとidの対応辞書

    if os.path.exists(path)==True:
        with open(path,"r")as f:
            for j,line in tqdm(enumerate(f)):
                line_split=line.split()
                w2vec[" ".join(line_split[0:-300])]=[float(i) for i in line_split[-300:]]

        for w,i in tqdm(word2id.items()):
            if w in w2vec:
                id2vec[i]=w2vec[w]

    with open("data/word2id.json","w")as f:
        t={"word2id":word2id,
            "id2vec":id2vec.tolist()}
        json.dump(t,f)

def data_process(input_path,output_path,train=False):
    with open(input_path,"r") as f:
        data=json.load(f)

    #corenlp=CoreNLP()

    questions=[]
    answers=[]
    sentences=[]
    question_interros=[]
    neg_interros=[]

    pairs=[]
    word_count=collections.defaultdict(int)

    all_count=0

    #context_text:文章
    #question_text:質問
    #answer_text:解答
    #answer_start,answer_end:解答の文章の中での最初と最後の位置
    for topic in tqdm(data["data"]):
        topic=topic["paragraphs"]
        for paragraph in topic:
            context_text=paragraph["context"].lower()
            for qas in paragraph["qas"]:
                all_count+=1

                question_text=qas["question"].lower()
                if len(qas["answers"])==0:
                    print("answer=0")
                    continue
                a=qas["answers"][0]
                answer_text=a["text"].lower()
                answer_start=a["answer_start"]
                answer_end=a["answer_start"]+len(a["text"])

                if len(question_text)<=5:#ゴミデータ(10個程度)は削除
                    continue

                #contextの中からanswerが含まれる文を見つけ出す
                sentence_text=answer_find(context_text,answer_start,answer_end)

                question_text=" ".join(tokenize(question_text))
                sentence_text=" ".join(tokenize(sentence_text))
                answer_text=" ".join(tokenize(answer_text))

                for word in question_text.split():
                    word_count[word]+=1
                for word in sentence_text.split():
                    word_count[word]+=1


    if train==True:
        #word_countから単語ベクトルを生成し、保存する
        vec_process(word_count)

if __name__ == "__main__":
    #main
    random.seed(0)


    data_process(input_path="data/squad-train-v1.1.json",
                output_path="data/train_data.json",
                train=True
                )
