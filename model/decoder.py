import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from func.utils import Word2Id,make_tensor,make_vec,make_vec_c,to_var
from func import constants,Beam
from model.attention import Attention
import heapq
import random

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.vocab_size = args.vocab_size
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.layer_size=args.layer_size
        self.batch_size=0
        self.hidden=0

        self.word_embed=nn.Embedding(self.vocab_size, self.embed_size,padding_idx=constants.PAD,
                                    _weight=torch.from_numpy(args.pretrained_weight).float())
        #self.hidden_exchange=nn.Linear(self.hidden_size*2,self.hidden_size)
        self.gru=nn.GRU(self.embed_size,self.hidden_size,num_layers=args.layer_size,bidirectional=False,dropout=args.dropout,batch_first=True)#decoderは双方向にできない

        self.attention=Attention(args)
        self.attention_wight=nn.Linear(self.hidden_size*3,self.hidden_size*3)
        self.out=nn.Linear(self.hidden_size*3,self.vocab_size)
        self.dropout=nn.Dropout(args.dropout)
        #self.out=nn.Linear(self.hidden_size*1,self.vocab_size)

    #decoderでのタイムステップ（単語ごと）の処理
    #input:(batch,1)
    #encoder_output:(batch,seq_len,hidden_size*direction)
    def decode_step(self,input,encoder_output):
        input=torch.unsqueeze(input,1)#(batch,1)

        embed=self.word_embed(input)#(batch,1,embed_size)

        embed=F.relu(embed)

        output,hidden=self.gru(embed,self.hidden)#(batch,1,hidden_size),(2,batch,hidden_size)

        self.hidden=hidden  #(2,batch,hidden_size)

        output=torch.squeeze(output,1)#(batch,hidden_size)

        use_attention=True
        #attentionの計算
        if use_attention:
            #encoderの出力と合わせてアテンションを計算
            attention_output=self.attention(output,encoder_output)#(batch,hidden_size*2)

            #アテンションの重みと元々の出力の重み和を計算してrelu
            #このフェーズは無くても良い(Opennmtなど)
            output=self.attention_wight(torch.cat((output,attention_output),dim=-1))#(batch,hidden_size*3)

        #relu
        output=self.dropout(F.relu(output))#(barch,hidden_size*3)

        #単語辞書のサイズに変換する
        output=self.out(output)#(batch,vocab_size)

        #outputの中で最大値（実際に出力する単語）を返す
        predict=torch.argmax(output,dim=-1) #(batch)

        return output,predict

    #encoder_output:(batch,seq_len,hidden_size*direction)
    #encoder_hidden:(direction*layer_size,batch,hidden_size)
    #output_words:(batch,output_seq_len)
    def forward(self,encoder_output,encoder_hidden,output_words,train=True):
        batch_size=output_words.size(0)
        output_seq_len=output_words.size(1)-1

        #初期隠れベクトル、batch_first=Trueでも(1,batch,hidden_size)の順番、正直無くても良い
        encoder_hidden=encoder_hidden.view(2,self.layer_size,batch_size,self.hidden_size)#次の行でaddするために分割
        self.hidden=torch.add(encoder_hidden[0],encoder_hidden[1])#(layer_size,batch,hidden_size)

        source = output_words[:, :-1]
        target = output_words[:, 1:]

        #use_teacherがFalseだとほとんど学習できない。テストの時のみ
        #他のものだとuse_teacherの割合が0.5で使用している。1でもいいはず。要調整
        #1なら全て正解データ、0なら全て出力されたデータ
        use_teacher=train

        #出力の長さ。教師がない場合は20で固定
        output_maxlen=output_seq_len
        teacher_forcing_ratio=1

        #decoderからの出力結果
        outputs=to_var(torch.from_numpy(np.zeros((output_seq_len,batch_size,self.vocab_size))))
        predict=to_var(torch.from_numpy(np.array([constants.SOS]*batch_size,dtype="long")))#(batch_size)
        for i in range(output_maxlen):
            #使用する入力。
            current_input=source[:,i] if random.random()<teacher_forcing_ratio else predict.view(-1)#(batch)
            output,predict=self.decode_step(current_input,encoder_output)#(batch,vocab_size),(batch)
            outputs[i]=output#outputsにdecoderの各ステップから出力されたベクトルを入力

        """
        #教師無しの学習:predictをcurrent_inputに投げている
        else:
            output_maxlen=20
            #decoderからの出力結果
            outputs=to_var(torch.from_numpy(np.zeros((output_maxlen,batch_size,self.vocab_size))))
            #decoderに渡す最初の<SOS>ベクトル
            current_input=to_var(torch.from_numpy(np.array([constants.SOS]*batch_size,dtype="long")))#(batch_size)
            for i in range(output_maxlen):
                output,predict=self.decode_step(current_input,encoder_output)#(batch,vocab_size),(batch)
                current_input=predict.view(-1,1)#(batch,1)
                outputs[i]=output#outputsにdecoderの各ステップから出力されたベクトルを入力
        """
        outputs=torch.transpose(outputs,0,1)#(batch,seq_len,vocab_size)
        return outputs

    #encoder_output:(batch,seq_len,hidden_size*direction)
    #encoder_hidden:(direction*layer_size,batch,hidden_size)
    #output_words:(batch,output_seq_len)
    def beam_decode(self,encoder_output,encoder_hidden,output_words,train=True):
        batch_size=output_words.size(0)
        output_seq_len=output_words.size(1)
        beam_width=1#beam探索の幅

        #初期隠れベクトル、batch_first=Trueでも(1,batch,hidden_size)の順番、正直無くても良い
        #self.hidden=torch.unsqueeze(torch.add(encoder_hidden[0],encoder_hidden[1]),0)#(1,batch,hidden_size)
        encoder_hidden=encoder_hidden.view(2,self.layer_size,batch_size,self.hidden_size)#次の行でaddするために分割
        #beamのため、それぞれ3倍
        output=torch.add(encoder_hidden[0],encoder_hidden[1])#(layer_size,batch,hidden_size)
        self.hidden=output.repeat(1,beam_width,1)
        encoder_output=encoder_output.repeat(beam_width,1,1)
        #decoderに渡す最初の<SOS>ベクトル
        #current_input=to_var(torch.from_numpy(np.array([[constants.SOS]*1]*batch_size,dtype="long")))#(batch_size,1)
        #use_teacherがFalseだとほとんど学習できない。テストの時のみ
        #他のものだとuse_teacherの割合が0.5で使用している。1でもいいはず。要調整

        #教師無しの学習:predictをcurrent_inputに投げている
        output_maxlen=5
        #decoderからの出力結果
        #trainの時とは違い、vocabサイズでは無く、beam_widthサイズの単語のみ保存
        outputs=[]
        new_batch_words=[]
        current_input=make_tensor([[constants.SOS]*1]*batch_size*beam_width)#(batch_size*beam_width,1)
        beam=Beam()
        for i in range(output_maxlen):
            output,predict=self.decode_step(current_input,encoder_output)#(batch*beam_width,vocab_size),(batch*beam_width)
            output=output.view(batch_size,beam_width,self.vocab_size)#(batch,beam_width,vocab_size)

            batch_words=new_batch_words#(batch,beam_width)(prob,indexs)
            new_batch_words=[]

            #batchごとに処理
            for i in range(batch_size):
                #上位beam_widthの単語について確率を計算していく
                print("now")
                if len(batch_words)==0 or 3>0:
                    words=[[output[i][0][k],[k]] for k in range(self.vocab_size)]#(vocab_size)
                else:
                    words=[]
                    for j in range(beam_width):
                        #新しいwordを加えてprobとindexを更新
                        #これをvocab_size分生成する
                        #(vocab_size)(prob,indexs)
                        word=[[batch_words[i][j][0]*output[i][j][k].item(),batch_words[i][j][1]+[k]] for k in range(self.vocab_size)]
                        words.extend(word)
                #words:(beam_width*vocab_size)(prob,indexs)
                #これをソートし、上位beam_widthを得る
                print("now")
                #words=sorted(words,key=lambda x:x[0])[0:beam_width]#(beam_width)
                #words=heapq.nlargest(beam_width,words)
                words=words[0:beam_width]
                new_batch_words.append(words)
            #new_batch_words(batch,beam_width)
            current_input=[[words[1][-1] for words in beam_words] for beam_words in new_batch_words]#(batch,beam_width)

            current_input=make_tensor(current_input).view(-1,1)#(batch*beam_width,1)

        outputs=[[words[1] for words in beam_words] for beam_words in new_batch_words]#(batch,beam_width,seq_len)
        #outputs=torch.transpose(outputs,0,1)#(batch,seq_len,vocab_size)
        return outputs
