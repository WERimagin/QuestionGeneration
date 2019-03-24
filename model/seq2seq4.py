import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from func import constants
from func.utils import Word2Id,make_tensor,make_vec,make_vec_c,to_var
import numpy as np
import random

class Encoder(nn.Module):

    def __init__(self, args):
        super(Encoder, self).__init__()

        #self.word_embed=nn.Embedding(args.vocab_size, args.embed_size,padding_idx=constants.PAD)
        self.word_embed=nn.Embedding(args.vocab_size, args.embed_size)
        self.gru=nn.GRU(args.embed_size,args.hidden_size,dropout=args.dropout,batch_first=True)

    def forward(self,input):#input:(batch,seq_len)
        #単語ベクトルへ変換
        embed = self.word_embed(input)#(batch,seq_len,embed_size)
        #GRUに投げる（単語ごとではなくEncoderではシーケンスを一括）
        output, hidden=self.gru(embed) #(batch,seq_len,hidden_size*direction),(direction*layer_size,batch,hidden_size)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.vocab_size = args.vocab_size
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.layer_size=args.layer_size
        self.batch_size=0
        self.hidden=0

        #self.word_embed=nn.Embedding(self.vocab_size, self.embed_size,padding_idx=constants.PAD)
        self.word_embed=nn.Embedding(self.vocab_size, self.embed_size)
        #self.hidden_exchange=nn.Linear(self.hidden_size*2,self.hidden_size)
        self.gru=nn.GRU(self.embed_size,self.hidden_size,dropout=args.dropout,batch_first=True)#decoderは双方向にできない

        self.out=nn.Linear(self.hidden_size,self.vocab_size)


    def decode_step(self,input,encoder_output):
        input=torch.unsqueeze(input,1)#(batch,1)

        embed=self.word_embed(input)#(batch,1,embed_size)

        output,hidden=self.gru(embed,self.hidden)#(batch,1,hidden_size),(2,batch,hidden_size)

        self.hidden=hidden  #(2,batch,hidden_size)

        output=torch.squeeze(output,1)#(batch,hidden_size)


        #relu
        output=F.relu(output)

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

        source = output_words[:, :-1]
        target = output_words[:, 1:]

        self.hidden=torch.zeros(1,batch_size,self.hidden_size)

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

        outputs=torch.transpose(outputs,0,1)#(batch,seq_len,vocab_size)
        return outputs

class Seq2Seq4(nn.Module):
    def __init__(self,args):
        super(Seq2Seq4, self).__init__()
        self.encoder=Encoder(args)
        self.decoder=Decoder(args)

    def forward(self, input_words,output_words,train=True,beam=False):
        #Encoderに投げる
        encoder_outputs, encoder_hidden = self.encoder(input_words)#(batch,seq_len,hidden_size*2)
        output=self.decoder(encoder_outputs,encoder_hidden,output_words,train) if beam==False else \
                self.decoder.beam_decode(encoder_outputs,encoder_hidden,output_words,train)
        return output
