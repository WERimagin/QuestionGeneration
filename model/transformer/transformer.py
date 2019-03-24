#https://www.pytry3g.com/entry/pytorch-seq2seq

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from func import constants
from func.utils import Word2Id,make_tensor,make_vec,make_vec_c,to_var
import numpy as np
from model.transformer.module import Attention,MultiHeadAttention,EncoderLayer,DecoderLayer

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.word_embed=nn.Embedding(args.vocab_size, args.hidden_size,padding_idx=constants.PAD)
        self.layers=nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layers)])
    #input:(batch,seq_len)
    def forward(self,input):
        output=self.word_embed(input)#(batch,seq_len,embed_size)

        #6層分
        for layer in self.layers:
            output=layer(output)

        return output


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder,self).__init__()

        self.word_embed=nn.Embedding(args.vocab_size, args.hidden_size, padding_idx=constants.PAD)
        self.layers=nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layers)])
        self.out=nn.Linear(args.hidden_size,args.vocab_size)

    #input:(batch,seq_len)
    def forward(self,input,encoder_output):
        input=input[:,:-1]
        output=self.word_embed(input)#(batch,seq_len,embed_size)

        #6層分のlayerを通す
        for layer in self.layers:
            output=layer(output,encoder_output)

        output=self.out(output)
        #output=torch.softmax(output,dim=-1)


        return output

class Transformer(nn.Module):
    def __init__(self,args):
        super(Transformer, self).__init__()
        args.n_layers=6
        args.head_num=6

        self.encoder=Encoder(args)
        self.decoder=Decoder(args)


    #input_words:(batch,seq_len)
    def forward(self, input_word,output_word,train=True,beam=False):

        encoder_output=self.encoder(input_word)#(batch,seq_len,hidden_size)
        output=self.decoder(output_word,encoder_output)
        return output
