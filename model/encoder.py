import torch
import torch.nn as nn
import torch.nn.functional as F
from func import constants

class Encoder(nn.Module):

    def __init__(self, args):
        super(Encoder, self).__init__()

        self.word_embed=nn.Embedding(args.vocab_size, args.embed_size,padding_idx=constants.PAD)
        #self.word_embed=nn.Embedding(args.vocab_size, args.embed_size,padding_idx=constants.PAD,\
        #                            _weight=torch.tensor(args.pretrained_weight,dtype=torch.float).to(args.device))
        self.gru=nn.GRU(args.embed_size,args.hidden_size,num_layers=args.layer_size,bidirectional=True,dropout=args.dropout,batch_first=True)

        self.dropout=nn.Dropout(args.dropout)

    def forward(self,input):#input:(batch,seq_len)
        #単語ベクトルへ変換
        embed = self.word_embed(input)#(batch,seq_len,embed_size)

        embed=self.dropout(embed)

        #GRUに投げる（単語ごとではなくEncoderではシーケンスを一括）
        output, hidden=self.gru(embed) #(batch,seq_len,hidden_size*direction),(direction*layer_size,batch,hidden_size)

        return output, hidden
