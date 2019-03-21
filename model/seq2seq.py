import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from model.encoder import Encoder
from model.decoder import Decoder


class Seq2Seq(nn.Module):
    def __init__(self,args):
        super(Seq2Seq, self).__init__()
        self.encoder=Encoder(args)
        self.decoder=Decoder(args)

    def forward(self, input_words,output_words,train=True,beam=False):
        #Encoderに投げる
        encoder_outputs, encoder_hidden = self.encoder(input_words)#(batch,seq_len,hidden_size*2)
        output=self.decoder(encoder_outputs,encoder_hidden,output_words,train) if beam==False else \
                self.decoder.beam_decode(encoder_outputs,encoder_hidden,output_words,train)
        return output
