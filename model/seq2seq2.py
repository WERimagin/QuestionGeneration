#https://www.pytry3g.com/entry/pytorch-seq2seq

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from func import constants
from func.utils import Word2Id,make_tensor,make_vec,make_vec_c,to_var
import numpy as np

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.word_embeddings = nn.Embedding(args.vocab_size, args.embed_size, padding_idx=constants.PAD)
        self.gru = nn.GRU(args.embed_size,args.hidden_size, batch_first=True)

    def forward(self, indices):
        embedding = self.word_embeddings(indices)
        if embedding.dim() == 2:
            embedding = torch.unsqueeze(embedding, 1)
        _, state = self.gru(embedding)

        return state


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.word_embeddings = nn.Embedding(args.vocab_size, args.embed_size, padding_idx=constants.PAD)
        self.gru = nn.GRU(args.embed_size,args.hidden_size, batch_first=True)
        self.output = nn.Linear(args.hidden_size, args.vocab_size)

    def forward(self, index, state):
        embedding = self.word_embeddings(index)
        if embedding.dim() == 2:
            embedding = torch.unsqueeze(embedding, 1)
        gruout, state = self.gru(embedding, state)
        output = self.output(gruout)
        return output, state

class Seq2Seq2(nn.Module):
    def __init__(self,args):
        super(Seq2Seq2, self).__init__()
        self.vocab_size=args.vocab_size
        self.encoder=Encoder(args)
        self.decoder=Decoder(args)

    #input_words:(batch,seq_len)
    def forward(self, input_words,output_words,train=True,beam=False):
        batch_size=output_words.size(0)
        output_seq_len=output_words.size(1)-1

        encoder_hidden = self.encoder(input_words)#(batch,hidden_size)
        # Create source and target
        source = output_words[:, :-1]
        target = output_words[:, 1:]
        decoder_hidden = encoder_hidden

        outputs=to_var(torch.from_numpy(np.zeros((output_seq_len,batch_size,self.vocab_size))))

        # Forward batch of sequences through decoder one time step at a time
        loss = 0
        for i in range(source.size(1)):
            decoder_output, decoder_hidden = self.decoder(source[:, i], decoder_hidden)#(batch,1,vocab_size,batch,1,hidden_size)
            decoder_output = torch.squeeze(decoder_output)#(batch,vocab_size)
            outputs[i]=decoder_output
            #loss += criterion(decoder_output, target[:, i])
        outputs=torch.transpose(outputs,0,1)#(batch,seq_len,vocab_size)
        return outputs
