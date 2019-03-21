import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):

    def __init__(self, args):
        super(Attention, self).__init__()
        self.hidden_size = args.hidden_size

        self.W=nn.Linear(self.hidden_size,self.hidden_size*2)

        self.attention_wight=nn.Linear(self.hidden_size,self.hidden_size*2)


    #input*W*encoder_outputでscoreを計算(general)
    #decoder_hidden:(batch,hidden_size)
    #encoder_output:(batch,seq_len,hidden_size*2)
    #return:(batch,hidden_size*2)
    def forward(self,decoder_hidden,encoder_output):
        decoder_hidden=torch.unsqueeze(decoder_hidden,dim=1)#(batch,1,hidden_size)
        encoder_output_transpose=torch.transpose(encoder_output,1,2)#(batch,hidden_size*2,seq_len)
        output=self.W(decoder_hidden)#(batch,1,hidden_size*2)
        output=torch.bmm(output,encoder_output_transpose)#(batch,1,seq_len)
        output=F.softmax(output,dim=-1)#(batch,1,seq_len)
        output=torch.bmm(output,encoder_output)#(batch,1,hidden_size*2)
        output=torch.squeeze(output,dim=1)#(batch,hidden_size*2)
        return output

    #scoreをconcatで計算
    #input:(batch,hidden_size)
    #encoder_output:(batch,seq_len,hidden_size*2)
    #return:(batch,hidden_size*2)
    def forward2(self,decoder_hidden,encoder_output):
        seq_len=encoder_output.size(1)
        decoder_hidden=decoder_hidden.unsqueeze(1).repeat(1,seq_len,1)#(batch,seq_len,hidden_size)
        output=torch.cat((decoder_hidden,encoder_output),dim=-1)#(batch,seq_len,hidden_size*3)
        output=torch.tanh(self.W(output))#(batch,seq_len,1)
        output=torch.squeeze(output)#(batch,seq_len)
        output=F.softmax(output,dim=-1)#(batch,seq_len)
        output=torch.unsqueeze(output,dim=1)#(batch,1,seq_len)
        output=torch.bmm(output,encoder_output)#(batch,1,hidden_size*2)
        output=torch.squeeze(output)#(batch,hidden_size*2)
        return output

    #forwardと同じはず
    #input*W*encoder_outputを返す
    #return:(batch,seq_len)(batch,1,hidden_size)*(hidden_size,hidden_size*2)*(batch,seq_len,hidden_size*2)
    #input:(batch,hidden_size),encoder_output:(batch,seq_len,hidden_size*2)
    def forward3(self,input,encoder_output):
        attention_input=torch.unsqueeze(input,1)#(batch,1,hidden_size)
        attention_encoder_input=torch.transpose(encoder_output,1,2)#(batch,hidden_size*2,seq_len)
        #batchの中身ごとに、input*W*encoder_outputを計算
        attention_output=F.softmax(torch.bmm(self.attention_wight(attention_input),attention_encoder_input),dim=-1)#(batch,1,seq_len)
        attention_output=torch.squeeze(torch.bmm(attention_output,encoder_output),1)#(batch,hidden_size*2)
        return attention_output
