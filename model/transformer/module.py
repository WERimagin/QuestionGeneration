import torch
import torch.nn as nn
import torch.nn.functional as F


#key:(batch,seq_len,dim_k)
#query:(batch,seq_len,dim_k)
#value:(batch,seq_len,dim_v)
def Attention(key,query,value):
    scaler=float(key.size(2)**0.5)#dim_kの平方根

    key=torch.transpose(key,1,2)#(batch,dim_k,seq_len)

    qk=torch.bmm(query,key)#(batch,seq_len,seq_len)
    qk=torch.softmax(torch.div(qk,scaler),dim=-1)#(batch,seq_len,seq_len)
    #qk=torch.softmax(qk,dim=-1)#(batch,seq_len,seq_len)

    output=torch.bmm(qk,value)#(batch,seq_len,dim_v)

    return output

class MultiHeadAttention(nn.Module):
    def __init__(self,args):
        super(MultiHeadAttention,self).__init__()
        self.hidden_size=args.hidden_size
        self.head_num=args.head_num
        self.head_dim=int(self.hidden_size/self.head_num)

        self.W=nn.Linear(100,200)

        self.Wk=nn.Linear(self.hidden_size,self.hidden_size)
        self.Wq=nn.Linear(self.hidden_size,self.hidden_size)
        self.Wv=nn.Linear(self.hidden_size,self.hidden_size)

        self.Wconcat=nn.Linear(self.hidden_size,self.hidden_size)

    #key:(batch,seq_len,dim)
    #query:(batch,seq_len,dim)
    #value:(batch,seq_len,dim)
    def forward(self,key,query,value):
        batch,k_seq_len,dim=key.size()
        _,q_seq_len,_=query.size()
        _,v_seq_len,_=value.size()

        head_k=self.Wk(key)#(batch,seq_len,dim)
        head_q=self.Wq(query)#(batch,seq_len,dim)
        head_v=self.Wv(value)#(batch,seq_len,dim)

        #(batch*head_num,seq_len,head_dim)
        head_k=head_k.view(batch,k_seq_len,self.head_num,self.head_dim).transpose(1,2).contiguous().view(batch*self.head_num,k_seq_len,self.head_dim)
        head_q=head_q.view(batch,q_seq_len,self.head_num,self.head_dim).transpose(1,2).contiguous().view(batch*self.head_num,q_seq_len,self.head_dim)
        head_v=head_v.view(batch,v_seq_len,self.head_num,self.head_dim).transpose(1,2).contiguous().view(batch*self.head_num,v_seq_len,self.head_dim)

        #(batch*head_num,q_seq_len,head_dim)
        output=Attention(head_k,head_q,head_v)

        #(batch,seq_len,dim)
        output=output.view(batch,self.head_num,q_seq_len,self.head_dim).transpose(1,2).contiguous().view(batch,q_seq_len,dim)

        #(batch,seq_len,dim)
        output=self.Wconcat(output)

        return output

class EncoderLayer(nn.Module):
    def __init__(self,args):
        super(EncoderLayer,self).__init__()

        self.hidden_size=args.hidden_size

        self.self_attention=MultiHeadAttention(args)
        self.ff1=nn.Linear(self.hidden_size,self.hidden_size)
        self.ff2=nn.Linear(self.hidden_size,self.hidden_size)
        self.norm=nn.LayerNorm(self.hidden_size)

    #input:(batch,seq_len,dim)
    def forward(self,input):
        #MultiHead
        residual=input
        output=self.self_attention(input,input,input)#(batch,seq_len,dim)
        output=torch.add(output,residual)
        output=self.norm(output)

        ##FF
        residual=output
        output=self.ff2(F.relu(self.ff1(output)))
        output=torch.add(output,residual)
        output=self.norm(output)

        return output

class DecoderLayer(nn.Module):
    def __init__(self,args):
        super(DecoderLayer,self).__init__()
        
        self.hidden_size=args.hidden_size

        self.self_attention=MultiHeadAttention(args)
        self.encdec_attention=MultiHeadAttention(args)
        self.ff1=nn.Linear(self.hidden_size,self.hidden_size)
        self.ff2=nn.Linear(self.hidden_size,self.hidden_size)
        self.norm=nn.LayerNorm(self.hidden_size)

    #input:(batch,seq_len,dim)
    def forward(self,input,encoder_output):

        residual=input
        output=self.self_attention(input,input,input)#(batch,seq_len,dim)
        output=torch.add(output,residual)
        output=self.norm(output)

        residual=output
        output=self.encdec_attention(encoder_output,output,encoder_output)#(batch,seq_len,dim)
        output=torch.add(output,residual)
        output=self.norm(output)

        residual=output
        output=self.ff2(F.relu(self.ff1(output)))
        output=torch.add(output,residual)
        output=self.norm(output)

        return output
