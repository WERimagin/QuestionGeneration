import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from func.utils import Word2Id,make_tensor,make_vec,make_vec_c,to_var
from func import constants,Beam
from model.attention import Attention
from queue import PriorityQueue
import random

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.vocab_size = args.vocab_size
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.layer_size=args.layer_size
        self.batch_size=0
        self.device=args.device
        self.beam_width=args.beam_width
        self.teacher_rate=args.teachder_rate

        self.word_embed=nn.Embedding(args.vocab_size, args.embed_size,padding_idx=constants.PAD)
        #self.word_embed=nn.Embedding(args.vocab_size, args.embed_size,padding_idx=constants.PAD,\
        #                            _weight=torch.tensor(args.pretrained_weight,dtype=torch.float).to(args.device))
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
    def decode_step(self,input,decoder_hidden,encoder_output):

        input=torch.unsqueeze(input,1)#(batch,1)
        embed=self.word_embed(input)#(batch,1,embed_size)
        embed=self.dropout(embed)
        embed=F.relu(embed)

        output,decoder_hidden=self.gru(embed,decoder_hidden.contiguous())#(batch,1,hidden_size),(2,batch,hidden_size)
        output=self.dropout(output)
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

        return output,decoder_hidden

    #encoder_output:(batch,seq_len,hidden_size*direction)
    #encoder_hidden:(direction*layer_size,batch,hidden_size)
    #output_words:(batch,output_seq_len)
    def forward(self,encoder_output,encoder_hidden,output_words,train=True):
        batch_size=output_words.size(0)
        output_seq_len=output_words.size(1)-1

        #初期隠れベクトル、batch_first=Trueでも(1,batch,hidden_size)の順番、正直無くても良い
        encoder_hidden=encoder_hidden.view(2,self.layer_size,batch_size,self.hidden_size)#次の行でaddするために分割
        decoder_hidden=torch.add(encoder_hidden[0],encoder_hidden[1])#(layer_size,batch,hidden_size)

        source = output_words[:, :-1]
        target = output_words[:, 1:]

        #use_teacherがFalseだとほとんど学習できない。テストの時のみ
        #他のものだとuse_teacherの割合が0.5で使用している。1でもいいはず。要調整
        #1なら全て正解データ、0なら全て出力されたデータ

        #出力の長さ。教師がない場合は20で固定
        output_maxlen=output_seq_len
        teacher_forcing_ratio=self.teacher_rate if train else 0

        #decoderからの出力結果
        outputs=torch.from_numpy(np.zeros((output_seq_len,batch_size,self.vocab_size))).to(self.device)
        predict=torch.from_numpy(np.array([constants.SOS]*batch_size,dtype="long")).to(self.device) #(batch_size)

        for i in range(output_maxlen):
            #使用する入力。
            current_input=source[:,i] if random.random()<teacher_forcing_ratio else predict.view(-1)#(batch)
            output,decoder_hidden=self.decode_step(current_input,decoder_hidden,encoder_output)#(batch,vocab_size),(batch)
            outputs[i]=output#outputsにdecoderの各ステップから出力されたベクトルを入力

        outputs=torch.transpose(outputs,0,1)#(batch,seq_len,vocab_size)

        return outputs

    #https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
    #encoder_output:(batch,seq_len,hidden_size*direction)
    #encoder_hidden:(direction*layer_size,batch,hidden_size)
    #output_words:(batch,output_seq_len)
    def beam_decode(self,encoder_outputs,encoder_hidden,output_words,train=True):

        beam_width=self.beam_width
        topk=1  # how many sentence do you want to generate
        decoded_batch=[]
        batch_size=output_words.size(0)
        output_seq_len=output_words.size(1)-1

        # decoding goes sentence by sentence
        #バッチの中の文ごと
        for idx in range(output_words.size(0)):
            encoder_hidden=encoder_hidden.view(2,self.layer_size,batch_size,self.hidden_size)
            decoder_hidden=torch.add(encoder_hidden[0],encoder_hidden[1])[:,idx,:].unsqueeze(1)#(layer_size,1,hidden_size)
            encoder_output = encoder_outputs[idx,:, :].unsqueeze(0)#(1,length,hidden)

            #最初の入力
            decoder_input = torch.LongTensor([constants.SOS]).to(self.device)#(1)

            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))#1

            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            nodes.put((-node.eval(), node))#-0.0
            qsize = 1

            #beam開始
            #nodesから単語idとhiddenを取り出して、decoderに投げる
            #出力結果からbeam_width分の結果を取り出し、nodesに入れる
            while True:
                if qsize > 500: break

                #nodeの取り出し
                score, n = nodes.get()
                decoder_input = n.wordid
                decoder_hidden = n.h

                #EOSの判定。EOSならendnodesに入れて次へ。文の個数が溜まってたら終了
                if n.wordid.item() == constants.EOS and n.prevNode != None:
                    endnodes.append((score,n))
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                #(1,vocab_size),(1,hidden_size)
                decoder_output,decoder_hidden=self.decode_step(decoder_input,decoder_hidden,encoder_output)
                log_prob, indexes = torch.topk(decoder_output, beam_width)#(1,beam_width),(1,beam_width)

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(1)#(1)
                    log_p = log_prob[0][new_k].item()
                    #前のスコアとの和を取る
                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nodes.put((score,node))
                qsize+=(beam_width-1)

            endnodes=[nodes.get() for _ in range(topk)] if len(endnodes)==0 else endnodes #(topk,len)
            utterances=[]
            #endnodesをidに変換
            for score, n in sorted(endnodes, key=lambda x:x[0]):
                utterance = []
                utterance.append(n.wordid.item())
                # back trace
                #noneが出るまで（つまり、一番先頭に辿り着くまで）
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid.item())

                #終わりからなので逆にする。SOSは除く
                utterances.append(utterance[-2::-1])#(topk,len)

            decoded_batch.append(utterances[0])#(batch,len)

        return decoded_batch

# starting node -  hidden vector, previous node, word id, logp, length
#node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
#nodes = PriorityQueue()
class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        self.h = hiddenstate
        self.prevNode = previousNode#再帰的に前のノードを記録しておく
        self.wordid = wordId#d
        self.logp = logProb#0
        self.leng = length#1

    def eval(self, alpha=1.0):
        reward = 0
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
