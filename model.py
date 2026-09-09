import os
import torch
import torch.nn as nn
import math

class embedding(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(self.vocab_size, self.d_model)
    def forward(self,x:torch.tensor)->torch.tensor:
        return self.embed(x) * math.sqrt(self.d_model)

class positionalencoding(nn.Module):
    def __init__(self,seq_len:int,d_model:int,dropout:float):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len,d_model)
        numer = torch.arange(0,seq_len,dtype=torch.float32).unsqueeze(1)
        denom = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(numer*denom)
        pe[:,1::2] = torch.cos(numer*denom)
        pe = pe.unsqueeze(0) #pe -> [1,seq_len,d_model]
        #keeps it in register
        self.register_buffer('pe',pe)
    def forward(self,x:torch.tensor)->torch.Tensor:
        x = x+self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)

class layer_norm(nn.Module):
    def __init__(self,features:int,eps:float=10**-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self,x:torch.tensor)->torch.tensor:
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps)+self.bias

class FeedForward(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout:float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model,d_ff)
        self.linear2 = nn.Linear(d_ff,d_model)
    def forward(self,x:torch.tensor)->torch.tensor:
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,num_heads:int,dropout:float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query:torch.tensor,key:torch.tensor,value:torch.tensor,mask:torch.tensor=None,dropout:nn.Dropout=None)->torch.tensor:
        d_k = query.shape[-1]
        scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value),p_attn

    def forward(self,q:torch.tensor,k:torch.tensor,v:torch.tensor,mask:torch.tensor=None)->torch.tensor:
        # dim of q,k,v -> [batch_size,seq_len,d_model]
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Split into multiple heads
        # dim -> [batch_size,seq_len,num_heads,d_k] -> [batch_size,num_heads,seq_len,d_k]
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        x,self.attention_weights = self.attention(query,key,value,mask,self.dropout)
        # dim -> [batch_size,num_heads,seq_len,d_k] -> [batch_size,seq_len,d_model]
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_k * self.num_heads)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self,features: int,dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = layer_norm(features)
    def forward(self,x:torch.tensor,sub_layer:nn.Module)->torch.tensor:
        return x + self.dropout(sub_layer(self.norm(x)))

class Encoderblock(nn.Module):
    def __init__(self,features: int,self_attention:MultiHeadAttention,feed_forward:FeedForward,dropout:float)->None:
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_conn = nn.ModuleList([ResidualConnection(features,dropout) for _ in range(2)])
    
    def forward(self,x:torch.tensor,mask:torch.tensor)->torch.tensor:
        x = self.residual_conn[0](x,lambda x: self.self_attention(x,x,x,mask))
        x = self.residual_conn[1](x,self.feed_forward)
        return x

class Encoder(nn.Module):
    def __init__(self,features: int,layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = layer_norm(features)
    def forward(self,x:torch.tensor,mask:torch.tensor)->torch.tensor:
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

class Decoderblock(nn.Module):
    def __init__(self,features: int,masked_self_attention:MultiHeadAttention,multi_head_attention:MultiHeadAttention,feed_forward:FeedForward,dropout:float)->None:
        super().__init__()
        self.masked_self_attention = masked_self_attention
        self.multi_head_attention = multi_head_attention
        self.feed_forward = feed_forward
        self.residual_conn = nn.ModuleList([ResidualConnection(features,dropout) for _ in range(3)])
    
    def forward(self,x:torch.tensor,encoder_output:torch.tensor,src_mask:torch.tensor,tgt_mask:torch.tensor)->torch.tensor:
        x = self.residual_conn[0](x,lambda x: self.masked_self_attention(x,x,x,tgt_mask))
        x = self.residual_conn[1](x,lambda x: self.multi_head_attention(x,encoder_output,encoder_output,src_mask))
        x = self.residual_conn[2](x,self.feed_forward)
        return x

class Decoder(nn.Module):
    def __init__(self,features: int,layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = layer_norm(features)
    def forward(self,x:torch.tensor,encoder_output:torch.tensor,src_mask:torch.tensor,tgt_mask:torch.tensor)->torch.tensor:
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)

class Projectionlayer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int)->None:
        super().__init__()
        self.linear = nn.Linear(d_model,vocab_size)
    def forward(self,x:torch.tensor)->torch.tensor:
        return self.linear(x)

class Transformer(nn.Module):
    def __init__(self,encoder:Encoder,decoder:Decoder,src_embed:embedding,tgt_embed:embedding,src_pos:positionalencoding,tgt_pos:positionalencoding,projection_layer:Projectionlayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self,src:torch.tensor,src_mask:torch.tensor)->torch.tensor:
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self,tgt:torch.tensor,encoder_output:torch.tensor,src_mask:torch.tensor,tgt_mask:torch.tensor)->torch.tensor:
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
    
    def project(self,x:torch.tensor)->torch.tensor:
        return self.projection_layer(x)

def make_model(src_vocab_size:int,tgt_vocab_size:int,src_seq_len:int,tgt_seq_len:int,d_model:int=512,d_ff:int=2048,num_heads:int=8,N:int=6,dropout:float=0.1)->Transformer:
    src_embed = embedding(d_model,src_vocab_size)
    tgt_embed = embedding(d_model,tgt_vocab_size)
    
    src_pos = positionalencoding(src_seq_len,d_model,dropout)
    tgt_pos = positionalencoding(tgt_seq_len,d_model,dropout)
    
    encoder_block = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttention(d_model,num_heads,dropout)
        feed_forward = FeedForward(d_model,d_ff,dropout)
        encoder_block.append(Encoderblock(d_model,encoder_self_attention,feed_forward,dropout))

    decoder_block = []
    for _ in range(N):
        decoder_masked_self_attention = MultiHeadAttention(d_model,num_heads,dropout)
        decoder_cross_attention = MultiHeadAttention(d_model,num_heads,dropout)
        feed_forward = FeedForward(d_model,d_ff,dropout)
        decoder_block.append(Decoderblock(d_model,decoder_masked_self_attention,decoder_cross_attention,feed_forward,dropout))
    
    encoder = Encoder(d_model,nn.ModuleList(encoder_block))
    decoder = Decoder(d_model,nn.ModuleList(decoder_block))
    
    projection_layer = Projectionlayer(d_model,tgt_vocab_size)
    
    model = Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model