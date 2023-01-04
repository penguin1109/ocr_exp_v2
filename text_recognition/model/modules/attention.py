""" Transformer Encoder Layer
1. Multi Head Attention
2. Add & Norm(=Layer Norm)
3. Feed Forward Network
4. Add & Norm(=Layer Norm)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


""" Multi Head Attention
1. Key, Value는 같은 값이지만 Query가 다른 값인 경우 (Transformer Decoder)
2. Query, Key, Value가 모두 같은 값일 경우 (Transformer Encoder)
"""

def get_multihead_attention(embed_dim, head_num, same_dim, kdim=None, vdim=None):
  if same_dim:
    return MultiHeadAttention(embed_dim, head_num)
  else:
    return MultiHeadAttention_Diff(embed_dim, head_num, kdim, vdim)

#### Multi Head Attention for the same Key, Query, Value ####
class MultiHeadAttention(nn.Module):
  def __init__(self, 
               embed_dim,
               head_num,
               dropout_rate=0.0):
    super(MultiHeadAttention, self).__init__()
    """ Args
    embed_dim: total dimension of the model
    head_n: number of parallel attention heads
    dropout_rate: Dropout rate on the Dropout Layer to prevent overfitting
    """
    self.embed_dim = embed_dim
    self.head_num = head_num
    self.head_dim = self.embed_dim // self.head_num
    assert self.head_dim * self.head_num == self.embed_dim, "The Embedding Dimension Should be divisable by number of heads"


    self.in_proj_weight = nn.Parameter(torch.empty(3 * self.embed_dim, self.embed_dim))
    self.register_parameter('q_proj_weight', None)
    self.register_parameter('k_proj_weight', None)
    self.register_parameter('v_proj_weight', None)

    self.in_proj_bias = nn.Parameter(torch.empty(3 * self.embed_dim)) ## 거의 아무 의미 없는 값들로 parameter을 채워주기 때문에
    self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    self.bias_k = nn.Parameter(torch.empty(1, 1, self.embed_dim))
    self.bias_v = nn.Parameter(torch.empty(1, 1, self.embed_dim))
  

  def forward(self, query, key, value):
    """ Args (근데 이 경우에는 target sequence length == source sequence length이다.)
    query: (L, N, E) = (target_sequence_length, batch_size, embed_dim)
    key: (S, N, E) = (source_sequence_length, batch_size, embed_dim)
    value: (S, N, E) = (source_sequence_length, batch_size, embed_dim)

    Outputs
    attention_output: (L, N, E) = (target_sequence_length, batch_size, embed_dim)
    attention_weight: (N, L, S) = (batch_size, target_sequence_length, source_sequence_length)
    """
    target_seq_length, batch_size, embed_dim = query.shape
    scaling = float(self.head_dim) ** -0.5
    out = F.linear(query, self.in_proj_weight, self.in_proj_bias)
    q, k, v = torch.tensor_split(out,3,dim = -1)
    q *= scaling
    k = torch.cat([k, self.bias_k.repeat(1, batch_size, 1)])
    v = torch.cat([v, self.bias_v.repeat(1, batch_size, 1)])
    q = q.contiguous().view(target_seq_length, -1, self.head_dim).transpose(0, 1)
    
    k = k.contiguous().view(-1, batch_size * self.head_num, self.head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, batch_size * self.head_num, self.head_dim).transpose(0, 1)

    attention_weight = torch.bmm(q, k.transpose(1, 2))
    attention_weight = F.softmax(attention_weight, dim = -1)

    attention_output = torch.bmm(attention_weight, v).transpose(0, 1).contiguous().view(-1, batch_size, embed_dim)
    attention_output = self.out_proj(attention_output)

    return attention_output, attention_weight.sum(dim = 1)/self.head_num


#### Multi Head Attention for Key Dimension & Value Dimension ####
class MultiHeadAttention_Diff(nn.Module):
  def __init__(self,
               embed_dim,
               head_num,
               kdim,
               vdim=None):
    super(MultiHeadAttention_Diff, self).__init__()
    """ Args
    embed_dim: total dimension of the model
    kdim: total numbers of features in key
    vdim: total numbers of features in the value
    """
    self.embed_dim = embed_dim
    self.head_num = head_num
    self.kdim = kdim
    self.vdim = vdim if vdim is not None else self.embed_dim
    self.head_dim = int(self.embed_dim / self.head_num)

    self.q_proj_weight = nn.Parameter(torch.Tensor(self.embed_dim, self.embed_dim))
    self.k_proj_weight = nn.Parameter(torch.Tensor(self.embed_dim, self.kdim))
    self.v_proj_weight = nn.Parameter(torch.Tensor(self.embed_dim, self.vdim))

    self.in_proj_bias = nn.Parameter(torch.empty(3 * self.embed_dim))


    self.bias_k = nn.Parameter(torch.empty(1, 1, self.embed_dim))
    self.bias_v = nn.Parameter(torch.empty(1, 1, self.embed_dim))

    self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
    
  
  def forward(self, query, key, value):
    target_seq_length, batch_size, embed_dim = query.shape
    scaling = float(self.head_dim) ** -0.5
    query = F.linear(query, self.q_proj_weight, self.in_proj_bias[0:self.embed_dim])
    key = F.linear(key, self.k_proj_weight, self.in_proj_bias[self.embed_dim: self.embed_dim * 2])
    value = F.linear(value, self.v_proj_weight, self.in_proj_bias[self.embed_dim*2:])

    query *= scaling

    k = torch.cat([key, self.bias_k.repeat(1, batch_size, 1)])
    v = torch.cat([value, self.bias_v.repeat(1, batch_size, 1)])
    """ Why use contiguous()?
    matrix를 reshape하는 등의 과정에서 이미지는 어쩔 수 없이 형태가 망가질 수 있다.
    그렇기 때문에 순서의 일관성을 유지하기 위해서는 torch.contiguous()를 사용해야 한다.
    """
    q = query.contiguous().view(target_seq_length, batch_size * self.head_num, self.head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, batch_size * self.head_num, self.head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, batch_size * self.head_num, self.head_dim).transpose(0, 1)

    attention_output_weights = torch.bmm(q, k.transpose(1, 2))
    attention_output_weights = F.softmax(attention_output_weights, dim = -1)
    attention_output = torch.bmm(attention_output_weights, v)
    attention_output = attention_output.transpose(0, 1).contiguous().view(target_seq_length, batch_size, embed_dim)
    attention_output = self.out_proj(attention_output)

    return attention_output, attention_output_weights.sum(dim = 1) / self.head_num



      
