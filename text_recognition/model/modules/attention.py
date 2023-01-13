import torch
import torch.nn as nn
import torch.nn.functional as F

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


    #self.in_proj_weight = nn.Parameter(torch.empty(3 * self.embed_dim, self.embed_dim))
    self.register_parameter('q_proj_weight', None)
    self.register_parameter('k_proj_weight', None)
    self.register_parameter('v_proj_weight', None)

    #self.in_proj_bias = nn.Parameter(torch.empty(3 * self.embed_dim)) ## 거의 아무 의미 없는 값들로 parameter을 채워주기 때문에
    self.in_proj = nn.Linear(self.embed_dim, self.embed_dim*3, bias=True) ## QKV Projection
    self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    #self.bias_k = nn.Parameter(torch.empty(1, 1, self.embed_dim))
    #self.bias_v = nn.Parameter(torch.empty(1, 1, self.embed_dim))
  

  def forward(self, query, key, value):
    """ Args (근데 이 경우에는 target sequence length == source sequence length이다.)
    query: (L, N, E) = (target_sequence_length, batch_size, embed_dim)
    key: (S, N, E) = (source_sequence_length, batch_size, embed_dim)
    value: (S, N, E) = (source_sequence_length, batch_size, embed_dim)

    Outputs
    attention_output: (L, N, E) = (target_sequence_length, batch_size, embed_dim)
    attention_weight: (N, L, S) = (batch_size, target_sequence_length, source_sequence_length)
    """
    target_seq_length, batch_size, embed_dim = query.shape ## 256, batch_size, 512
    scaling = float(self.head_dim) ** -0.5 ## multi head self attention에서 필요한 부분
    out = self.in_proj(query)
    out = out.reshape(batch_size, target_seq_length, self.head_num, self.head_dim*3).permute(0, 2, 1, 3) 
    # out = F.linear(query, self.in_proj_weight, self.in_proj_bias)
    q, k, v = torch.tensor_split(out,3,dim = -1)
    q *= scaling

    attention_weight = torch.matmul(q, k.transpose(-2, -1))
    attention_weight = F.softmax(attention_weight, dim = -1)

    ## 아래의 attention output에 대한 reshape연산으로 multi-head output을 concatenate하는 것과 동일한 효과가 나게 된다.
    attention_output = torch.matmul(attention_weight, v).permute(0, 2, 1, 3).reshape(batch_size, target_seq_length, self.embed_dim)
    # print(attention_output.shape)
    attention_output = self.out_proj(attention_output)

    return attention_output.transpose(1, 0), attention_weight.sum(dim = 1)/self.head_num