import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from attention import MultiHeadAttention
from multi_head_attention import MultiheadAttention

class TransformerEncoderLayer(nn.Module):
  def __init__(self, 
               model_dim=512,
               head_num=8,
               dim_feedforward=2048,
               dropout=0.0,
               activation=nn.Tanh()):
    super(TransformerEncoderLayer, self).__init__()
    """ Args
    model_dim: dimension of the model
    head_num: number if heads in the multi head attention layer
    dim_feedforward: middle dimension in the feed forward network
    """
    ## (1) Multi Head Attention
    # self.self_attn = MultiHeadAttention(model_dim, head_num, dropout)
    self.self_attn = MultiheadAttention(
      embed_dim=model_dim, num_heads=head_num, dropout=0., bias=True, add_bias_kv=False,
      add_zero_attn=False, kdim=None, vdim=None, 
    )
    ## (2) Feed Forward Network
    self.linear1 = nn.Linear(model_dim, dim_feedforward)
    self.dropout = nn.Dropout(dropout)
    self.activation = activation
    self.linear2 = nn.Linear(dim_feedforward, model_dim)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

    ## (3) Add & Norm
    self.norm1 = nn.LayerNorm(model_dim)
    self.norm2 = nn.LayerNorm(model_dim)
    
  
  def forward(self, x):
    """ Args
    x: input feature map from the ResNet-45
    """
    attn, attn_weight = self.self_attn(x,x,x)
    x = x + self.dropout1(attn)
    x = self.norm1(x)
    attn = self.linear2(self.dropout(self.activation(self.linear1(x))))
    x = x + self.dropout2(attn)
    x = self.norm2(x)

    return x