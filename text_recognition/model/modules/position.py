import torch
import torch.nn as nn
import torch.nn.functional as F

"""Postitional Encoding (L, C)
- The positional encoding module outputs the input feature map tensor added pixel wise with the position encoded vector
- In the paper, the max_length is set to the 75. (Not written on the paper, but is told by the author of the paper)
- In the paper, the embedding dimension is not written.
"""

class PositionEncoding(nn.Module):
  def __init__(self, 
              max_length=75, # +1, ## Additional Stop Token 추가
              embedding_dim=512,
              dropout_rate=0.1,
              device=torch.device('cuda')):
    super(PositionEncoding, self).__init__()
    """sin, cos encoding 구현
    max_length: 전체 단어 / 문장의 최대 길이 (단, Hangul Net에서는 3 X 단어의 수이다.)
    embedding_dim: Dimension of the model
    """
    self.dropout = nn.Dropout(dropout_rate)
    encoding = torch.zeros(max_length, embedding_dim, device = device)
    encoding.requires_grad = False
    pos = torch.arange(0, max_length, device = device)
    pos = pos.float().unsqueeze(dim = 1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)).to(device)
    # _2i = torch.arange(0, embedding_dim, step = 2, device = device).float()

    encoding[:, ::2] = torch.sin(pos * div_term) # torch.sin(pos / (1000 ** (_2i / embedding_dim)))
    encoding[:, 1::2] = torch.cos(pos*div_term) # torch.cos(pos / (1000 ** (_2i / embedding_dim)))
    encoding = encoding.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', encoding)
    
  
  def forward(self, x):
    """ Args
    별건 아니고 1,2,3.. 순서대로 알아서 위치 정보에 대한 embedded vector을 입력 sequence에 더해 주면 결과를 모델이 알아서 학습을 하게 될 것이다.
    x: (sequence_length, batch_size, embedding_dimension)
    out: (sequence_length, batch_size, embedding_dimension)
    """
    seq_len, batch_size, embed_dim = x.shape
    x = x + self.pe[:seq_len, :]

    return self.dropout(x)
  

