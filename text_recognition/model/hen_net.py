import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from encoder import ResTransformer
from decoder import AttentionalDecoder

""" Hangul Net
1. ResModel (ResNet45)
2. Transformer Encoder
3. Position Attention based Decoder
4. Linear Classifier (Generates the class of the hangul graphemes)
"""
class HENNet(nn.Module):
  def __init__(self, 
               max_seq_length=75, ## 논문의 저자들이 지정한 가장 긴 길이의 sequence length인데, 75라는 것은 총 문자의 개수가 25개라는 것이다.
               embedding_dim=512, ## transformer encoder에서 model output의 dimension을 나타냄
               class_n=52, ## 한글에서의 초성-중성-종성의 개수를 나타냄
               ):
    super(HENNet, self).__init__()
    #self.resnet = resnet45()
    self.transformer_encoder = ResTransformer() # 이 안에 ResNet45가 있음
    self.attention_decoder = AttentionalDecoder() # 초성-중성-종성의 각각에 순서대로 번호를 부여하여 위치 임베딩 벡터륾 만들어줌
    self.cls = nn.Linear(embedding_dim, class_n) # (N, T, Embed Dim) -> (N, T, Grapheme Class #)

  
  def forward(self, x):
    #feature = self.resnet(x)
    #logger.info(feature.shape)
    encoder_out = self.transformer_encoder(x)
    att_vec, att_score = self.attention_decoder(encoder_out)

    pred = self.cls(att_vec) # (Batch #, Seq Length, Grapheme Class #)

    return pred
