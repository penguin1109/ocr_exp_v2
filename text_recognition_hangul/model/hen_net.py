import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from encoder import ResTransformer
from decoder import AttentionalDecoder
DEVICE=torch.device("cuda:6") if torch.cuda.is_available() else torch.device("cpu")
""" Hangul Net
1. ResModel (ResNet45)
2. Transformer Encoder
3. Position Attention based Decoder
4. Linear Classifier (Generates the class of the hangul graphemes)
"""
class HENNet(nn.Module):
  def __init__(self, 
              img_w, img_h, res_in, 
               max_seq_length=75, ## 논문의 저자들이 지정한 가장 긴 길이의 sequence length인데, 75라는 것은 총 문자의 개수가 25개라는 것이다.
               embedding_dim=512, ## transformer encoder에서 model output의 dimension을 나타냄
               class_n=52, ## 한글에서의 초성-중성-종성의 개수를 나타냄
               ):
    super(HENNet, self).__init__()
    #self.resnet = resnet45()
    self.transformer_encoder = ResTransformer(
      img_w=img_w, img_h=img_h, res_in=res_in, device=torch.device('cuda:6'),
      model_dim=embedding_dim) # 이 안에 ResNet45가 있음
    self.attention_decoder = AttentionalDecoder( 
              img_h=img_h, img_w=img_w,
               in_channel=embedding_dim,
               unet_channel=64,
               max_seq_length=max_seq_length,
               embedding_dim=embedding_dim) 
    self.cls = nn.Linear(embedding_dim, class_n) # (N, T, Embed Dim) -> (N, T, Grapheme Class #)

  
  def forward(self, x):
    #feature = self.resnet(x)
    #logger.info(feature.shape)
    encoder_out = self.transformer_encoder(x)
    att_vec, att_score = self.attention_decoder(encoder_out)

    pred = self.cls(att_vec) # (Batch #, Seq Length, Grapheme Class #)

    return pred

if __name__ == "__main__":
  net = HENNet(img_w=128, img_h=32, res_in=32, max_seq_length=30, embedding_dim=512, class_n=54).to(DEVICE)
  sample = torch.rand((1, 3, 32,128)).to(DEVICE)

  out = net(sample)
  print(out.shape) ## (Batch Size, Max Sequence Length, Class Number)