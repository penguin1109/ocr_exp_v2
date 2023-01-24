import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.position import PositionEncoding
from modules.transformer import TransformerEncoderLayer
from modules.resnet import resnet45

""" Transformer Encoder
- ResNet-50
- Multi-Head Attention
- Add&Norm (=Residual Connection & Layer Norm)
- Feed Forward
- Add&Norm
"""
class ResTransformer(nn.Module):
    def __init__(self, 
                img_w, img_h,res_in,device,
                 feedforward_dim=2048,
                 model_dim=512,
                 head_num=8,
                 dropout=0.1,
                 num_layers=5,):
        super(ResTransformer, self).__init__()
        self.resnet = resnet45(res_in)
        self.img_w = img_w
        self.img_h = img_h
        self.d_model = model_dim
        self.nhead = head_num
        self.inner_dim = feedforward_dim
        self.dropout = dropout
        self.activation = nn.Tanh() # nn.ReLU(inplace=True) # nn.Tanh() # nn.GELU() # nn.ReLU()
        self.num_layers = num_layers

        self.pos_encoder = PositionEncoding(embedding_dim=self.d_model, max_length=(img_w//4) * (img_h // 4), dropout_rate = 0.1, device =device)
        encoder_layer = TransformerEncoderLayer(model_dim=self.d_model, head_num=self.nhead, 
                dim_feedforward=self.inner_dim, dropout=self.dropout, activation=self.activation)
        self.transformer = nn.ModuleList([
            encoder_layer for _ in range(self.num_layers)
        ])


    def forward(self, images):
        feature = self.resnet(images) ## (B, 512, 8, 32)
        n, c, h, w = feature.shape
        feature = feature.view(n, c, -1).permute(2, 0, 1) ## (8*32, B, 512)
     
        feature = self.pos_encoder(feature)

        for idx, layer in enumerate(self.transformer):
          feature = layer(feature)

        feature = feature.permute(1, 2, 0).view(n, c, h, w)

        return feature