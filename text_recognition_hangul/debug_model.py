def get_attention_map(att_map):
    att_mat = torch.stack(att_map).squeeze(0)
    att_mat = torch.mean(att_mat, dim=0)
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
  def __init__(self, ch_in, ch_mid, model_dim):
    super(SimpleCNN, self).__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(ch_in, ch_mid, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(ch_mid), nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(ch_mid, model_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(model_dim), nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
  def forward(self, x):
    x1 = self.conv1(x)
    x2 = self.conv2(x1)
    return x1, x2

if __name__ == "__main__":
  import yaml
  import numpy as np
  import os, json, sys, cv2
  import torch
  import torch.nn.functional as F
  from torch.utils.data import DataLoader
  from model.hen_net import HENNet
  import matplotlib.pyplot as plt
  from dataset import HENDatasetOutdoor, HENDatasetV2
  from einops import rearrange

  DEVICE = torch.device("cuda:6") if torch.cuda.is_available() else torch.device('cpu')
  with open('/home/guest/ocr_exp_v2/text_recognition_hangul/configs/printed_data.yaml', 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
  dataset = HENDatasetV2(mode='train', DATA_CFG=cfg['DATA_CFG'])
  #dataset= HENDatasetOutdoor(mode='train', DATA_CFG=cfg['DATA_CFG'])
  label_converter = dataset.label_converter;
  model_cfg = cfg['MODEL_CFG']
  train_cfg = cfg['TRAIN_CFG']
  data_cfg = cfg['DATA_CFG']
  model = HENNet(
      img_w=model_cfg['IMG_W'], img_h=model_cfg['IMG_H'], res_in=model_cfg['RES_IN'],
      encoder_layer_num=model_cfg['ENCODER_LAYER_NUM'], 
      activation=model_cfg['ACTIVATION'],
      adaptive_pe=model_cfg['ADAPTIVE_PE'],
      seperable_ffn=model_cfg['SEPERABLE_FFN'],
      use_resnet=model_cfg['USE_RES'],
      batch_size=model_cfg['BATCH_SIZE'],
      rgb=model_cfg['RGB'],
      head_num=model_cfg['HEAD_NUM'],
      max_seq_length=model_cfg['MAX_SEQ_LENGTH'],
      embedding_dim=model_cfg['EMBEDDING_DIM'],
      class_n=len(label_converter.characters)).to(DEVICE)
  #model.load_state_dict(torch.load(
  #  '/home/guest/ocr_exp_v2/weight/convFFN_layer6_relu_3e_5_REARRANGE/2023-01-28_min_loss.pth'))
  print("===All Keys Matched Sucessfullly===")
  # print(dataset.label_converter.char_with_no_tokens)
  loader = DataLoader(dataset, batch_size=30)
  MEAN=[0.5,0.5,0.5];STD=[0.5,0.5,0.5];
  NUMS = len(loader)
  from tqdm import tqdm
  loop = tqdm(loader)
  IMAGE=[]
  
  with open('debug.txt', 'w') as f:
    for name, param in model.named_parameters():
      if param.requires_grad:
        f.writelines("-"*80 + "\n")
        f.writelines(f"{name}  |  {param} \n")
        f.writelines("-"*80 + '\n')

  for idx, batch in enumerate(loop):
    image, label, text,_ = batch
    IMAGE.append(image.detach().cpu().numpy())
    # label = torch.argmax(label, dim=-1)
    label = torch.stack([
        torch.argmax(F.softmax(x, dim=-1), dim=-1) for x in label
    ], dim=0)
    print(label)
    model.eval()
    pred = model(image.to(DEVICE), batch_size=1, mode='train')
    pred_text, pred_scores, pred_lengths = label_converter.decode(pred)
    print('-' * 80)
    #print(pred_text)
    #print(text)
    for b in range(image.shape[0]):
      print(f'PREDICTION : {pred_text[b]} TARGET : {text[b]}')# SCORE : {pred_scores[b]}')
    break
  DEST_DIR=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug')
  os.makedirs(DEST_DIR,exist_ok=True)
 
  res_out = model.transformer_encoder.resnet(image.to(DEVICE))
  # print(model.transformer_encoder.resnet)
  cnn = SimpleCNN(1, 64, 512).to(DEVICE)
  # print(cnn)
  cnn_out = cnn(image.to(DEVICE))

  out = model.transformer_encoder.pos_encoder(res_out, batch_size=30)
  if isinstance(out, tuple):
    _, x = out
  else:x=out
  encoder_output, attn_mask = model.transformer_encoder(image.to(DEVICE), batch_size=1)
  
  s, n, c = x.shape
  x = rearrange(x, '(h w) n c -> h w n c', h=8,w=48)
  x = rearrange(x, 'h w n c -> n c h w')
  plt.imsave(os.path.join(DEST_DIR, 'cnn_out.png'),cnn_out[1].detach().cpu().numpy()[10,100], cmap='gray')
  plt.imsave(os.path.join(DEST_DIR, 'res_out.png'), res_out.detach().cpu().numpy()[10, 100], cmap='gray')
  plt.imsave(os.path.join(DEST_DIR, 'pe_out.png'), x.detach().cpu().numpy()[10, 100], cmap='gray' )
  plt.imsave(os.path.join(DEST_DIR, 'att_out.png'), encoder_output.detach().cpu().numpy()[0,100], cmap='gray')


  #for l in range(sample.shape[0]):
  #  place = sample[l, :,:].detach().cpu().numpy() 
  #  cv2.imwrite(os.path.join(DEST_DIR, f"{l}.png"), place * 255)
  target_img = image[10].detach().permute(1, 2, 0).cpu().numpy()
  target_img  = target_img # * data_cfg['MEAN'] + data_cfg['STD']
  cv2.imwrite(os.path.join(DEST_DIR, "target.png"), target_img * 255)


  IMAGE = np.concatenate(IMAGE, axis=0)
  """
  mean_ = np.array([np.mean(x, axis=(1,2)) for x in IMAGE])
  mean_r=mean_[:,0].mean()
  mean_g=mean_[:,1].mean()
  mean_b=mean_[:,2].mean()

  std_=np.array([np.std(x, axis=(1,2)) for x in IMAGE])
  std_r=std_[:,0].mean()
  std_g=std_[:,1].mean()
  std_b=std_[:,2].mean()

  print(f"{mean_r}  {mean_g}   {mean_b}")
  print(f"{std_r}    {std_g}    {std_b}")



   # print(image[0].max())
   #  print(image[0].min())
    # print(text)

  """ 