from tqdm import tqdm
import torch
USE_CUDA=torch.cuda.is_available()
DEVICE=torch.device('cuda:6' if USE_CUDA else 'cpu')
from loss import FocalLoss, SoftCrossEntropyLoss

def train_one_epoch(model, train_dataloader, optimizer):
  loop=tqdm(train_dataloader)
  model.train()
  torch.set_grad_enabled(True)
  criterion = SoftCrossEntropyLoss()
  outs = []

  for idx, batch in enumerate(loop):
    image, label, text = batch
    pred = model(image.to(DEVICE))
    loss = criterion(pred, label.to(DEVICE))
    # loss = F.cross_entropy(pred, label.to(DEVICE)) ## loss function은 그냥 우선은 cross entropy 사용
  
    outs.append(loss)

    optimizer.zero_grad() # clear gradients
    loss.backward() ## backward pass
    nn.utils.clip_grad_norm_(model.parameters(), 5.0) ## gradient clipping to make the model training converge(방향은 유지하고 gradient의 크기 제한한)
    optimizer.step() ## update parameters
    label.detach();image.detach();pred.detach();

    loop.set_postfix({"loss": loss.detach()})
  epoch_metric = torch.mean(torch.stack([x for x in outs]))

  return epoch_metric, model

def test_one_epoch(model, test_dataloader, converter):
  loop = tqdm(test_dataloader)
  torch.set_grad_enabled(False)
  criterion = SoftCrossEntropyLoss()
  model.eval()

  preds = []
  correct = 0
  targets = []
  for idx, batch in enumerate(loop):
    image, label, text =batch
    pred = model(image.to(DEVICE))
    # loss = F.cross_entropy(pred, label.to(DEVICE))
    loss = criterion(pred, label.to(DEVICE))
    loop.set_postfix({"loss": loss.detach()})
    preds.append(converter.decode(pred))
    targets.append(text)
    pred.detach();image.detach();label.detach();
  
  for pred, gt in zip(preds, targets):
    if pred == gt:
      correct += 1
  
  accuracy = (correct / len(test_dataloader)) * 100

  return accuracy
  
  


import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import yaml, os
import numpy as np
from torch.utils.data import DataLoader
# import pytorch_lightning as pl
from loguru import logger
import datetime
TODAY=datetime.datetime.now()
TODAY=TODAY.strftime('%Y-%m-%d')
CONFIG_DIR='/home/guest/ocr_exp_v2/text_recognition_hangul/configs'
from dataset import HENDataset, HENDatasetV2, HENDatasetOutdoor
from model.hen_net import HENNet

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str)
  args = parser.parse_args()
  config_name=args.config
  config_dir = os.path.join(CONFIG_DIR, config_name)
  with open(config_dir, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
  data_cfg = cfg['DATA_CFG']
  train_cfg = cfg['TRAIN_CFG']
  model_cfg = cfg['MODEL_CFG']

  if 'Outdoor' in data_cfg['DATASET']:
    train_dataset = HENDatasetOutdoor(mode='train', DATA_CFG=data_cfg)
    test_dataset = HENDatasetOutdoor(mode='test', DATA_CFG=data_cfg)
  elif 'V2' in data_cfg['DATASET']:
    train_dataset = HENDatasetV2(mode='train', DATA_CFG=data_cfg)
    test_dataset = HENDatasetV2(mode='test', DATA_CFG=data_cfg)
  else:
    train_dataset = HENDataset(mode='train', DATA_CFG=data_cfg)
    test_dataset = HENDataset(mode='test', DATA_CFG=data_cfg)

  train_loader = DataLoader(train_dataset, batch_size=train_cfg['BATCH_SIZE'], shuffle=True, drop_last=True)
  test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

  label_converter = train_dataset.label_converter

  model = HENNet(
      img_w=model_cfg['IMG_W'], img_h=model_cfg['IMG_H'], res_in=model_cfg['RES_IN'],
      max_seq_length=model_cfg['MAX_SEQ_LENGTH'],
      embedding_dim=model_cfg['EMBEDDING_DIM'],
      class_n=len(label_converter.characters)) 
  
  if model_cfg['PRETRAINED'] != '':
    model.load_state_dict(torch.load(model_cfg['PRETRAINED']))
  
  model.to(DEVICE)
  optimizer = torch.optim.Adam(model.parameters(), lr = train_cfg['LR'])
  #optimizer = torch.optim.Adagrad(model.parameters(), lr=1.0) ## Adagrad는 알아서 adaptive learning rate를 찾아가기 때문에 처음 learning rate는 1이어야 한다.
  min_loss=100
  max_acc = 0.0
  logger.info("==== START TRAINING ====")
  for epoch in range(train_cfg['EPOCH']):
    epoch_loss, model = train_one_epoch(model, train_loader, optimizer)
    if epoch_loss < min_loss:
      min_loss = epoch_loss
      torch.save(model.state_dict(), os.path.join(train_cfg['WEIGHT_FOLDER'], f"{TODAY}_min_loss.pth"))
    if (epoch+1) % train_cfg['EVAL_EPOCH'] == 0:
      logger.info("=== START EVALUATION ===")
      accuracy = test_one_epoch(model, test_loader, label_converter)
      if max_acc < accuracy:
        mac_acc = accuracy
        torch.save(model.state_dict(), os.path.join(train_cfg['WEIGHT_FOLDER'], f"{TODAY}_best.pth"))
    torch.save(model.state_dict(), os.path.join(train_cfg['WEIGHT_FOLDER'], f"{TODAY}_epoch{epoch}.pth"))
    






