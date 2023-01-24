from tqdm import tqdm
import torch
USE_CUDA=torch.cuda.is_available()
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
DEVICE=torch.device('cuda:6' if USE_CUDA else 'cpu')
from loss import FocalLoss, SoftCrossEntropyLoss

from torch.nn import CTCLoss, CrossEntropyLoss
from torch.optim.lr_scheduler import CyclicLR
from scheduler import CosineAnnealingWarmUpRestarts

def train_one_epoch(model, train_dataloader, optimizer,  train_cfg):
  loop=tqdm(train_dataloader)
  model.train()
  torch.set_grad_enabled(True)
  if train_cfg['LOSS_FN'] == 'CTC': 
    ## 근데 이 loss function은 RNN과 같은 모듈이 포함되어 있을때 (예측 부분으로) 사용하는 것이 맞다.
    criterion = CTCLoss()
  elif train_cfg['LOSS_FN'] == 'SOFTCE':
    criterion = SoftCrossEntropyLoss()
  elif train_cfg['LOSS_FN'] == 'CE':
    criterion = CrossEntropyLoss()
  elif train_cfg['LOSS_FN'] == 'FOCAL':
    criterion = FocalLoss()
  outs = []

  for idx, batch in enumerate(loop):
    image, label, text = batch
    pred = model(image.to(DEVICE)) ## [B, L, C]
    if train_cfg['LOSS_FN'] == "CTC":
      pred = F.log_softmax(pred, dim=2).permute(1,0,2)
      label = torch.argmax(label, dim=1)
      B = pred.shape[1];T = pred.shape[0]
      input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long)
      target_lengths = torch.randint(low=1, high=pred.shape[2], size=(B,), dtype=torch.long)
      loss = criterion(pred, label.to(DEVICE),input_lengths, target_lengths)
    else:
      loss = criterion(pred, label.to(DEVICE))
    # loss = F.cross_entropy(pred, label.to(DEVICE)) ## loss function은 그냥 우선은 cross entropy 사용
  
    outs.append(loss)

    optimizer.zero_grad() # clear gradients
    loss.backward() ## backward pass
    nn.utils.clip_grad_norm_(model.parameters(), 20.0) ## gradient clipping to make the model training converge(방향은 유지하고 gradient의 크기 제한한)
    optimizer.step() ## update parameters
    #label.detach();image.detach();pred.detach();

    loop.set_postfix({"loss": loss.detach()})
  epoch_metric = torch.mean(torch.stack([x for x in outs]))

  return epoch_metric, model, optimizer

def test_one_epoch(model, test_dataloader, converter):
  loop = tqdm(test_dataloader)
  torch.set_grad_enabled(False)
  criterion = CrossEntropyLoss()
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
    #pred.detach();image.detach();label.detach();
  
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
    debug_dataset = HENDatasetOutdoor(mode='debug', DATA_CFG=data_cfg)
  elif 'V2' in data_cfg['DATASET']:
    train_dataset = HENDatasetV2(mode='train', DATA_CFG=data_cfg)
    test_dataset = HENDatasetV2(mode='test', DATA_CFG=data_cfg)
    debug_dataset = HENDatasetV2(mode='debug', DATA_CFG=data_cfg)
  else:
    train_dataset = HENDataset(mode='train', DATA_CFG=data_cfg)
    test_dataset = HENDataset(mode='test', DATA_CFG=data_cfg)
    debug_dataset = HENDataset(mode='debug', DATA_CFG=data_cfg)

  train_loader = DataLoader(train_dataset, batch_size=train_cfg['BATCH_SIZE'], shuffle=True, drop_last=True)
  test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
  debug_loader = DataLoader(debug_dataset, batch_size = 30, shuffle=True)

  label_converter = train_dataset.label_converter

  model = HENNet(
      img_w=model_cfg['IMG_W'], img_h=model_cfg['IMG_H'], res_in=model_cfg['RES_IN'],
      head_num=model_cfg['HEAD_NUM'],
      max_seq_length=model_cfg['MAX_SEQ_LENGTH'],
      embedding_dim=model_cfg['EMBEDDING_DIM'],
      class_n=len(label_converter.characters)) 
  
  if model_cfg['PRETRAINED'] != '':
    model.load_state_dict(torch.load(model_cfg['PRETRAINED']))
  
  model.to(DEVICE)
  if train_cfg['OPTIMIZER'] == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr = train_cfg['LR'], momentum=train_cfg['MOMENTUM'])
  elif train_cfg['OPTIMIZER'] == 'ADAM':
    optimizer = torch.optim.Adam(model.parameters(), lr = train_cfg['LR'], betas=[0.9, 0.999])
  else:
    raise RuntimeError
  if model_cfg['PRETRAINED_OPTIM'] != '':
    optimizer.load_state_dict(torch.load(model_cfg['PRETRAINED_OPTIM']))
  #optimizer = torch.optim.Adagrad(model.parameters(), lr=1.0) ## Adagrad는 알아서 adaptive learning rate를 찾아가기 때문에 처음 learning rate는 1이어야 한다.
  
  if train_cfg['SCHEDULER'] == 'CYCLIC':
    cycle_momentum=False if isinstance(optimizer, torch.optim.Adam) else True
    scheduler = CyclicLR(optimizer, base_lr = train_cfg['LR'], max_lr=0.01, \
          step_size_up=250000, step_size_down=None, mode='triangular2', \
          cycle_momentum=cycle_momentum)
  elif train_cfg['SCHEDULER'] == 'STEP':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
  else:
    raise RuntimeError

  max_acc = 0.0
  if train_cfg['DEBUG'] == True:
    logger.info("==== START DEBUGGING ====")
    for epoch in range(200):
      epoch_loss , model, optimizer = train_one_epoch(model, debug_loader, optimizer, train_cfg)
    scheduler.step()

  logger.info("==== START TRAINING ====")
  for epoch in range(train_cfg['EPOCH']):
    epoch_loss, model, optimizer, scheduler = train_one_epoch(model, train_loader, optimizer,  train_cfg)
    ## DEBUGGING TO CHECK IF THE MODEL IS NOT IN THE LOCAL MINIMA ##
    """
    for i, para in enumerate(model.parameters()):
      print(f'{i + 1}th parameter tensor:', para.shape)
      print(para)
      print(para.grad)
      break
    """
    if epoch_loss < min_loss:
      min_loss = epoch_loss
      torch.save(model.state_dict(), os.path.join(train_cfg['WEIGHT_FOLDER'], f"{TODAY}_min_loss.pth"))
    if (epoch+1) % train_cfg['EVAL_EPOCH'] == 0:
      logger.info("=== START EVALUATION ===")
      accuracy = test_one_epoch(model, test_loader, label_converter, train_cfg)
      if max_acc < accuracy:
        mac_acc = accuracy
        torch.save(model.state_dict(), os.path.join(train_cfg['WEIGHT_FOLDER'], f"{TODAY}_best.pth"))
    torch.save(model.state_dict(), os.path.join(train_cfg['WEIGHT_FOLDER'], f"{TODAY}_epoch{epoch}.pth"))
    torch.save(optimizer.state_dict(), os.path.join(train_cfg['OPTIM_FOLDER'], f"{TODAY}_epoch{epoch}.pth"))

    scheduler.step()
    logger.info(f"EPOCH: {epoch+1} LR: {scheduler.get_last_lr()} LOSS: {epoch_loss}")




