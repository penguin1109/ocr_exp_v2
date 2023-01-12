import torch
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger
import datetime
TODAY=datetime.datetime.now()
TODAY=TODAY.strftime('%Y-%m-%d')
def Accuracy_metric(prediction, label):
  ## label이 one-hot으로 주어져 있음
  converter = HangulLabelConverter(add_num=True, add_eng=True)

  def change(input, pred=False):
    pred_text = []
    if pred:
      input = F.softmax(input, dim=2)
      for score in input:
        score_ = score.argmax(dim=1)
        text = ''
        for idx, s in enumerate(score_):
          temp = converter.char_decoder_dict[s.item()]
          if temp == converter.null_char:
            break ## <null> char이 나오면 이제 끝났다는 뜻
          if temp == converter.unknown_char:
            text += '' ## 예측 불가능한건 공백
          else:
            text += temp
        pred_text.append(text)
    else:
      input = torch.argmax(input, dim=2)
      for score in input:
        text = ''
        for idx, s in enumerate(score):
          temp = converter.char_decoder_dict[s.item()]
          if temp == converter.null_char:
            break ## <null> char이 나오면 이제 끝났다는 뜻
          if temp == converter.unknown_char:
            text += '' ## 예측 불가능한건 공백
          else:
            text += temp
        pred_text.append(text)
    return pred_text
  
  prediction = change(prediction, True)
  label = change(label, False)
  acc_list = [(pred==targ) for pred, targ in zip(prediction, label)]
  accuracy = 1.0 * sum(acc_list) / len(acc_list)

  return accuracy

class Trainer(object):
  def __init__(self, config):
    super(Trainer, self).__init__()
    self.config = config
  
    train_dataset = HENNetDataset(config.BASE_DIR, aug=False, mode='train')
    valid_dataset = HENNetDataset(config.BASE_DIR, aug=False, mode='valid')
    self.train_dataloader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True)
    self.valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    self.model = HENNet(class_n = len(train_dataset.converter.characters)).cuda()
    self._load()
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.config.learning_rate)
    self.criterion = FocalLoss()

    self.best_metric = 0

  def _load(self):
    if self.config.pretrained_model != '':
      self.model.load_state_dict(torch.load(self.config.pretrained_model))
    
  def train(self):
    logger.info("=====START TRAINING=====")
    for epoch in range(self.config.epoch):
      running_loss = 0.0
      if epoch == 6:
        for g in self.optimizer.param_groups:
          g['lr'] = 1e-5
      self.model.train()
      loop = tqdm(self.train_dataloader)
      for idx, batch in enumerate(loop):
        self.optimizer.zero_grad()
        image, label, text, length = batch
        image = image.cuda()
        prediction = self.model(image)
        loss = self.criterion(prediction, label.cuda(), length)
        loss.backward()
        self.optimizer.step()
        loop.set_postfix({"Loss": loss.item()})
        running_loss += loss.item()
        
      logger.info(f"Loss of epoch {epoch}: {running_loss / len(loop)}")
      
      if epoch % (self.config.valid_epoch) == 1:
        loop = tqdm(self.valid_dataloader)
        with torch.no_grad():
          running_acc = 0.0
          for idx, batch in enumerate(loop):
            image, label, text, length = batch
            image = image.cuda()
            prediction = self.model(image)
            acc = Accuracy_metric(prediction, label.cuda())
            loop.set_postfix({"ACC": acc})
            running_acc += acc
        logger.info(f"Accuracy is : {running_acc / len(loop)}")
        if self.best_metric < running_acc:
          self.best_metric = running_acc
          torch.save(model.state_dict(), self.config.work_dir + f"/{TODAY}_best.pth")

      torch.save(model.state_dict(),self.config.work_dir + f"{TODAY}.pth")
    logger.info("======END TRAINING======")