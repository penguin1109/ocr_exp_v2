import torch
import torch.nn as nn
import torch.nn.functional as F

def to_contiguous(tensor):
  if tensor.is_contiguous():
    return tensor
  else:
    return tensor.contiguous()

class FocalLoss(nn.Module):
  def __init__(self, max_length=75, alpha=0.99, gamma=1, use_focal=False,
               size_average=True, sequence_normalize=False, sample_normalize=True):
    super(FocalLoss, self).__init__()
    self.alpha = alpha
    self.ignore_index=0
    self.gamma = gamma
    self.use_focal = use_focal
    self.size_average = size_average
    self.sequence_normalize = sequence_normalize
    self.sample_normalize = sample_normalize

  def forward(self, prediction, ground_truth, length):
    batch_size, max_length = prediction.size(0), prediction.size(1)
    ground_truth = ground_truth[:, :max_length]
    
    mask = torch.zeros(batch_size, max_length)
    for i in range(batch_size):
      mask[i, :length[i]].fill_(1)
    mask = mask.type_as(prediction)[:, :max_length]
    
    prediction = to_contiguous(prediction).view(-1, prediction.size(2))
    prediction = F.log_softmax(prediction, dim=1)
    ground_truth = torch.argmax(ground_truth, dim=2)
    ground_truth = to_contiguous(ground_truth).view(-1, 1) ## one_hot

    mask = to_contiguous(mask).view(-1, 1)

    loss = -prediction.gather(1, ground_truth.long()) * mask

    if self.use_focal:
      p = torch.exp(-loss)
      loss = self.alpha * ((1-p) ** self.gamma) * loss
      if self.size_average:
        loss = torch.sum(loss) / torch.sum(mask)
      else:
        loss = torch.sum(loss)

    else:
      loss = torch.sum(loss)
      if self.sequence_normalize:
        loss = loss / torch.sum(mask)
      if self.sample_normalize:
        loss = loss / batch_size
    
    return loss
    


class SoftCrossEntropyLoss(nn.Module):
  def __init__(self, reduction="mean"):
    super().__init__()
    self.reduction = reduction
  
  def forward(self, input, target, softmax=True):
    if softmax:
      log_prob = F.log_softmax(input, dim=1)
    else:log_prob = torch.log(input)
    loss = -(target*log_prob).sum(dim=-1) ## 이걸 사용하는 경우에는 ground truth가 One-Hot Encoding되어 있음
    if self.reduction == 'mean':
      return loss.mean()
    elif self.reduction == 'sum':
      return loss.sum()
    else:return loss
