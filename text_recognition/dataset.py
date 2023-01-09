from matplotlib.figure import Text
from torch.utils.data import Dataset, DataLoader
import os
import zipfile
import re
import random
import io
from PIL import Image
import json
from torchvision import transforms
import numpy as np
from .transform import *
np.random.seed(1004)
import math
## trainset1은 medicine
## trainset2는 cosmetics

def catch_metadata(data: dict):
  img_path = data['src_path']
  label_path = data['label_path']
  bbox = data['annotations'][0]['polygons']
  text = []
  points = []
  for anno in bbox:
    text.append(anno['text'])
    X = np.array(anno['points']).T[0]
    Y = np.array(anno['points']).T[1]
    points.append([min(X), min(Y), max(X), max(Y)])
  return {
      "img_path": img_path,
      "label_path": label_path,
      "text": text,
      "points": points
  }

class HENNetDataset(Dataset):
  def __init__(self, BASE_DIR, mode='train', img_h=32, img_w = 128, aug = True):
    super(HENNetDataset, self).__init__()
    self.base_dir = BASE_DIR
    self.mode = mode
    self.img_files = []
    self.label_files = []
    self._get_file_path(os.path.join(BASE_DIR, 'train1.zip')) ## medicine
    self._get_file_path(os.path.join(BASE_DIR, 'train2.zip')) ## cosmetics
    self.img_archive = {
        "1": zipfile.ZipFile(os.path.join(BASE_DIR, 'train1.zip'), 'r'),
        "2": zipfile.ZipFile(os.path.join(BASE_DIR, 'train2.zip'), 'r')
    }
    self.valid_img_archive = {
        "image": zipfile.ZipFile(os.path.join(BASE_DIR, 'validation_image.zip')),
        "label": zipfile.ZipFile(os.path.join(BASE_DIR, 'validation_label.zip')),
        "label_dir": os.path.join(BASE_DIR, 'validation_label.zip')
    }

    self.converter = HangulLabelConverter(add_num=True, add_eng=True)
    self.aug = aug
    self.img_w, self.img_h = img_w, img_h


  def _get_file_path(self, dir):
    if 'train1' in dir:
      label_base = os.path.join(self.base_dir, 'Label1')
    else:
      label_base = os.path.join(self.base_dir, 'result') # 'Label2')

    names = sorted(zipfile.ZipFile(dir, 'r').namelist())
    EXTENSION = ['.jpg', '.jpeg']
    for name in names:
      
      if os.path.splitext(name)[-1].lower() in EXTENSION:## 이미지 파일인 경우에만
        self.img_files.append(os.path.join(dir, name))
        label_name ='/'.join(name.split('/')[1:]).replace('images', 'annotations')
        label_name = label_name.split('.')[0] + '.json'
        self.label_files.append(os.path.join(label_base, label_name))
        
  
  def _crop_bbox(self, image, points):
    """ Args
    :points: [minX, minY, maxX, maxY]
    :image: np.ndarray image
    """
    minX, minY, maxX, maxY = points
    minX, minY = math.floor(minX), math.floor(minY)
    maxX, maxY = math.ceil(maxX), math.ceil(maxY)
    crop = image.crop((minX, minY, maxX, maxY))
    return crop

  def _select_label(self, meta):
    labels = []
    for idx, text in enumerate(meta['text']):
      replace = re.sub('[^ sA-Za-z0-9가-힣,.()]', '', text)
      if replace == '': 
        continue
      label = self.converter.encode(text)
      length = len(self.converter.encode(text, padding=False))
      if label is not None:
        labels.append((label, idx, text, length))
    index = np.random.choice([int(i) for i in range(len(labels))])
    points = meta['points'][labels[index][1]]
    return points, labels[index][0], labels[index][-2], labels[index][-1]
    
  def __len__(self):
    if self.mode == 'valid':
      return 100
    return 2000 ## 한번에 4000개씩!

  def _resize(self, image):
    return cv2.resize(image, (self.img_w, self.img_h))

  def _aug_training(self, image):
    aug= transforms.Compose([
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
            ])
    if self.aug:
      image = aug(image)
    image = self._resize(np.array(image))

    return image

  def __getitem__(self, idx):
    if self.mode == 'valid':
      image_names = self.valid_img_archive["image"].namelist()[3:]
      label_names = self.valid_img_archive["label"].namelist()[3:]
      assert len(image_names) == len(label_names)
      idx = np.random.choice([int(i) for i in range(len(image_names))])
      image = Image.open(io.BytesIO(self.valid_img_archive["image"].read(image_names[idx])))
      with zipfile.ZipFile(self.valid_img_archive["label_dir"], "r") as z:
        with z.open(label_names[idx]) as f:
          meta = catch_metadata(json.loads(f.read()))
        

    else:
      idx = np.random.choice([int(i) for i in range(len(self.img_files))])
      img_path = self.img_files[idx]
      img_name  = '/'.join(img_path.split('/')[-4:])
    
      if 'train1' in img_path:
        image = io.BytesIO(self.img_archive["1"].read(img_name))
      else:
        image = io.BytesIO(self.img_archive["2"].read(img_name))
      image = Image.open(image)
      label_path = self.label_files[idx]

      with open(label_path, 'r') as f:
        meta = catch_metadata(json.load(f))
    points, label, text, length = self._select_label(meta)

    image = self._crop_bbox(image, points)
    image = self._aug_training(image)
    image = transforms.ToTensor()(image)


    return image, label, text, length



    
