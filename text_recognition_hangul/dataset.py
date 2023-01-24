## (2) Define the PyTorch Based Dataset
import random, os, json, cv2, re
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
random.seed(42) ## randomness에 일정한 비율(?) 아무튼 random choice등을 할때에 중복된 선택 방지 및 재 구현이 가능하도록 하기 위함이다.
from label_converter import HangulLabelConverter

def load_printed_data(label_data, image_file_dict:dict):
  """ AIHUB의 한글 인쇄체 데이터셋을 사용하기 위한 함수
  """
  annotations = label_data['annotations']
  # idx = random.choice([int(i) for i in range(len(annotations))])
  # annotation = annotations[idx]

  dtype= random.choice([ '글자(음절)', '단어(어절)',])# '문장'])
  if dtype == '문장':
    image_files = image_file_dict['sentence']
  elif dtype == '글자(음절)':
    image_files = image_file_dict['syllable']
  elif dtype == '단어(어절)':
    image_files = image_file_dict['word']

  idx = random.choice([int(i) for i in range(len(image_files))])
  image_fdir = image_files[idx]

  image_fname = image_fdir.split('/')[-1]
  image = cv2.imread(image_fdir)
  #print(image_fdir)
  for anno in annotations:
    if anno['image_id'] == image_fname.split('.')[0]:
      break
  return image, anno['text']
  

def load_outdoor_data(label_folder_dir, image_file_dict: dict):
  """ AIHUB의 야외 한글 간판등 데이터셋을 사용하기 위한 함수
  label_folder_dir: 현재 데이터셋으로 사용하고 싶은 outdoor_1 ~ outdoor_5까지 중에서 해당하는 부분의 json 파일 파일이 저장된 폴더를 의미한다.
  """
  fname= label_folder_dir.split('/')[-1]
  
  image_folder_dir = os.path.join(os.path.dirname(label_folder_dir), fname.replace('target_', ''))
  image_folder_dir = image_folder_dir.replace('/outdoor_target', '')
  image_file_dirs = image_file_dict[image_folder_dir] ## 해당 이미지 폴더에 있는 모든 이미지 경로 배열

  idx = random.choice([int(i) for i in range(len(image_file_dirs))])

  image_file_dir = image_file_dirs[idx]
  json_file_name = image_file_dir.split('/')[-1].split('.')[0] + '.json'

  with open(os.path.join(label_folder_dir, json_file_name), 'r') as f:
    label_data = json.load(f)

  bbox = label_data['annotations'][0]['bbox']
  text = label_data['annotations'][0]['text']

  image = cv2.imread(image_file_dir)
  x, y, w, h = bbox

  if x == None or y == None or w == None or h == None:
    return None, None

  croped_image = image[y:y+h, x:x+w]

  return croped_image, text

class BaseDataset(Dataset):
  def __init__(self, mode, DATA_CFG: dict):
    super().__init__()
    self.data_cfg = DATA_CFG
    self.mode = mode
    self.base_folder = DATA_CFG['BASE_FOLDER'] ## 모든 데이터들이 저장이 되어 있는 폴더 
    self.base_characters = DATA_CFG['BASE_CHARACTERS'] ## list
    self.add_num = DATA_CFG['ADD_NUM'] ## 숫자 학습 할지 말지 (bool)
    self.add_eng = DATA_CFG['ADD_ENG'] ## 영어 학습 할지 말지 (bool)
    self.add_special = DATA_CFG['ADD_SPECIAL'] ## 특수 문자 학습 할지 말지 (bool)
    self.max_length = DATA_CFG['MAX_LENGTH'] ## 최대 문자열 길이 (어절로 따지면 25개가 최대)
    self.img_w = DATA_CFG['IMG_W']
    self.img_h = DATA_CFG['IMG_H']
    
    self.label_converter = HangulLabelConverter(
        base_character=''.join(self.base_characters), add_num=self.add_num, add_eng=self.add_eng,
        add_special=self.add_special, max_length=self.max_length
    )

class HENDatasetOutdoor(BaseDataset):
  def __init__(self, mode, DATA_CFG):
    super().__init__(mode, DATA_CFG)
    base_dir='/home/guest/ocr_exp_v2/data/croped_outdoor'
    self.base_dir=base_dir
    image_files = os.listdir(base_dir)
    self.image_files = list(map(lambda x: os.path.join(base_dir, x), image_files))
    label_dir='/home/guest/ocr_exp_v2/data/croped_outdoor.json'
    with open(label_dir, 'r') as f:
      self.label_data = json.load(f)['annotations']
    self._filter_length()

  def __len__(self):
    if self.mode == 'train':
      return int(len(self.image_files) * 0.5)
    elif self.mode == 'examine':
      return len(self.image_files)
    else:
      return 100
  
  def _filter_length(self):
    for i in self.label_data:
      text = i['text']
      label = self.label_converter.encode(text, padding=False, one_hot=False)
      if len(label)> self.max_length:
        print(text)
        print(i['image'])
        self.label_data.remove(i)
        self.image_files.remove(os.path.join(self.base_dir, i['image']))

  def __getitem__(self, idx):
    if self.mode == 'debug' or self.mode == 'test':
      select = idx
    else:
      select = random.choice(self.image_files)
    for label in self.label_data: ## 이미지 데이터에 맞는 라벨 데이터를 찾아줄 수 있다.
      if label['image'] == select.split('/')[-1]:
        break
    text = label['text']

    image = cv2.imread(select)
    image = cv2.resize(image, (self.img_w, self.img_h))
    tensor_image = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=self.data_cfg['MEAN'], std=self.data_cfg['STD'])
    ])(image)

    label = self.label_converter.encode(text, padding=True, one_hot=True)

    return tensor_image, label, text

class HENDatasetV2(BaseDataset):
  ## 오직 인쇄체 데이터셋만을 사용하기 위한 데이터셋
  def __init__(self, mode, DATA_CFG):
    super().__init__(mode, DATA_CFG)
    base_dir=DATA_CFG['BASE_FOLDER']
    self.data_cfg = DATA_CFG
    data_files = os.listdir(os.path.join(base_dir, 'croped_sentence'))
    self.image_files = sorted([x for x in data_files if x.split('.')[-1] == 'png'])
    label_data = list(set(data_files) - set(self.image_files))[0]
    label_data = os.path.join(base_dir, label_data)
    with open(label_data, 'r') as f:
      self.label_data = json.load(f)['annotations']

    self._filter()

  def _filter(self):
    #out_of_char = f"[^{self.label_converter.char_with_no_tokens}]"
    out_of_char = f"[^가-힣]"
    filtered_label_data = []
    filtered_image_data = []
    for data in self.label_data:
      text = data['text']
      if re.search(out_of_char, text):
        #print(text)
        continue
      filtered_label_data.append(data)
      filtered_image_data.append(data['image'])
    self.label_data = filtered_label_data
    self.image_files = sorted(filtered_image_data)
  

  def __len__(self):
    if self.mode == 'train':
      return len(self.image_files) # int(0.5 * len(self.image_files))
    elif self.mode == 'debug':
      return 3000
    else:
      return 100
    
  def __getitem__(self, idx):

    image = cv2.imread(self.image_files[idx])
    image = cv2.resize(image, (self.img_w, self.img_h))
    tensor_image = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(self.data_cfg['MEAN'], self.data_cfg['STD'])
    ])(image)

    text = self.label_data[idx]['text']
    label = self.label_converter.encode(text, padding=True, one_hot=True)

    return tensor_image, label, text




class HENDataset(BaseDataset):
  ## 여러개의 데이터셋을 같이 사용하기 위한 방법이다. (금융, 야외 촬영, 인쇄체 등)
  ## 야외 촬영 데이터셋과 이미지 데이터셋을 모두 같이 사용하기 위한 데이터셋이다.
  def __init__(self, mode, DATA_CFG):
    super().__init__(mode, DATA_CFG)
    
    self.use_outdoor = DATA_CFG['USE_OUTDOOR']
    if DATA_CFG['USE_OUTDOOR'] == True:
      outdoor_dir = os.path.join(self.base_folder, 'outdoor_target')
      self.outdoor_labels = list(map(lambda x: os.path.join(outdoor_dir, x), sorted(os.listdir(outdoor_dir))))
    printed_target = os.path.join(self.base_folder,  'printed_data_info.json')
    with open(printed_target, 'r') as f:
      self.printed_target_data = json.load(f)

    self._make_image_dict()

  def __len__(self):
    if self.mode == 'train':
      items = self.image_file_dict.values()
      n = sum(list(map(lambda x: len(x), items)))
      return n
    else:
      return 100 ## Testing이나 Evaluate할 때는 한번에 하나씩 사용하기 때문에 -> 즉, batch=1으로 inference를 하게 될 것이라는 뜻이다.
  
  def __getitem__(self, idx):
    keys = sorted(list(self.image_file_dict.keys()))
    idx = idx % len(keys)
    
    while True:
      if 'outdoor' in keys[idx]:
        folder_idx = int(keys[idx].split('_')[-1])
        image, text = load_outdoor_data(self.outdoor_labels[folder_idx-1], self.image_file_dict)
      else:
        image, text = load_printed_data(self.printed_target_data, self.image_file_dict)
      #print(text)
      if image is None or text is None:
        continue
      try:
        tensor_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.img_h, self.img_w))
        ])(image)
      except:
        continue
      label = self.label_converter.encode(text, one_hot=True, padding=True)
      if label.shape[0] > self.max_length:
        continue
      else:
        break
    return tensor_image, label, text

  def _make_image_dict(self):
    image_file_dict = {}
    FOLDERS=os.listdir(self.base_folder)
    #print(FOLDERS)
    for folder in FOLDERS:
      if folder not in [
        'word', 'syllable'
      ]:continue
      if 'outdoor' in folder:
        dir = os.path.join(self.base_folder, folder)
        image_file_dict[dir] = list(map(lambda x: os.path.join(dir, x), sorted(os.listdir(dir))))
      elif folder in ['word', 'syllable']:
        dir = os.path.join(self.base_folder, folder)
        if (os.path.isdir(dir) ==False):
          continue
        DTYPES = os.listdir(dir)
        for dtype in DTYPES:
          new_dir = os.path.join(dir, dtype)
          if os.path.isdir(new_dir) == False or 'ipynb' in new_dir:
            continue
          image_file_dict[dtype] = list(map(lambda x: os.path.join(new_dir, x), sorted(os.listdir(new_dir))))
    self.image_file_dict = image_file_dict

if __name__ == "__main__":
  import yaml
  import numpy as np
  import torch
  import torch.nn.functional as F
  from torch.utils.data import DataLoader
  with open('/home/guest/ocr_exp_v2/text_recognition_hangul/configs/printed_data.yaml', 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
  dataset = HENDatasetV2(mode='debug', DATA_CFG=cfg['DATA_CFG'])
  print(len(dataset))
  # print(dataset.label_converter.char_with_no_tokens)
  loader = DataLoader(dataset, batch_size=30)
  MEAN=[0.0,0.0,0.0];STD=[0,0,0];
  NUMS = len(loader)
  from tqdm import tqdm
  loop = tqdm(loader)

  for idx, batch in enumerate(loop):
    image, label, text = batch
    print(image[0].max())
    print(image[0].min())
    # print(text)

  """ MEAN & STD FINDING
  for idx, batch in enumerate(loop):
    image, label, text = batch
    # print(' '.join(text))
    for b in range(image.shape[0]):
      #print(image[b].shape)
      #print(np.unique(np.array(image[b].detach().cpu().numpy())))
      
      cv2.imwrite(f"{b}.png", image[b].detach().permute(1,2 ,0).cpu().numpy()*255)
    for c in range(3):
      MEAN[c] += image[:,c,:,:].mean()
      STD[c] += image[:, c, :, :].std()
    if idx == 3:
      break
  from loguru import logger
  for i in range(3):
    MEAN[i] /= idx
    STD[i] /= idx
  logger.info(f"MEAN: {MEAN}")
  logger.info(f"STD: {STD}")
      # print(image[:,c,:,:].mean())
      # print(image[:, c,:,:].std())
    # break
  """