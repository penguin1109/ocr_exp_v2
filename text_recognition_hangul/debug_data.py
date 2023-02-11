from dataset import HENDatasetV2, HENDatasetOutdoor, HENDatasetV3
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset
from jamo_utils import jamo_merge, jamo_split
from label_converter_hennet import HangulLabelConverter, GeneralLabelConverter
import yaml, os, json


YAML_DIR='/home/guest/speaking_fridgey/ocr_exp_v2/text_recognition_hangul/configs'
# YAML_NAME='medicine_data.yaml'
YAML_NAME='multi_data_hangul.yaml'
# YAML_NAME='printed_data.yaml'
# YAML_NAME='outdoor_data.yaml'
with open(os.path.join(YAML_DIR, YAML_NAME), 'r') as f:
    CONFIG = yaml.load(f, Loader=yaml.FullLoader)

data_cfg = CONFIG['DATA_CFG']
# BASE_FOLDER=os.path.join(data_cfg['BASE_FOLDER'], 'croped_sentence')
# FILES=os.listdir(BASE_FOLDER)
# IMAGE_FILES = [x for x in FILES if x.split('.')[-1] == 'png']
# JSON_FILE = list(set(FILES) - set(IMAGE_FILES))[0]

# with open(os.path.join(BASE_FOLDER, JSON_FILE), 'r') as f:
#     META_DATA=json.load(f)['annotations']

# print(META_DATA[0:len(META_DATA):10])
# CONVERTER = GeneralLabelConverter(max_length = data_cfg['MAX_LENGTH'] // 3)
CONVERTER = HangulLabelConverter(add_num=True, add_eng=True, add_special=False, max_length=75)
# print(CONVERTER.characters) ## null, 중성 없음, 자-모음, unknown

pred = CONVERTER.encode('아니양 같아',) #  one_hot=False)
# print(len(jamo_merge.join_jamo_char('ㅇ', 'ㅏ', None)))
print(CONVERTER.encode('아니양 같아',one_hot=False))
print(CONVERTER.decode(pred))
print(CONVERTER.char_encoder_dict)
#print(pred)
# print(jamo_merge.join_jamo_char('ㅇ', 'ㅏ', ' '))
if 'outdoor' in YAML_NAME:
    dataset = HENDatasetOutdoor(mode='train', DATA_CFG=data_cfg, raio=1.0)
elif 'printed' in YAML_NAME:
    dataset = HENDatasetV2(mode='train',DATA_CFG=data_cfg, ratio=1.0)
elif 'multi' in YAML_NAME:
    datasets = []
    datasets.append(HENDatasetV3(mode='train', DATA_CFG=data_cfg, ratio=1.0, base_dir='/home/guest/speaking_fridgey/ocr_exp_v2/data/medicine_croped'))
    datasets.append(HENDatasetV3(mode='train', DATA_CFG=data_cfg, ratio=1.0,base_dir='/home/guest/speaking_fridgey/ocr_exp_v2/data/cosmetics_croped'))
    dataset = ConcatDataset(datasets)
elif 'medicine' in YAML_NAME:
    dataset = HENDatasetV3(mode='train', DATA_CFG=data_cfg)
    # dataset = HENDatasetV3(mode='train', DATA_CFG=data_cfg)
loader = torch.utils.data.DataLoader(dataset, batch_size=30, shuffle=True)
for idx, batch in enumerate(loader):
    image, label, text, file_name = batch
    print(text, file_name,)
    break
print(label.shape) ## [Batch Size, Max Length, Class # ]
for l in label:
    print(torch.argmax(l, dim=-1))
"""
# print(dataset.image_files[:10])
# print(dataset.label_data[:10])
target = label
# print(target.shape)
# target = F.one_hot(label).float()
ignore_map = target[:,:,0] == 1
print(ignore_map[0]) ## 각 batch마다 각 위치의 원소가 null이면 True이다.
target[ignore_map] = 0

indices = torch.stack([
        torch.where(x == False)[0][-1] for x in ignore_map
    ], dim=0)
print(indices.sum())
# print(torch.unsqueeze(indices, -1))
loss = torch.ones((30, 1)) * 100

prediction  = torch.rand((30, 30, 55))
target[ignore_map] = prediction[ignore_map] = 0
print(F.cross_entropy(prediction, target, reduction='mean'))
#none_loss = F.cross_entropy(prediction, label, reduction='none')
# mean_loss = F.cross_entropy(prediction, label, reduction = 'mean')
# custom_mean = none_loss.mean()
# print(none_loss, mean_loss, custom_mean)
# print(loss /torch.unsqueeze(indices, -1))
"""