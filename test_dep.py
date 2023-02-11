import os, json, sys, yaml
sys.path.append('/home/guest/ocr_exp_v2/text_recognition_hangul')

from text_recognition_hangul.dataset import HENDatasetV3
from torch.utils.data import DataLoader
with open('/home/guest/ocr_exp_v2/text_recognition_hangul/configs/multi_data_hangul.yaml', 'r') as f:
    data = yaml.load(f, yaml.FullLoader)

data_cfg = data['DATA_CFG']
dataset = HENDatasetV3('train', data_cfg, 1.0)
loader = DataLoader(dataset, batch_size = 30)
for idx, batch in enumerate(loader):
    continue
""""
ANNOTATION_FOLDER='/home/guest/ocr_exp_v2/data/medicine/annotations'
IMAGE_FOLDER='/home/guest/ocr_exp_v2/data/medicine/images'
files = os.listdir(ANNOTATION_FOLDER)

with open(os.path.join(ANNOTATION_FOLDER, files[0])) as f:
    data = json.load(f)
print(data['annotations'])
print(os.path.join(IMAGE_FOLDER, files[0].split('.')[0] + '.jpg'))

IMAGE_FOLDER='/home/guest/ocr_exp_v2/data/medicine_croped'
IMAGE_FILES=os.listdir(IMAGE_FOLDER)
# IMAGE_FILES=[os.path.join(IMAGE_FOLDER, x) for x in IMAGE_FILES]
LABEL_FILE=os.path.join(IMAGE_FOLDER, 'new_target_data.txt')
with open(LABEL_FILE, 'r') as f:
    data = f.readlines()
print(len(IMAGE_FILES), len(data))
for idx, d in enumerate(data):
    image_dir, text = d.strip('\n').split('\t')
    if idx == 1000:
        print(os.path.join(IMAGE_FOLDER, image_dir), text)
        break
"""