import numpy as np
import re

def extract_info(json_data: dict,max_n):
  annotations = json_data['annotations'] ## list
  annotations = annotations[0]

  bbox = []
  for poly in annotations['polygons']:
    text = poly['text']
    points = poly['points']
    minX = np.array(points).T[0].min(); minY = np.array(points).T[1].min();
    maxX = np.array(points).T[0].max(); maxY =np.array(points).T[1].max();

    if re.compile('[^ 가-힣a-zA-Z0-9]').sub('', text) == '':
      continue 
    text = re.compile('[^ 가-힣a-zA-Z0-9]').sub(' ', text) ## 특수 문자는 제거한 상태로 학습을 시키도록 한다.
    if len(text) < max_n or len(text) > 25: ## max sequence length 보다는 짧아야 하고 최단 길이보다는 길어야 한다.
      continue
    bbox.append({'points': [minX, minY, maxX, maxY], 'text': text})
  return bbox

def crop_medicine_data(image, json_data: dict, max_n):
  bbox = extract_info(json_data, max_n)
  # draw_image = image.copy()
  text = []
  points = []
  for idx, box in enumerate(bbox):
    pts = box['points']
    pts = list(map(lambda x: int(x), pts))
    pt1, pt2 = pts[:2], pts[2:]
    
    text.append(box['text'])
    # cv2.rectangle(draw_image, pt1, pt2, color=(255,0,0), thickness = 3)
    points.append(pts)
  return text, points



idx = 0
from tqdm import tqdm
import cv2, os, json
MEDICINE_FOLDER='/home/guest/ocr_exp_v2/data/medicine'

IMAGE=os.path.join(MEDICINE_FOLDER, 'images')
TARGET=os.path.join(MEDICINE_FOLDER, 'annotations')

MEDICINE_IMAGE_DIR=sorted(os.listdir(IMAGE))
MEDICINE_TARGET_DIR=sorted(os.listdir(TARGET))

MEDICINE_DEST='/home/guest/ocr_exp_v2/data/medicine_croped'
os.makedirs(MEDICINE_DEST, exist_ok=True)
with open(os.path.join(MEDICINE_DEST, 'new_target_data.txt'), 'w') as ftext:
  loop = tqdm(zip(MEDICINE_IMAGE_DIR, MEDICINE_TARGET_DIR))
  for image_dir, target_dir in loop:
    if (image_dir.split('.')[0] != target_dir.split('.')[0]):
      print(image_dir, target_dir)
      break
    image = cv2.imread(os.path.join(IMAGE, image_dir))
    with open(os.path.join(TARGET, target_dir), 'r') as f:
      json_data = json.load(f)
    text, points = crop_medicine_data(image, json_data, max_n=5)
    for t, p in zip(text, points):
      try:
        croped = image[p[1]:p[3], p[0]:p[2]]
        cv2.imwrite(os.path.join(MEDICINE_DEST, f"{idx}.png"), croped)
        ftext.write(f"{idx}.png\t{t}\n")
        # print(f"{idx} {t}\n")
        loop.set_postfix({"IDX": idx, "TEXT": t})
        idx += 1
      except:
        continue
      # if (idx == 100000):
      #   exit()

ftext.close()