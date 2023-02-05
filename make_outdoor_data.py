DATA_PATH='/home/guest/ocr_exp_v2/data'
DEST_PATH='/home/guest/ocr_exp_v2/data/croped_outdoor'

FOLDERS=[
    'outdoor_1', 'outdoor_2', 'outdoor_3', 'outdoor_4', 'outdoor_5'
]
from tqdm import tqdm
import os, cv2, json, re
from loguru import logger
os.makedirs(DEST_PATH, exist_ok=True)

IMAGE_FOLDERS=[os.path.join(DATA_PATH, x) for x in FOLDERS]
TARGET_FOLDERS=[os.path.join(DATA_PATH, 'outdoor_target', x) for x in FOLDERS]

cnt = 0
DICT={'annotations': []}
for image_base, target_base in zip(IMAGE_FOLDERS, TARGET_FOLDERS):
    image_names = sorted(os.listdir(image_base))
    label_names = sorted(os.listdir(target_base))
    label_names = [x for x in label_names if x.split('.')[-1] != 'zip']
    
    for img, label in tqdm(zip(image_names, label_names)):
        image = cv2.imread(os.path.join(image_base, img))
        with open(os.path.join(target_base, label), 'r') as f:
            meta = json.load(f)
        bbox = meta['annotations'][0]['bbox']
        text = meta['annotations'][0]['text']
        compiled = re.compile('[ 가-힣a-zA-Z0-9]').sub(text, '')
        if compiled != '': ## 한글, 영어, 숫자, 공백외에 다른 문자가 존재하는 경우에 제외
            continue
        if text != '' and 'x' not in text and 'X' not in text and len(compiled) <= 75:
            try:
                x, y, w, h = bbox
                croped = image[y:y+h, x:x+w]
                cv2.imwrite(os.path.join(DEST_PATH, f"{cnt}.jpg"), croped)
                DICT['annotations'].append({
                    "image": f"{cnt}.jpg", "text": compiled
                })
                cnt += 1
            except:
                logger.info(f"{text} NOT VALID")
                continue


with open(os.path.join(DATA_PATH, 'croped_outdoor.json'), 'w') as f:
    json.dump(DICT, f, indent=4)


"""
annotations 안에 list에서 첫번째 'bbox', 'text'정보를 얻으면 된다.
"""