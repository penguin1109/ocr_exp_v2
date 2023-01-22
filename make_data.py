import numpy as np
import cv2

def split_gt_text(meta_data:dict, image:np.array):
  text = meta_data['text']
  tokens = text.split(' ') ## 공백기준으로 텍스트 문장이 나눠진다면 이를 이미지에서도 나눌 수 있을 것이다.
  cuts = cut_images(image) ## 공백인 영역의 시작과 끝 부분을 저장해 놓았고, 이 거리가 일정 이상이면 사용
  # 이미지에서 공백의 크기는 거의 일정하다는 점을 이용해서 Text와 Blank를 구분하고자 한다.
  H, W, C = image.shape
  if cuts[0] > cuts[-1]:
    cuts.reverse()

  space = text.count(' ')
  char_number = len(''.join(text.split(' '))) * 3 + space * 1.2
  prev_end = 0
  ranges = []
  for i in range(0, len(cuts)-1,2 ): ## 2개씩 묶음으로 확인해야 한다. (cut_images에서 공백 시작, 공백 끝 부분을 저장하였기 때문이다.)
    left, right = cuts[i], cuts[i+1]
    if (right - left) >= (W // int(char_number)): ## 공백이 전체 가로의 길이를 문자의 개수를 나눈 값 보다 크다면
      if (left-prev_end) >= 30: ## 혹시나 
        ranges.append([prev_end, left])
      prev_end = right

  ranges.append([prev_end, image.shape[1]-1]) ## 마지막 잘리는 부분을 추가해 주어야 한다.
  return ranges, tokens



def cut_images(image:np.array):
  H,W, C = image.shape
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  cuts = []
  prev = 0; was_blank=False;add_blank=False;
  load_w = 0
  while (load_w < W):
    if (image[:, load_w].sum() == (255 * H)): ## 공백 부분 시작
      cuts.append(load_w)
      load_w+=1
      if (load_w == W):break
      while (image[:, load_w].sum() == (255 * H)): ## 공백 부분 끝나기 전까지의 부분을 유지해 주자
        load_w += 1
        if (load_w == W):break
      if (load_w == W):break
      cuts.append(load_w)
    else:
      load_w+=1

  return cuts




SENTENCE_PATH='/home/guest/ocr_exp_v2/data/printed_text/sentence/sentence' ## 문장 AIHUB데이터가 저장되어 있는 경로
JSON_PATH='/home/guest/ocr_exp_v2/data/printed_text/printed_data_info.json' ## 정답 라벨이 저장되어 있는 경로
import os, json, re
import numpy as np
import cv2
from tqdm import tqdm
with open(JSON_PATH, 'r') as f:
    target_data=json.load(f)
SENTENCE_FILES=sorted(os.listdir(SENTENCE_PATH)) ## 문장 이미지 데이터 파일명들
SENTENCE_DATA = [data for data in \
                 target_data['annotations']  if data['attributes']['type'] == '문장' and \
                 data['image_id'] + '.png' in SENTENCE_FILES] ## 문장이면서 문장 이미지가 존재하는 meta data만 저장


MISSED=[]
loop = tqdm(zip(SENTENCE_DATA, sorted(SENTENCE_FILES))) 
cnt = 0;
for meta_info, image_dir in loop:
  sample_image = cv2.imread(os.path.join(SENTENCE_PATH, image_dir)) 
  ranges, tokens = split_gt_text(meta_info, sample_image)
  if len(ranges) != len(tokens):
    cnt += 1;
    MISSED.append(image_dir) ## 개수 놓친애들 
    loop.set_postfix({"MISSED": cnt})

## 텍스트 분리하는 부분이 올바른 결과를 주지 않는 경우에 사용을 안 하게 된다.
SENTENCE_FILES=[x for x in SENTENCE_FILES if x not in MISSED]
SENTENCE_DATA=[x for x in SENTENCE_DATA if x['image_id'] + '.png' not in MISSED]
loop = tqdm(zip(SENTENCE_FILES, SENTENCE_DATA))
JSON_BASE='/home/guest/ocr_exp_v2/data/printed_text'
DEST_BASE='/home/guest/ocr_exp_v2/data/printed_text/croped_sentence'
image_cnt = 0
DICT={'annotations': []}
for idx, (image_fname, image_data) in enumerate(loop):
  image = cv2.imread(os.path.join(SENTENCE_PATH, image_fname))
  bins, tokens = split_gt_text(image_data, image)
  assert len(bins) == len(tokens)
  for bin, tok in zip(bins, tokens):
    if 10 >= len(tok) >= 3: ## 어절의 길이가 3이상 10이하인 데이터만 선택
      croped_image = image[:, bin[0]:bin[1], :]
      # cv2.imwrite(os.path.join(DEST_BASE, f"{image_cnt}.png"), croped_image)
      if re.compile('[ 가-힣]+').sub('', tok) == '':
        only_ko=True ## 한글만 포함하는 경우에는
      else:only_ko=False
      DICT['annotations'].append({
          'image': os.path.join(DEST_BASE, f"{image_cnt}.png"),
          'text': tok,
          'ko': only_ko
      })
      #loop.set_postfix({"idx": image_cnt, "text": tok})
      image_cnt += 1

with open(os.path.join(JSON_BASE, 'croped_target.json'),'w') as f:
    json.dump(DICT, f, indent=4)


