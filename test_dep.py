import os
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