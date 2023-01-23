from text_recognition_hangul.model.hen_net import HENNet
from text_recognition_hangul.label_converter import HangulLabelConverter
import cv2, os, sys
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda:6') if torch.cuda.is_available() else torch.device('cpu')
BASE='/home/guest/ocr_exp_v2/data/printed_text/word/word'
WORD_FILES=os.listdir(BASE)
MODEL_DIR='/home/guest/ocr_exp_v2/weight/2023-01-21_epoch2.pth'
model = HENNet(
    max_seq_length=36,
    embedding_dim=512,
    class_n=54
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_DIR))
image = cv2.imread(os.path.join(BASE, WORD_FILES[4]))
converter = HangulLabelConverter(max_length=36)
tensor_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 128))
])(image)
tensor_image = tensor_image.unsqueeze(0).to(DEVICE)
pred = model(tensor_image)
plt.imshow(image)
print(converter.decode(pred))
