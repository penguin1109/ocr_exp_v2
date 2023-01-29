""" ocr_runner.py
- integrates the OCR components (text detector -> text recognizer) for the whole process
- loads the pretrained weight from the local memory (다른 방법도 있으려나?ㅎ)
- [TODO] Use the Graph Neural Network for extracting key information from the extracted text information and location feature
    -> Fusing <position of the text in the recipt image> + <Encoded words>
    -> 상품명, 수량의 정보로 이진 분류만 하면 되는거고 나머지 class는 별로 의미가 없을 것임..
"""
import os, sys, yaml, cv2
BASE=os.path.dirname(os.path.abspath(__file__))
TEXT_DETECTION=os.path.join(BASE, 'text_detection')
TEXT_RECOGNITION_HANGUL=os.path.join(BASE, 'text_recognition_hangul')
TEXT_RECOGNITION_MULTI=os.path.join(BASE, 'text_recognition_multi')

sys.path.append(TEXT_DETECTION);sys.path.append(TEXT_RECOGNITION);

from text_detection.detection import DetectBot
from text_recognition_multi.predict import PredictBot

def crop_images(image, bbox):
    croped = {}
    for idx, box in enumerate(bbox):
        x1,y1,x2,y2=box
        croped[idx] = image[y1:y2, x1:x2, :]
    return croped

def run_ocr(
    detection_cfg: dict, 
    image_path: str, ## (우선은 경로 사용) 이미지는 array의 형태로 flask 서버에서 받아올 것이다.
    detection_model_path: str, ## 사전학습된 CTPN모델 경로,
    remove_white: bool, ## text detection을 수행하기 위해서 주변 테두리를 자르는 전처리 과정을 거칠 것인가
    recognition_model_path: str, ## 사전학습된 HangulNet 모델 경로
    recognition_cfg: dict,
    ):
    detect_bot = DetectBot(
        model_path=detection_model_path, cfg=detection_cfg, remove_white=remove_white
    )
    detected, box, image = detect_bot(image_path) ## bounding box 그려진 원본 이미지와 detection box 좌표
    croped_dict = crop_images(image, bbox=box)
    
    if recognition_model_path is None:
        return detected ## recognition path가 없는 경우에는 그냥 감지된 영역에 bbox처리된 이미지만 return
    else:
        recog_bot = PredictBot(recognition_model_path)
        pred_dict = recog_bot.predict_one_call(croped_dict)
        
        for idx, b in enumerate(box):
            pred_dict[idx]['bbox'] = b
        return pred_dict

    


if __name__ == "__main__":
    CONFIG_PATH=os.path.join(BASE, 'ocr_runner.yaml')
    with open(CONFIG_PATH) as f:
        config = yaml.load(f, Loader = yaml.FullLoader)
    DETECTION_CFG=config['DETECTION_PREDICTION']
    RECOGNITION_CFG=config['RECOGNITION_PREDICTION']
    
    IMAGE_PATH=os.path.join(TEXT_DETECTION, 'demo', 'sample', 'recipt14.jpg')
    DETECTION_MODEL_PATH=os.path.join(TEXT_DETECTION, 'demo', 'weight', 'CTPN_FINAL_CHECKPOINT.pth')
    detected = run_ocr(
        DETECTION_CFG, IMAGE_PATH, DETECTION_MODEL_PATH, True, None, None)

    cv2.imwrite(IMAGE_PATH.split('.')[0] + '_result' + '.jpg', detected)