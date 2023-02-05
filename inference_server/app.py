
import os, sys, yaml, cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ocr_runner import run_ocr

from flask import Flask
app = Flask(__name__)



@app.route('/')
def test():
    print("START")
    with open('ocr.yaml', 'r') as f:
        cfg_data = yaml.load(f, Loader=yaml.FullLoader)
    
    recognition_cfg = cfg_data['RECOGNITION_PREDICTION']
    detection_cfg = cfg_data['DETECTION_PREDICTION']
    
    img = cv2.imread('/home/guest/ocr_exp_v2/text_detection/demo/sample/recipt3.jpg')
    detect_dict = run_ocr(
        detection_cfg=detection_cfg,
        image_path=img,
        remove_white=True,
        recognition_cfg=recognition_cfg,
    )
    for key, value in detect_dict.items():
        print(f"VALUE: {value['text']}")
    return "END OF TESTING OCR"

@app.route('/prediction', methods=['POST'])
def prediction():
    with open('ocr_runner.yaml', 'r') as f:
        cfg_data = yaml.load(f)
    
    recognition_cfg = cfg_data['RECOGNITION_PREDICTION']
    detection_cfg = cfg_data['DETECTION_PREDICTION']
    img = request.files['img'] ## Flutter 앱으로부터 이미지를 POST method로 받는다.
    detect_dict = run_ocr(
        detection_cfg=detection_cfg,
        image_path=None,
        remove_white=True,
        recognition_cfg=recognition_cfg,
    ) ## 여기까지는 우선 bounding box와 그에 대응되는 predicted text의 정보가 들어 있다.
    return "END OF PROCESSING OCR"

if __name__ == "__main__": ## 현재 동작하는 유일한 서버임을 보장
    # 단, 외부에서 접근 가능하게 하기 위해서는 run() method의 호출을 변경해서 서버의 접근은 open해야 한다.
    # app.run()
    app.debug=True
    app.run(port=8000, host='0.0.0.0')
    # app.run(port=5000) # host='0.0.0.0') # port 주소를 바꿔주었더니 FLASK 앱이 실행이 되는 것을 확인할 수 있었다.