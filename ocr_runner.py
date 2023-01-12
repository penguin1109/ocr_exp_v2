""" ocr_runner.py
- integrates the OCR components (text detector -> text recognizer) for the whole process
- loads the pretrained weight from the local memory (다른 방법도 있으려나?ㅎ)
- [TODO] Use the Graph Neural Network for extracting key information from the extracted text information and location feature
    -> Fusing <position of the text in the recipt image> + <Encoded words>
    -> 상품명, 수량의 정보로 이진 분류만 하면 되는거고 나머지 class는 별로 의미가 없을 것임..
"""

