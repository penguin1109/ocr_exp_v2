""" predict.py
- 입력 이미지를 받아서 (이때 text box단위로 잘린 이미지여야 한다.)
- model의 sqeuence length X class number의 길이의 prediction vector을 바탕으로 실제 글자로 변환
- 한글, 영어, 숫자는 예측 그대로 사용하고 '[UNK]'이면 공백으로 남겨 놓음
"""