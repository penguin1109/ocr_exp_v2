import torch
import torch.nn as nn
import torch.nn.functional as F
from jamo import h2j, j2hcj
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from jamo_utils.jamo_split import split_syllables, split_syllable_char
from jamo_utils.jamo_merge import join_jamos

ALPHABET='abcdefghijklmnopqrstuvwxyz'
NUMBER='0123456789'
SPECIAL='.,()'
CHANGE_DICT = {
    "[": "(", "]": ")", "【": "(", "】":")", 
    "〔": "(", "〕":")", "{":"(", "}":")", 
    ">": ")", "<":"(", "|": "ㅣ", "-": "ㅡ", "/": "ㅣ", "~": "ㅡ", "!": "ㅣ",
} ## 특수 문자들을 한글로 바꾸거나 소괄호와 비슷하게 생긴 특수문자들은 모두 소괄호로 변경해 준다.

class HangulLabelConverter(object):
    def __init__(self, 
                base_character=' ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ',
                add_num=False,
                add_eng=False,
                add_special=False,
                max_length=75,
                blank_char=u'\u2227',
                null_char=u'\u2591', ## 이제 텍스트 문자가 끝났음을 의미
                unknown_char=u'\u2567'):
        if add_special:
          additional_character = SPECIAL
        else:
          additional_character = '' ## 일부 포함하고 싶은 특수 문자도 character에 추가해 준다.
        self.blank_char=blank_char
        if add_num:
            additional_character += ''.join(str(i) for i in range(10))
        if add_eng:
            additional_character += ''.join(list(map(chr, range(97, 123))))
        self.char_with_no_tokens = base_character + additional_character
        additional_character += unknown_char ## 문자 dictionary에 포함되지 않는 경우에는 unknown_char 로 처리를 하도록 한다.
        additional_character += blank_char
        self.characters = base_character + additional_character if additional_character != ''  \
            else base_character
        
        # tokens = ['[GO]' '[s]']
        self.null_label = 0 ## <null>의 index는 0으로 설정을 해 준다.
        character_list = list(self.characters)
        # self.characters = [null_char] + tokens + character_list
        self.characters = [null_char] + character_list
        self.char_encoder_dict = {}
        self.char_decoder_dict = {}
        self.max_length = max_length
        self.null_char = null_char
        self.unknown_char = unknown_char

      

        for i, char in enumerate(self.characters):
            self.char_encoder_dict[char] = i ## 데이터셋을 만들떄 ground truth label을 학습을 위해 numeric label로 변환
            self.char_decoder_dict[i] = char ## 모델의 예측에 softmax를 취해서 각각의 sequence의 문자마다 예측한 class number label을 character로 변환

    def encode(self, text, one_hot=True, padding=True):
        def onehot(label, depth, device=None):
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label, device = device)
            onehot = torch.zeros(label.size() + torch.Size([depth]), device=device)
            onehot = onehot.scatter_(-1, label.unsqueeze(-1), 1)

            return onehot
        
        new_text, label = '', []
        ## (1) 입력된 한글 + 숫자 + 영어 문자열을 분리한다.
        # jamo 라이브러리를 사용하면 숫자와 영어는 그대로 둠
        text = text.lower() ## 소문자 사용
        for key, value in CHANGE_DICT.items():
          text = text.replace(key, value)


        text = text.replace(' ', self.blank_char) ## 원래 단어에 있는 공백은 제거하기
        split_toks = split_syllables(text, pad=' ')
        #print(split_toks)
        for tok in split_toks:
            if tok is None:
              temp_idx = int(self.char_encoder_dict[' '])
              label.append(temp_idx)
              new_text += ' '
            elif tok in self.char_encoder_dict:
              temp_idx = int(self.char_encoder_dict[tok])
              label.append(temp_idx)
              new_text += tok
            else:
              label.append(int(self.char_encoder_dict[self.unknown_char]))
        """
        jamo_str = j2hcj(h2j(text))
        text = text.replace(' ', '')
        for idx, j in enumerate(jamo_str.strip(' ')):
        
            if j != ' ':
                new_text += j
                try:
                  temp_idx = int(self.char_encoder_dict[j])
                  label.append(temp_idx)
                except:
                  label.append(int(self.char_encoder_dict[self.unknown_char]))
        """
        if list(set(label)) == [self.unknown_char]:
          return None
        ## (2) char_dict를 사용해서 라벨 만들기
        length = torch.tensor(len(new_text) + 1).to(dtype=torch.long)
        #label = ' '.join(label)
        if padding:
          label = label + [self.null_label] * (self.max_length - len(label))

        label = torch.tensor(label).to(dtype=torch.long)
        ## (3) Cross Entropy학습을 위해서 one hot vector로 만들기
        if one_hot:
            label = onehot(label, len(self.characters))
        return label
    
    def decode(self, predicted):
        ## (1) Softmax 계산을 통해 0-1사이의, 합이 1인 logit으로 변경
        # scores = F.softmax(predicted, dim=2)
        scores=predicted ## output이 LogSoftmax를 붙여서 나옴
        pred_text, pred_scores, pred_lengths = [], [], []
        if len(scores.shape) == 2:
            scores=scores.unsqueeze(0)
            
        for score in scores:
            # score_ = torch.argmax(F.softmax(score, dim=-1), dim=1)
            score_ =torch.argmax(F.softmax(score, dim=-1), dim=-1)
            text = ''
            for idx, s in enumerate(score_):
                temp = self.char_decoder_dict[s.item()]
                if temp == self.null_char:
                  break
                if temp == self.unknown_char: ## 특수 문자등과 같이 예측 대상 character이 아닌 경우
                    text += ''
                if temp == self.blank_char: ## 종성의 None과는 다른 단어사이의 공
                    text += temp
                else:
                    text += temp
            ## (2) 자음과 모음이 분리되어 있는 문자열을 하나의 글자로 merge
            text = join_jamos(text)
            text = text.replace(self.blank_char, ' ')
            pred_text.append(text)
            pred_scores.append(score.max(dim=1)[0])
            pred_lengths.append(min(len(text)+1, self.max_length))

        return pred_text, pred_scores, pred_lengths




if __name__ == "__main__":
  label_converter=HangulLabelConverter(max_length=30)
  text = '아 진짜 짜증나 ABC123'
  # print(split_syllables(text))
  label = label_converter.encode(text, one_hot=True)
  print(label.shape)
  print(label)
  print(torch.argmax(label,dim=-1))
  
