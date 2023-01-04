import torch
import torch.nn as nn
import torch.nn.functional as F
from jamo import h2j, j2hcj
from jamo_utils.jamo_merge import join_jamos

class HangulLabelConverter(object):
    def __init__(self, 
                base_character=' ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ',
                add_num=False,
                add_eng=False,
                max_length=75):
        aditional_character=''
        if add_num:
            additional_character += ''.join(str(i) for i in range(10))
        if add_eng:
            additional_character += ''.join(list(map(chr, range(97, 123))))
        additional_character += '[UNK]' ## 특수 문자의 경우에는 '[UNK]'로 처리를 하도록 한다.
        self.characters = base_character + aditional_character if aditional_character != ''  \
            else base_character
        
        tokens = ['[GO]' '[s]']
        character_list = list(self.character)
        self.character = tokens + character_list
        self.char_encoder_dict = {}
        self.char_decoder_dict = {}
        self.max_length = max_length

        for i, char in enumerate(self.character):
            self.char_encoder_dict[char] = i ## 데이터셋을 만들떄 ground truth label을 학습을 위해 numeric label로 변환
            self.char_decoder_dict[i] = char ## 모델의 예측에 softmax를 취해서 각각의 sequence의 문자마다 예측한 class number label을 character로 변환

    def encode(self, text, one_hot=True):
        def onehot(label, depth, device=None):
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label, device = device)
            onehot = torch.zeros(label.size() + torch.Size([depth]), device=device)
            onehot = onehot.scatter_(-1, label.unsqueeze(-1), 1)

            return onehot
        
        new_text, label = '', ''
        ## (1) 입력된 한글 + 숫자 + 영어 문자열을 분리한다.
        # jamo 라이브러리를 사용하면 숫자와 영어는 그대로 둠
        jamo_str = j2hcj(h2j(text))
        for idx, j in enumerate(jamo_str.strip(' ')):
            if j != ' ':
                new_text += j
                label += self.char_encoder_dict[j]
        ## (2) char_dict를 사용해서 라벨 만들기
        length = torch.tensor(len(new_text) + 1).to(dtype=torch.long)
        label = torch.tensor(label).to(dtype=torch.long)
        ## (3) Cross Entropy학습을 위해서 one hot vector로 만들기
        if one_hot:
            label = onehot(label, len(self.character))
        return label
    
    def decode(self, predicted):
        ## (1) Softmax 계산을 통해 0-1사이의, 합이 1인 logit으로 변경
        scores = F.softmax(predicted, dim=2)
        pred_text, pred_scores, pred_lengths = [], [], []
        for score in scores:
            score_ = score.argmax(dim=1)
            text = ''
            for idx, s in enumerate(score_):
                temp = self.char_decoder_dict[s]
                if temp == '[UNK]':
                    text += ''
                else:
                    text += temp
            ## (2) 자음과 모음이 분리되어 있는 문자열을 하나의 글자로 merge
            text = join_jamos(text)
            pred_text.append(text)
            pred_scores.append(score.max(dim=1)[0])
            pred_lengths.append(min(len(text)+1, self.max_length))

        return pred_text, pred_scores, pred_lengths



