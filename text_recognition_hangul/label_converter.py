import torch
import torch.nn as nn
import torch.nn.functional as F
from jamo import h2j, j2hcj
import os, sys, re
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

class GeneralLabelConverter(object):
  def __init__(self,
                max_length,
                base_character=' 0123456789abcdefghijklmnopqrstuvwxyz가각간갇갈감갑값갓강갖같갚갛개객걀걔거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀귓규균귤그극근글긁금급긋긍기긴길김깅깊까깍깎깐깔깜깝깡깥깨꺼꺾껌껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꾼꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냇냉냐냥너넉넌널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐댓더덕던덜덟덤덥덧덩덮데델도독돈돌돕돗동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률륭르른름릇릎리릭린림립릿링마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몬몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭘뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벨벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브븐블비빌빔빗빚빛빠빡빨빵빼뺏뺨뻐뻔뻗뼈뼉뽑뿌뿐쁘쁨사삭산살삶삼삿상새색샌생샤서석섞선설섬섭섯성세섹센셈셋셔션소속손솔솜솟송솥쇄쇠쇼수숙순숟술숨숫숭숲쉬쉰쉽슈스슨슬슴습슷승시식신싣실싫심십싯싱싶싸싹싼쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓴쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액앨야약얀얄얇양얕얗얘어억언얹얻얼엄업없엇엉엊엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷옹와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡잣장잦재쟁쟤저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쩔쩜쪽쫓쭈쭉찌찍찢차착찬찮찰참찻창찾채책챔챙처척천철첩첫청체쳐초촉촌촛총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칫칭카칸칼캄캐캠커컨컬컴컵컷케켓켜코콘콜콤콩쾌쿄쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택탤터턱턴털텅테텍텔템토톤톨톱통퇴투툴툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔팝패팩팬퍼퍽페펜펴편펼평폐포폭폰표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄햇행향허헌험헤헬혀현혈협형혜호혹혼홀홈홉홍화확환활황회획횟횡효후훈훌훔훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘',
                null_char=u'\u2227',
                unknown_char=u'\u2567'
                ):
    self.characters = null_char + base_character + unknown_char
    self.char_encoder_dict = {}
    self.char_decoder_dict = {}
    self.max_length = max_length
    self.null_label = null_char
    self.unknown_label = unknown_char
    
    for i, char in enumerate(self.characters):
      self.char_encoder_dict[char] = i
      self.char_decoder_dict[i] = char

  def encode(self, text,one_hot=True, padding=True):
    text = text.lower()
    text = re.compile('[^가-힣a-z0-9]').sub('', text)
    def onehot(label, depth, device=None):
      if not isinstance(label, torch.Tensor):
        label = torch.tensor(label, device = device)
      out = torch.zeros(label.size() + torch.Size([depth]), device=device)
      out = out.scatter_(-1, label.unsqueeze(-1), 1)

      return out

    label = []
    for tok in text:
      if tok not in self.char_encoder_dict:
        label.append(self.char_encoder_dict[self.unknown_label])
      else:
        label.append(self.char_encoder_dict[tok])
    if padding:
      label = label + [self.char_encoder_dict[self.null_label]] * (self.max_length - len(label))
    label = torch.tensor(label).to(dtype = torch.long)
    if one_hot:
      label = onehot(label, len(self.characters))
    return label
  
  def decode(self, pred):
    pred_text = [];pred_scores = [];pred_lengths = [];
    if len(pred.shape) == 2:
      pred = pred.unsqueeze(0)
    for p in pred:
      score_ =torch.argmax(F.softmax(p, dim=-1), dim=-1)
      text = ''
      for idx, s in enumerate(score_):
        temp = self.char_decoder_dict[s.item()]
        if temp == self.null_label:
          break
        if temp == self.unknown_label:
          text += ''
        else:
          text += temp
      
      pred_text.append(text)
      pred_scores.append(p.max(dim=1)[0])
      pred_lengths.append(min(len(text)+1, self.max_length))

    return pred_text, pred_scores, pred_lengths

    
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
            additional_character += ''.join(list(map(chr, range(97, 123)))).upper()
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
        # text = text.lower() ## 소문자 사용
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
  label_converter=HangulLabelConverter(max_length=30, add_num=True, add_eng=True)
  text = '아 진짜 짜증나 ABC123'.lower()
  # print(split_syllables(text))
  label = label_converter.encode(text, one_hot=True)
  print(label_converter.char_encoder_dict)
  print(label.shape)
  print(label)
  print(torch.argmax(label,dim=-1))
  
