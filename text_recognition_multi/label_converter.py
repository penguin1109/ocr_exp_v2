import torch
from loguru import logger
device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) if len(s) <= batch_max_length else batch_max_length for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        
        for i, t in enumerate(text):
            #print(t)
            t = t.lower()
            text = []
            for tok in list(t):
              if tok in self.dict:
                text.append(self.dict[tok])
              else:
                #logger.info(f"{tok} NOT AVAILABLE IN CHAR DICT -> REPLACED TO SPACE")
                text.append(self.dict[' ']) ## character list에 없다면 공백으로 남겨 두기로 한다.
              ## 아마도 특수 문자가 제거되는 상황일 것이다.
            #text = [self.dict[char] for char in text]
            try:
              batch_text[i][:len(text)] = torch.LongTensor(text)
            except:
              batch_text[i][:len(text)] = torch.LongTensor(text)[:batch_max_length]
              print(f"TEXT : {text} LENGTH : {len(text)}")
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
              ## Connected Target인 경우에 long range dependency를 고려하는 상황이기 때문에 동일한 무자여도 연속적으로 반복 될 수가 있다.
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    ## 근데 아마도 Decode를 하는 과정에서는 정해진 text안에 들어있을 것이기 때문에 비어있는 경우는 없을 것이라 생각한다.
                    char_list.append(self.character[t[i]])
            
            text = ''.join(char_list)
            # print(text)

            texts.append(text)
        return texts
