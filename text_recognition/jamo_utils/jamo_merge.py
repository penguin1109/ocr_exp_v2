import os, sys, re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jamo_utils import check_hangul, INITIAL, MEDIAL, FINAL, CHAR_INDICES,\
     get_jamo_type, CHARSET

def join_jamo_char(init, mid, final=None):
    chars = (init, mid, final)
    for c in filter(None, chars):
        check_hangul(c, jamo_only=True)
    idx = tuple(CHAR_INDICES[pos][c] if c is not None else c \
        for pos, c in zip((INITIAL, MEDIAL, FINAL), chars))
    init_idx, med_idx, final_idx = idx
    final_idx = 0 if final_idx is None else final_idx + 1

    return chr(0xac00 + 28 * 21 * init_idx + 28 * med_idx + final_idx)

def join_jamos(s: str, ignore_err=True):
    last_t = 0
    queue = []
    new_string = ''
    def flush(n = 0):
        new_queue = []
        while len(queue) > n:
            new_queue.append(queue.pop())
        if len(new_queue) == 1:
            if not ignore_err:
                raise ValueError(f"Invalid jamo character: {new_queue[0]}")
            result = new_queue[0]

        elif len(new_queue) >= 2:
            try:
                result = join_jamo_char(*new_queue)
            except (ValueError, KeyError):
                if not ignore_err:
                    raise ValueError(f"Invalid jamo characters: {new_queue}")
                result = "".join(new_queue)
        else:
            result = None
        return result
    
    for char in s:
        if char not in CHARSET:
            if queue:
                new_c = flush() + char
            else:
                if re.sub('[A-Za-z0-9]', '', char) == '': ## 숫자, 영어 인 경우
                    new_c = char
                else: ## 특수 문자인 경우
                    new_c = ''
            last_t = 0
        else:
            t = get_jamo_type(char)
            new_c = None
            if t & FINAL == FINAL:
                if not (last_t == MEDIAL):
                    new_c = flush()
            elif t == INITIAL:
                new_c = flush()
            elif t == MEDIAL:
                if last_t & INITIAL ==INITIAL:
                    new_c = flush(1)
                else:
                    new_c = flush()
            last_t = t
            queue.insert(0, char)
                
        if new_c:
            new_string += new_c
    if queue:
        new_string += flush()
    return new_string
            

                

            
    


if __name__ == "__main__":
    # init, mid, final = map(str, input().split(' '))
    s = str(input())
    print(join_jamos(s))