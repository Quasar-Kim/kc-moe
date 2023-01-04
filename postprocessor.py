import re

NER_TAG_REGEXP = re.compile(r'\[ (.+?) \| (.+?) \]')
ERROR_TAG = '-'

class InvalidTaggingException(Exception):
    pass

def regression(output_or_target, *, base, example=None, is_target=False):
    try:
        # to number
        n = float(output_or_target)
        if is_target:
            return n
        # round
        n = base * round(n / base)
    except ValueError:
        # conversion failed
        return -1
    return n

def map_output(output_or_target, *, mapping, example=None, default_value, is_target=False):
    if is_target:
        return output_or_target
    output = output_or_target
    if output not in mapping:
        return default_value
    return mapping[output]

# output_or_target: string
# example - elements are numpy()-ed tensors
#  -> strings are utf-8 encoded bytes
def to_ner_output(output_or_target, example=None, is_target=False):
    if is_target:
        # labels를 대신 돌려주면 된다
        return [label.decode('utf-8') for label in example['labels']]
    else:
        try:
            # 태그된 단어를 찾아서 단어와 태그 추출
            output = output_or_target.strip()
            taggings = []
            untagged_sentence = output
            # 태깅 정보를 제거했을 때 제거된 문장에서 태그된 단어의 index를 알아내려면 
            # 그 태그된 단어 앞의 특수문자([, |, 띄어쓰기 등) 개수를 센 다음에 그만큼 빼줘야 함
            # 그 빼줘야 하는 값
            # 초기값 2는 '[ '에 해당함
            n_formatting_chars = 2
            for match in NER_TAG_REGEXP.finditer(output):
                try:
                    word = match.group(1)
                    tag = match.group(2)
                    start_idx = match.start(1) - n_formatting_chars # start() - inclusvie
                    end_idx = match.end(1) - n_formatting_chars - 1 # end() - exclusive => inclusive
                    taggings.append({
                        'word': word,
                        'tag': tag,
                        'start_idx': start_idx,
                        'end_idx': end_idx
                    })
                    # 끝부분의 ' | <tag> ] ' 와 다음 태그의 '[ '에 해당하는 값
                    n_formatting_chars += 3 + len(tag) + 2 + 2
                    untagged_sentence = untagged_sentence.replace(match.group(0), word)
                except IndexError:
                    # 모델이 태깅 포멧을 지키지 않아서 그룹이 2개가 아님
                    # 예시: 지둥이 일어나면 [ 방화벽의 | ] 반사경이...
                    raise InvalidTaggingException('Invalid tagging format')

            if untagged_sentence != example['text'].decode('utf-8'):
                raise InvalidTaggingException('Output sentence does not match with input sentence')

            # 위에서 추출한 정보에서
            # 1. 태깅된 단어가 샘플의 단어 리스트에 있는지
            # 2. 몇번째 index에 있는지
            # 확인하고 태그 리스트 만들기
            cursor = -1 # 글자 인덱스
            word_idx = -1 # cursor 바로 앞에 있는 단어의 인덱스
            words = [word.decode('utf-8') for word in example['words']]
            predictions = ['-'] * len(words)
            for tagging in taggings:
                while cursor + 1 < tagging['start_idx']:
                    word_idx += 1
                    cursor += len(words[word_idx]) + 1 # 1은 띄어쓰기
                
                start_idx = word_idx + 1
                words_in_tagged_span = []
                while cursor < tagging['end_idx']:
                    word_idx += 1
                    word = words[word_idx]
                    words_in_tagged_span.append(word)
                    cursor += len(word) + 1
                end_idx = word_idx

                if ' '.join(words_in_tagged_span) == tagging['word']:
                    is_first = True
                    for i in range(start_idx, end_idx + 1):
                        if is_first:
                            suffix = 'B'
                            is_first = False
                        else:
                            suffix = 'I'
                        predictions[i] = tagging['tag'] + '_' + suffix
                else:
                    # 모델이 태깅한 단어가 주어진 단어와 다를 수 있음
                    # 예시: 지둥이 일어나 [ 면 방화벽의 | 전문용어 ] 반사경이...
                    raise InvalidTaggingException('Invalid tagging span')
        except InvalidTaggingException:
            predictions = [ERROR_TAG] * len(example['words'])
        return predictions