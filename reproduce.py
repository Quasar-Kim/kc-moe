import tensorflow as tf

def prepare_example(example):
    processed = dict()
    for k, v in example.items():
        if isinstance(v, str):
            processed[k] = tf.convert_to_tensor(v)
        elif isinstance(v, list):
            processed[k] = tf.convert_to_tensor([tf.constant(item) for item in v])
        else:
            raise NotImplementedError("wow")
    return processed

example = prepare_example({
	"text": " 11년 개봉된 무비 ‘슈퍼스타 아르바뜨용’의 사실 됨됨이으로 얘깃거리를 모았던 위펑씨가 기신의 목숨을 담은 책 동감. 소나타’(대경북스 · 5만2000원)를 펴냈다 . ",
	"words": ["11년", "개봉된", "무비", "‘슈퍼스타", "아르바뜨용’의", "사실", "됨됨이으로", "얘깃거리를", "모았던", "위펑씨가", "기신의", "목숨을", "담은", "책", "동감.", "소나타’(대경북스", "·", "5만2000원)를", "펴냈다", "."],
	"labels": ["DAT_B", "-", "FLD_B", "AFW_B", "AFW_I", "-", "-", "-", "-", "PER_B", "-", "-", "-", "-", "AFW_I", "AFW_I", "-", "NUM_B", "-", "-"]
})

LABEL_MAPPING = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(['PER', 'FLD', 'AFW', 'ORG', 'LOC', 'CVL', 'DAT', 'TIM', 'NUM', 'EVT', 'ANM', 'PLT', 'MAT', 'TRM']),
        values=tf.constant(['인물', '분야', '인공물', '단체', '장소', '문화', '날짜', '시간', '숫자', '사건', '동물', '식물', '물질', '전문용어'])
    ),
    default_value='-'
)

@tf.autograph.experimental.do_not_convert
def to_ner_input(example):
    # 1. 레이블 바꾸기
    def _map_label(label):
        def _change_tag_name():
            splitted = tf.strings.split(label, sep=tf.constant('_'))
            tag_name, suffix = splitted[0], splitted[1]
            out = LABEL_MAPPING[tag_name] + tf.constant('_') + suffix
            return out
        return tf.cond(
            label == tf.constant('-'),
            lambda: label,
            _change_tag_name
        )
    labels = tf.map_fn(_map_label, elems=example['labels'])

    # 2. prompt 만들기
    prompt = tf.constant('ner 문장: ') + example['text']
    tf.print('prompt:', prompt)

    # 3. target 만들기
    def _reduce_to_target(acc, arg):
        word, label = arg
        def _insert_word_to_tag():
            after_str_len = tf.strings.length(label, unit='UTF8_CHAR') + 2 # label length + special chars '|  ]' - suffix '_B'
            before_str_len = tf.strings.length(acc, unit='UTF8_CHAR') - after_str_len
            s1 = tf.strings.substr(acc, pos=0, len=before_str_len, unit='UTF8_CHAR')
            s2 = tf.strings.substr(acc, pos=-1 * after_str_len, len=after_str_len, unit='UTF8_CHAR') # '| <label> ]'
            out = s1 + word + tf.constant(' ') + s2
            tf.print('inserted', out)
            return out
        def _to_tagged_word():
            splitted = tf.strings.split(label, sep='_')
            tag = splitted[0]
            out = acc + tf.constant(' [ ') + word + tf.constant(' | ') + tag + tf.constant(' ]')
            tf.print('tagged', out)
            return out
        return tf.cond(
            label == tf.constant('-'),
            lambda: acc + tf.constant(' ') + word,
            lambda: tf.cond(
                tf.strings.substr(label, pos=-1, len=1, unit='UTF8_CHAR') == tf.constant('B'),
                _to_tagged_word,
                _insert_word_to_tag
            )
        )
    targets = tf.foldl(_reduce_to_target, elems=(example['words'], labels), initializer=tf.constant(''))
    out = {
        'inputs': prompt,
        'targets': tf.strings.strip(targets),
        'labels': labels,
        'words': example['words'],
        'text': tf.strings.strip(example['text'])
    }
    tf.print(out, summarize=-1)
    return out

out = to_ner_input(example)
print(out)