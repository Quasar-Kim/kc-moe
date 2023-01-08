import seqio
import tensorflow as tf
from mecab import MeCab
from functools import partial

tokenizer = MeCab()

@seqio.map_over_dataset
def retokenize(example, target_columns):
    for k, v in example.items():
        if k in target_columns:
            out = tf.py_function(_retokenize_sentence, inp=[v], Tout=tf.string)
            out.set_shape(tf.TensorShape([]))
            example[k] = out
    return example

def _retokenize_sentence(str_tensor):
    try:
        sentence = str_tensor.numpy().decode('utf-8')
        tokens = tokenizer.morphs(sentence)
    except Exception as err:
        print(str_tensor)
        raise err
    return tf.convert_to_tensor(' '.join(tokens), dtype=tf.string)

@seqio.map_over_dataset
def to_prompt(example, *, prefix, text_columns, target_column, text_prefixes=None):
    if len(text_columns) > 1:
        sentences = []
        sentence_prefixes = text_prefixes or [f'문장{i}' for i in range(1, len(text_columns) + 1)]
        for col, sentence_prefix in zip(text_columns, sentence_prefixes):
            sentence = example[col]
            part = tf.constant(sentence_prefix) + tf.constant(': ') + sentence
            sentences.append(part)
        prompt_sentence = tf.strings.join(sentences, separator='  ')
    else:
        text_column = text_columns[0]
        sentence = example[text_column]
        prompt_sentence = tf.constant('문장: ') + sentence
    prompt = tf.strings.join([prefix, ' ', prompt_sentence])
    output = {
        'inputs': prompt
    }
    if target_column is not None:
        output['targets'] = example[target_column]

    return output

@seqio.map_over_dataset
def float_to_str(example, *, target_column, precision):
    n = example[target_column]
    example[target_column] = tf.strings.as_string(n, precision=precision)
    return example

@seqio.map_over_dataset
def ensure_str(example):
    for k, v in example.items():
        if v.dtype is not tf.string:
            example[k] = tf.strings.as_string(v)
    return example

def map(dataset, *, target_column, mapping):
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(list(mapping.keys())),
            values=tf.constant(list(mapping.values()))
        ),
        default_value='Invalid'
    )

    def _map(example):
        original = example[target_column]
        example[target_column] = table[original]
        return example

    return dataset.map(_map, num_parallel_calls=tf.data.AUTOTUNE)

LABEL_MAPPING = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(['PER', 'FLD', 'AFW', 'ORG', 'LOC', 'CVL', 'DAT', 'TIM', 'NUM', 'EVT', 'ANM', 'PLT', 'MAT', 'TRM']),
        values=tf.constant(['인물', '분야', '인공물', '단체', '장소', '문화', '날짜', '시간', '숫자', '사건', '동물', '식물', '물질', '전문용어'])
    ),
    default_value='-'
)

@seqio.map_over_dataset
def round(example, *, target_column, base: float):
    n = example[target_column]
    example[target_column] = base * tf.math.round(n / base)
    return example

@seqio.map_over_dataset
def to_ner_input(example):
    # 1. 레이블 바꾸기
    def _map_label(label):
        def _change_tag_name():
            splitted = tf.strings.split(label, sep=tf.constant('_'))
            tag_name, suffix = splitted[0], splitted[1]
            return LABEL_MAPPING[tag_name] + tf.constant('_') + suffix
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
