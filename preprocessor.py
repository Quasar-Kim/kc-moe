import seqio
import tensorflow as tf
from mecab import MeCab

tokenizer = MeCab()

@tf.autograph.experimental.do_not_convert
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

@tf.autograph.experimental.do_not_convert
@seqio.map_over_dataset
def to_single_sentence_classification_input(example, *, prefix, text_columns, target_column):
    sentences = []
    if len(text_columns) > 1:
        for i, col in enumerate(text_columns):
            sentence = example[col]
            part = tf.strings.format(
                '문장{}: {}',
                (i, sentence)
            )
            sentences.append(part)
    else:
        text_column = text_columns[0]
        sentence = example[text_column]
        part = tf.strings.format(
            '문장: {}',
            sentence
        )
        sentences.append(part)
    prompt_sentence = tf.strings.join(sentences, separator=' ')
    prompt = tf.strings.join([prefix, ' ', prompt_sentence])
    return {
        'inputs': prompt,
        'targets': example[target_column]
    }

@tf.autograph.experimental.do_not_convert
@seqio.map_over_dataset
def ensure_str(example):
    for k, v in example.items():
        if v.dtype is not tf.string:
            example[k] = tf.strings.as_string(v)
    return example
