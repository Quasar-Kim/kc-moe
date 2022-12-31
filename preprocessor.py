import seqio
import tensorflow as tf
from mecab import MeCab

tokenizer = MeCab()

@seqio.map_over_dataset
def retokenize(example, target_columns):
    # copy
    result = {k: tf.identity(v) for k, v in example.items()}

    for k, v in example.items():
        if k in target_columns:
            result[k] = tf.py_function(_retokenize_sentence, inp=[v], Tout=tf.string)
    
    return result

def _retokenize_sentence(str_tensor):
    try:
        sentence = str_tensor.numpy().decode('utf-8')
        tokens = tokenizer.morphs(sentence)
    except Exception as err:
        print(str_tensor)
        raise err
    return tf.convert_to_tensor(' '.join(tokens), dtype=tf.string)

@seqio.map_over_dataset
def remap(example, *, target_column, mapping):
    target = example[target_column]
    example[target_column] = mapping[target]
    return example

@seqio.map_over_dataset
def to_single_sentence_classification_prompt(example, *, prefix, text_columns, target_column):
    sentences = []
    if len(text_columns) > 1:
        for i, col in enumerate(text_columns):
            sentence = example[col]
            sentences.append(f'문장{i}: {sentence}')
    else:
        text_column = text_columns[0]
        sentence = example[text_column]
        sentences.append(f'문장: {sentence}')
    prompt = prefix + ' ' + '  '.join(sentences)
    return {
        'inputs': prompt,
        'target': example[target_column]
    }

@seqio.map_over_dataset
def ensure_str(example):
    for k, v in example.items():
        if not isinstance(v, str):
            example[k] = str(v)
    return example
