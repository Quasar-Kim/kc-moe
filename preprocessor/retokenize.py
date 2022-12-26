import seqio
import tensorflow as tf
from mecab import MeCab

tokenizer = MeCab()

def retokenize_sentence(str_tensor):
    try:
        sentence = str_tensor.numpy().decode('utf-8')
        tokens = tokenizer.morphs(sentence)
    except Exception as err:
        print(str_tensor)
        raise err
    return tf.convert_to_tensor(' '.join(tokens), dtype=tf.string)

@seqio.map_over_dataset
def retokenize(example, target_columns):
    # copy
    result = {k: tf.identity(v) for k, v in example.items()}

    for k, v in example.items():
        if k in target_columns:
            result[k] = tf.py_function(retokenize_sentence, inp=[v], Tout=tf.string)
    
    return result