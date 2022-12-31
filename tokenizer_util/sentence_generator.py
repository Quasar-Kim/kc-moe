import tensorflow_datasets as tfds
from preprocessor import retokenize
import gin

@gin.register
def generate_retokenized_sentence_from_tfds(
    *, 
    dataset,
    split,
    text_column,
    data_dir = None
):
    ds = tfds.load(dataset, split=split, data_dir=data_dir, shuffle_files=True)
    ds = retokenize(ds, target_columns=[text_column])
    for example in ds.as_numpy_iterator():
        sentence = example[text_column].decode('utf-8')
        yield sentence

@gin.register
def generate_sentence_from_tfds(
    *,
    dataset,
    data_dir,
    split,
    text_column
):
    ds = tfds.load(dataset, split=split, data_dir=data_dir, shuffle_files=True)
    for example in ds.as_numpy_iterator():
        sentence = example[text_column].decode('utf-8')
        yield sentence
