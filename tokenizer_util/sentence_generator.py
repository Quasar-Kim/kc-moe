import tensorflow_datasets as tfds
from preprocessor.retokenize import retokenize
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
    ds = ds.shuffle(buffer_size=1_000_000)
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
    ds = tfds.load(dataset, split=split, data_dir=data_dir)
    ds = ds.shuffle(buffer_size=1_000_000)
    for example in ds.as_numpy_iterator():
        sentence = example[text_column].decode('utf-8')
        yield sentence