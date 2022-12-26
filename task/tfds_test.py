import seqio
from preprocessor import retokenize
from functools import partial
import tensorflow as tf

seqio.TaskRegistry.add(
    name='tfds_test',
    source=seqio.TfdsDataSource('tfds_test:1.0.0', splits={ 'train': 'train[:90%]', 'test': 'train[90%:]' }),
    preprocessors=[
        partial(retokenize, target_columns=['text'])
    ],
    output_features={
        'text': seqio.Feature(
            vocabulary=seqio.PassThroughVocabulary(size=10),
            dtype=tf.string,
            rank=None
        )
    }
)