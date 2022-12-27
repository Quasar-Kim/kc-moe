import seqio
from preprocessor import retokenize
from functools import partial
from t5.data import preprocessors
import gin

@gin.register
def get_vocabulary(*, vocab_file, extra_ids):
    return seqio.SentencePieceVocabulary(vocab_file, extra_ids=extra_ids)

DEFAULT_OUTPUT_FEATURES = {
    'inputs': seqio.Feature(
        vocabulary=get_vocabulary(),
        add_eos=True,
        required=False
    ),
    'targets': seqio.Feature(
        vocabulary=get_vocabulary(),
        add_eos=True
    )
}

seqio.TaskRegistry.add(
    name='tfds_test',
    source=seqio.TfdsDataSource('tfds_test:1.0.0', splits={ 'train': 'train[:90%]', 'test': 'train[90%:]' }),
    preprocessors=[
        partial(retokenize, target_columns=['text']),
        partial(seqio.preprocessors.rekey, key_map={
            'inputs': None,
            'targets': 'text'
        }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[]
)