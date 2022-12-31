import seqio
import preprocessor
from functools import partial
from t5.data import preprocessors as t5_preprocessor
from t5.evaluation import metrics

TFDS_DATA_DIR = 'gs://kc-moe/dataset/tfds'
DEFAULT_VOCAB_FILE = 'gs://kc-moe/vocab/morpheme_aware_unigram_32k.model'
DEFAULT_OUTPUT_FEATURES = {
    'inputs': seqio.Feature(
        vocabulary=seqio.SentencePieceVocabulary(DEFAULT_VOCAB_FILE, extra_ids=100),
        add_eos=True,
        required=False
    ),
    'targets': seqio.Feature(
        vocabulary=seqio.SentencePieceVocabulary(DEFAULT_VOCAB_FILE, extra_ids=100),
        add_eos=True
    )
}

def get_tfds_source(*args, **kwargs):
    return seqio.TfdsDataSource(*args, **kwargs, tfds_data_dir=TFDS_DATA_DIR)

# seqio.TaskRegistry.add(
#     name='tfds_test',
#     source=get_tfds_source('tfds_test:1.0.0', splits={ 'train': 'train[:90%]', 'test': 'train[90%:]' }),
#     preprocessors=[
#         partial(preprocessor.retokenize, target_columns=['text']),
#         partial(seqio.preprocessors.rekey, key_map={
#             'inputs': None,
#             'targets': 'text'
#         }),
#         seqio.preprocessors.tokenize,
#         seqio.CacheDatasetPlaceholder(),
#         t5_preprocessor.span_corruption,
#         seqio.preprocessors.append_eos_after_trim
#     ],
#     output_features=DEFAULT_OUTPUT_FEATURES,
#     metric_fns=[]
# )

seqio.TaskRegistry.add(
    name='kcbert_cleaned',
    source=get_tfds_source(
        'kcbert_cleaned:1.0.0',
        splits={ 
            'train': 'train[:90%]', 
            'test': 'train[90%:]' 
        }
    ),
    preprocessors=[
        preprocessor.ensure_str,
        partial(seqio.preprocessors.rekey, key_map={
            'inputs': None,
            'targets': 'text'
        }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5_preprocessor.span_corruption,
        seqio.preprocessors.append_eos_after_trim
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[]
)

seqio.TaskRegistry.add(
    name='nsmc',
    source=get_tfds_source(
        'nsmc:1.0.0',
        splits={
            'train': 'train[:90%]',
            'validation': 'train[90%:]',
            'test': 'test'
        },
    ),
    preprocessors=[
        preprocessor.ensure_str,
        partial(
            preprocessor.to_single_sentence_classification_input,
            prefix='nsmc',
            text_columns=['text'],
            target_column='label'
        ),
        partial(preprocessor.retokenize, target_columns=['inputs', 'targets']),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[
        metrics.accuracy
    ]
)