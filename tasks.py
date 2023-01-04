import seqio
import preprocessor
import postprocessor
import metric
from functools import partial
from t5.data import preprocessors as t5_preprocessor
from t5.evaluation import metrics as t5_metrics

TFDS_DATA_DIR = 'gs://kc-moe/dataset/tfds'
DEFAULT_VOCAB_FILE = 'gs://kc-moe/vocab/morpheme_aware_unigram_32k.model'
DEFAULT_EXTRA_IDS = 100
DEFAULT_OUTPUT_FEATURES = {
    'inputs': seqio.Feature(
        vocabulary=seqio.SentencePieceVocabulary(DEFAULT_VOCAB_FILE, extra_ids=DEFAULT_EXTRA_IDS),
        add_eos=True,
        required=False
    ),
    'targets': seqio.Feature(
        vocabulary=seqio.SentencePieceVocabulary(DEFAULT_VOCAB_FILE, extra_ids=DEFAULT_EXTRA_IDS),
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
            'train': 'train[:-300000]', # 0 ~ -300000(317571849)
            'validation': 'train[-300000:]' # 300000 examples
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
            preprocessor.map,
            target_column='label',
            mapping={
                '0': '부정적',
                '1': '긍정적'
            }
        ),
        partial(
            preprocessor.to_prompt,
            prefix='nsmc',
            text_columns=['text'],
            target_column='label'
        ),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[
        t5_metrics.accuracy
    ]
)

seqio.TaskRegistry.add(
    name='kornli',
    source=get_tfds_source(
        'kornli:1.0.0',
        splits=['train', 'validation', 'test']
    ),
    preprocessors=[
        preprocessor.ensure_str,
        partial(
            preprocessor.map,
            target_column='label',
            mapping={
                '0': '중립',
                '1': '모순',
                '2': '수반'
            }
        ),
        partial(
            preprocessor.to_prompt,
            prefix='kornli',
            text_columns=['text1', 'text2'],
            text_prefixes=['전제', '가설'],
            target_column='label'
        ),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[
        t5_metrics.accuracy
    ]
)

seqio.TaskRegistry.add(
    name='korsts',
    source=get_tfds_source(
        'korsts:1.0.0',
        splits=['train', 'validation', 'test']
    ),
    preprocessors=[
        partial(
            preprocessor.round,
            target_column='score',
            base=0.2
        ),
        partial(
            preprocessor.float_to_str,
            target_column='score',
            precision=1
        ),
        partial(
            preprocessor.to_prompt,
            prefix='korsts',
            text_columns=['text1', 'text2'],
            target_column='score'
        ),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    postprocess_fn=postprocessor.regression,
    metric_fns=[
        t5_metrics.spearman_corrcoef
    ]
)

seqio.TaskRegistry.add(
    name='question_pair',
    source=get_tfds_source(
        'question_pair:1.0.0',
        splits={
            'train': 'train[:90%]',
            'validation': 'train[90%:]',
            'test': 'test'
        }
    ),
    preprocessors=[
        preprocessor.ensure_str,
        partial(
            preprocessor.map,
            target_column='label',
            mapping={
                '0': '같음',
                '1': '다름'
            }
        ),
        partial(
            preprocessor.to_prompt,
            prefix='질문쌍',
            text_columns=['text', 'pair'],
            target_column='label'
        ),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[
        t5_metrics.accuracy
    ]
)

seqio.TaskRegistry.add(
    name='hate_speech',
    source=get_tfds_source(
        'hate_speech:1.0.0',
        splits=['train', 'validation']
    ),
    preprocessors=[
        preprocessor.ensure_str,
        partial(
            preprocessor.map,
            target_column='label',
            mapping={
                '2': '혐오',
                '1': '공격적',
                '0': '일반'
            }
        ),
        partial(
            preprocessor.to_prompt,
            prefix='혐오댓글',
            text_columns=['text'],
            target_column='label'
        ),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[
        t5_metrics.sklearn_metrics_wrapper(
            'f1_score',
            metric_dict_str='f1_macro',
            metric_post_process_fn=lambda x: x * 100,
            average='macro'
        )
    ]
)

# NOTE: 데이터에 문제가 있어 중단
# seqio.TaskRegistry.add(
#     name='naver_ner',
#     source=get_tfds_source(
#         'naver_ner:1.0.0',
#         splits={
#             'train': 'train[:72000]',
#             'validation': 'train[72000:]',
#             'test': 'test'
#         }
#     ),
#     preprocessors=[
#         preprocessor.to_ner_input,
#         seqio.preprocessors.tokenize,
#         seqio.CacheDatasetPlaceholder(),
#         seqio.preprocessors.append_eos_after_trim
#     ],
#     output_features=DEFAULT_OUTPUT_FEATURES,
#     postprocess_fn=postprocessor.to_ner_output,
#     metric_fns=[
#         metric.naver_ner_f1
#     ]
# )
