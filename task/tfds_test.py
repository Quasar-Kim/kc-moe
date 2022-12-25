import tfds_test
import seqio

seqio.TaskRegistry.add(
    name='tfds_test',
    source=seqio.TfdsDataSource('tfds_test:1.0.0', splits={ 'train': 'train[:75%]', 'test': 'train[75%:]' }),
    preprocessors=[],
    output_features={
        'text': seqio.Feature(
            vocabulary=seqio.PassThroughVocabulary(size=1)
        )
    }
)

ds = seqio.get_mixture_or_task('tfds_test').get_dataset(
    sequence_length={ 'text': 256 },
    split='test',
    shuffle=True,
    num_epochs=1,
    use_cached=False,
    seed=42
)

next(iter(ds))