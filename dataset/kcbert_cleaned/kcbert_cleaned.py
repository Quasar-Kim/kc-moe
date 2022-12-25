import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_io as tfio

class KcbertCleaned(tfds.dataset_builders.TfDataBuilder):
    def __init__(self, **kwargs):
        super().__init__(
            name='kcbert_cleaned',
            version='1.0.0',
            split_datasets=self._get_splits(),
            features=tfds.features.FeaturesDict({
                'text': tfds.features.Text()
            }),
            **kwargs
        )

    def _get_splits(self):
        ds = tf.data.Dataset.list_files('gs://kc-moe/dataset/parquet/kcbert-cleaned')
        ds = ds.interleave(
            lambda file: tfio.IODataset.from_parquet(file, columns=['text']),
            cycle_length=50, # arbitary
            num_parallel_calls=tf.data.AUTOTUNE
        )
        return {
            'train': ds
        }