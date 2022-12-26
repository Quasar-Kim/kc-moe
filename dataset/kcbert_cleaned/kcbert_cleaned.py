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
        ds_option = tf.data.Options()
        ds_option.experimental_deterministic = False

        ds = tf.data.Dataset.list_files('gs://kc-moe/dataset/parquet/kcbert-cleaned/*.parquet', shuffle=False)
        ds = ds.with_options(ds_option)
        ds = ds.interleave(
            lambda file: tfio.IODataset.from_parquet(file, columns={
                'text': tf.TensorSpec(shape=(), dtype=tf.string)
            }),
            cycle_length=5,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        return {
            'train': ds
        }