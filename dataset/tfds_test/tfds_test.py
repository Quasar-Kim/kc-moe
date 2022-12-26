"""tfds_test dataset."""

import tensorflow_datasets as tfds
import tensorflow_io as tfio

class TfdsTest(tfds.dataset_builders.TfDataBuilder):
  def __init__(self, **kwargs):
    super().__init__(
      name='tfds_test',
      version='1.0.0',
      split_datasets={
        'train': tfio.IODataset.from_parquet('gs://kc-moe/dataset/parquet/kcbert-cleaned/part.0.parquet', columns=['text'])
      },
      features=tfds.features.FeaturesDict({
        'text': tfds.features.Text()
      }),
      **kwargs
    )
