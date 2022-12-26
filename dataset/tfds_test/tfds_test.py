"""tfds_test dataset."""

import tensorflow_datasets as tfds
from tensorflow_datasets.core.dataset_info import DatasetInfo
from tensorflow_datasets.core import GeneratorBasedBuilder
import tensorflow_io as tfio
import pandas as pd

# class TfdsTest(tfds.dataset_builders.TfDataBuilder):
#   def __init__(self, **kwargs):
#     super().__init__(
#       name='tfds_test',
#       version='1.0.0',
#       split_datasets={
#         'train': tfio.IODataset.from_parquet('deleteme2.parquet', columns=['text'])
#       },
#       features=tfds.features.FeaturesDict({
#         'text': tfds.features.Text()
#       }),
#       **kwargs
#     )

class TfdsTest(GeneratorBasedBuilder):
  VERSION = tfds.core.Version('1.0.0')

  def _info(self):
    return DatasetInfo(
      builder=self,
      features=tfds.features.FeaturesDict({
        'text': tfds.features.Text()
      })
    )
  
  def _split_generators(self, *args):
    return {
      'train': self._generate_examples()
    }

  def _generate_examples(self):
    df = pd.read_parquet('gs://kc-moe/dataset/parquet/kcbert-cleaned/part.0.parquet')
    for i, row in df.iterrows():
      example = row.to_dict()
      yield i, example