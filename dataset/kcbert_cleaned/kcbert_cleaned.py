from dask.distributed import Client
import dask.dataframe as dd
from tensorflow_datasets import features
from tensorflow_datasets.core import GeneratorBasedBuilder, Version
from tensorflow_datasets.core.dataset_info import DatasetInfo

class KcbertCleaned(GeneratorBasedBuilder):
    VERSION = Version('1.0.0')

    def _info(self):
        return DatasetInfo(
            builder=self,
            features=features.FeaturesDict({
                'text': features.Text()
            })
        )

    def _split_generators(self, *args):
        return {
            'train': self._generate_examples()
        }

    def _generate_examples(self):
        df = dd.read_parquet('gs://kc-moe/dataset/parquet/kcbert-cleaned', engine='fastparquet')
        for i, (_, row) in enumerate(df.iterrows()):
            example = row.to_dict()
            yield i, example

