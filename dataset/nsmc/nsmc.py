from tensorflow_datasets import features
from tensorflow_datasets.core import GeneratorBasedBuilder, Version
from tensorflow_datasets.core.dataset_info import DatasetInfo
import pandas as pd

class Nsmc(GeneratorBasedBuilder):
    VERSION = Version('1.0.0')

    def _info(self):
        return DatasetInfo(
            builder=self,
            features=features.FeaturesDict({
                'text': features.Text(),
                'label': features.ClassLabel(num_classes=2)
            })
        )
    
    def _split_generators(self, *args):
        splits = ['train', 'test']
        return {split: self._generate_examples(split) for split in splits}

    def _generate_examples(self, split):
        url = f'gs://kds-258505083a16ad29d33a74d5c2dacc78743d11e7ade0fa5e527206d9/dataset/parquet/nsmc/{split}.parquet'
        df = pd.read_parquet(url, engine='fastparquet')
        for index, row in df.iterrows():
            example = row.to_dict()
            yield index, example