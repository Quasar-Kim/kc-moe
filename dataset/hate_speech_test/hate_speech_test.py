from tensorflow_datasets import features
from tensorflow_datasets.core import GeneratorBasedBuilder, Version
from tensorflow_datasets.core.dataset_info import DatasetInfo
import pandas as pd

class HateSpeechTest(GeneratorBasedBuilder):
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
            'test': self._generate_examples()
        }

    def _generate_examples(self):
        df = pd.read_csv('./test.csv').rename(columns={'comments': 'text'})
        for index, row in df.iterrows():
            example = row.to_dict()
            yield index, example