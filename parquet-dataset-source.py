from seqio import DataSource
import tensorflow_io as tfio
import tensorflow as tf

class DeadSimpleDataSourceBase(DataSource):
    def num_input_examples(self, split: str):
        raise NotImplementedError('num_input_examples() is not supported')
    
    def list_shards(self, split: str):
        raise NotImplementedError('no list_shards() is not supported')

class ParquetFileDataSource(DeadSimpleDataSourceBase):
    def __init__(self, *, parquet_file, columns):
        self._file = parquet_file
        self._columns = columns
        super().__init__(splits=(), caching_permitted=True)

    def get_dataset(self):
        return tfio.IODataset.from_parquet(self._file, columns=self._columns)

class ShardedParquetDataSource(DeadSimpleDataSourceBase):
    def __init__(self, *, parquet_dir, columns):
        self._dir = parquet_dir
        self._columns = columns
        super().__init__(splits=(), caching_permitted=True)

    def get_dataset(self):
        ds_options = tf.data.Options()
        ds_options.experimental_deterministic = False

        ds = tf.data.Dataset.list_files(self._dir)
        ds = ds.with_options(ds_options)
        return ds.interleave(
            lambda file: tf.IODataset.from_parquet(file, columns=self._columns),
            cycle_length=10 # arbitary
            num_parallel_calls=tf.data.AUTOTUNE
        )