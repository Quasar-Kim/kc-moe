import pandas as pd
from sentencepiece import SentencePieceProcessor
from pprint import pprint

def bench(ds_name, splits, column):
    dfs = []
    for split in splits:
        df = pd.read_parquet(f'gs://kc-moe/dataset/parquet/{ds_name}/{split}.parquet', engine='fastparquet')
        dfs.append(df)
    df = pd.concat(dfs)

    # 1. character level stats
    lens = df[column].str.len()
    stats = {
        'char_max_length': lens.max(),
        'char_mean_length': lens.mean(),
        'char_min_length': lens.min()
    }

    # 2. token level stats
    tokenizer = SentencePieceProcessor()
    tokenizer.load('vocab/morpheme_aware_unigram_32k.model')
    counts = []
    unks = []
    for _, text in df[column].items():
        ids = tokenizer.encode_as_ids(text)
        counts.append(len(ids))
        unk_count = ids.count(tokenizer.unk_id)
        if unk_count > 0:
            unks.append((text, ids, unk_count))
    stats['token_count'] = sum(counts)
    stats['tokens_max_count'] = max(counts)
    stats['tokens_mean_count'] = stats['token_count'] / len(counts)
    stats['unk_tokens_count'] = sum(unks)
    stats['unk_percentage'] = (stats['unk_tokens_count'] / stats['token_count']) * 100

    print(f'{ds_name} - {column}')
    pprint(stats)

    # write all samples with unk tokens
    if len(unks) == 0:
        return
    with open(f'{ds_name}.log', 'w') as f:
        for text, ids, unk_count in unks:
            f.write(f'original: {text}\n')
            f.write(f'decoded: {tokenizer.decode(ids)}\n')
            f.write(f'count: {unk_count}')

bench('nsmc', splits=['train', 'test'], column='text')
bench('kornli', splits=['train', 'validation', 'test'], column='text1')
bench('kornli', splits=['train', 'validation', 'test'], column='text2')
bench('korsts', splits=['train', 'validation', 'test'], column='text1')
bench('korsts', splits=['train', 'validation', 'test'], column='text2')
bench('question-pair', splits=['train', 'test'], column='text')
bench('question-pair', splits=['train', 'test'], column='pair')
bench('hate-speech', splits=['train', 'validation'], column='text')
