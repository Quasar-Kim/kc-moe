import sentencepiece as spm
import sentencepiece.sentencepiece_model_pb2 as pb_model
import io
import gin
import argparse

@gin.configurable
def train(
    *,
    model_name,
    model_type,
    vocab_size,
    n_extra_symbols,
    sentence_iterator
):
    model = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=sentence_iterator,
        model_type=model_type,
        vocab_size=vocab_size,
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=-1,
        split_by_whitespace=True,
        model_writer=model,
        train_extremely_large_corpus=True
    )
    model_proto = pb_model.ModelProto()
    model_proto.ParseFromString(model.getvalue())
    save_model(model_proto, f'{model_name}.model')

    # add extra symbols
    extra_symbols = [f'▁<extra_id_{i}>' for i in reversed(range(n_extra_symbols))] 
    for symbol in extra_symbols:
        symbol_proto = pb_model.ModelProto().SentencePiece()
        symbol_proto.piece = symbol
        symbol_proto.score = 0.0
        symbol_proto.type = pb_model.ModelProto.SentencePiece.USER_DEFINED
        model_proto.pieces.append(symbol_proto)
    save_model(model_proto, f'{model_name}.{n_extra_symbols}extra.model')

    print(f'Model training complete')

def save_model(model_proto, name):
    model_path = f'vocab/{name}'
    with open(model_path, 'wb') as f:
        f.write(model_proto.SerializeToString())
    print(f'Saved {model_path}')
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin_file', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    gin.parse_config_file(args.gin_file)
    train()
