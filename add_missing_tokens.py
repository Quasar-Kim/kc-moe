import sentencepiece.sentencepiece_model_pb2 as pb_model

model_proto = pb_model.ModelProto()
model_proto.ParseFromString(open('vocab/morpheme_aware_unigram_32k.model', 'rb').read())

def make_symbol_proto(token):
    symbol_proto = pb_model.ModelProto().SentencePiece()
    symbol_proto.piece = token
    symbol_proto.score = 0.0
    return symbol_proto

tokens = '"#&\':=?@^_[|]'
for token in tokens:
    symbol_proto = make_symbol_proto(token)
    model_proto.pieces.append(symbol_proto)
    symbol_proto_2 = make_symbol_proto('▁' + token)
    model_proto.pieces.append(symbol_proto_2)


with open('vocab/fixed.vocab', 'wb') as f:
    f.write(model_proto.SerializeToString())

extra_symbols = [f'▁<extra_id_{i}>' for i in reversed(range(100))] 
for symbol in extra_symbols:
    symbol_proto = pb_model.ModelProto().SentencePiece()
    symbol_proto.piece = symbol
    symbol_proto.score = 0.0
    symbol_proto.type = pb_model.ModelProto.SentencePiece.USER_DEFINED
    model_proto.pieces.append(symbol_proto)

with open('vocab/fixed.100extra.vocab', 'wb') as f:
    f.write(model_proto.SerializeToString())


