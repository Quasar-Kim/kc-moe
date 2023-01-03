import postprocessor

def prepare_example(example):
    processed = dict()
    for k, v in example.items():
        if isinstance(v, str):
            processed[k] = v.encode('utf-8')
        elif isinstance(v, list):
            processed[k] = [item.encode('utf-8') for item in v]
        else:
            raise Exception('Unsupported type')
    return processed

def test_one_tagged():
    text = '지둥이 일어나면 [ 방화벽의 | 전문용어 ] 반사경이 위로 들리게 되고요 .'
    example = {
    	"text": "지둥이 일어나면 방화벽의 반사경이 위로 들리게 되고요 .",
    	"words": ['지둥이', '일어나면', '방화벽의', '반사경이', '위로', '들리게', '되고요', '.'],
    }
    output = postprocessor.to_ner_output(text, prepare_example(example), is_target=False)
    assert output == ['-', '-', '전문용어_B', '-', '-', '-', '-', '-']

def test_two_tagged():
    text = '지둥이 일어나면 [ 방화벽의 | 전문용어 ] 반사경이 [ 위로 | 방향 ] 들리게 되고요 .'
    example = {
    	"text": "지둥이 일어나면 방화벽의 반사경이 위로 들리게 되고요 .",
    	"words": ['지둥이', '일어나면', '방화벽의', '반사경이', '위로', '들리게', '되고요', '.'],
    }
    output = postprocessor.to_ner_output(text, prepare_example(example), is_target=False)
    assert output == ['-', '-', '전문용어_B', '-', '방향_B', '-', '-', '-']

def test_multiple_words_tagged():
    text = '지둥이 일어나면 [ 방화벽의 반사경이 | 전문용어 ] 위로 들리게 되고요 .'
    example = {
    	"text": "지둥이 일어나면 방화벽의 반사경이 위로 들리게 되고요 .",
    	"words": ['지둥이', '일어나면', '방화벽의', '반사경이', '위로', '들리게', '되고요', '.'],
    }
    output = postprocessor.to_ner_output(text, prepare_example(example), is_target=False)
    assert output == ['-', '-', '전문용어_B', '전문용어_I', '-', '-', '-', '-']

def test_invalid_format_no_tag():
    text = '지둥이 일어나면 [ 방화벽의 | ] 반사경이 위로 들리게 되고요 .'
    example = {
    	"text": "지둥이 일어나면 방화벽의 반사경이 위로 들리게 되고요 .",
    	"words": ['지둥이', '일어나면', '방화벽의', '반사경이', '위로', '들리게', '되고요', '.'],
    }
    output = postprocessor.to_ner_output(text, prepare_example(example), is_target=False)
    assert output == ['-', '-', '-', '-', '-', '-', '-', '-']

def test_invalid_format_no_word():
    text = '지둥이 일어나면 [ | 전문용어 ] 반사경이 위로 들리게 되고요 .'
    example = {
    	"text": "지둥이 일어나면 방화벽의 반사경이 위로 들리게 되고요 .",
    	"words": ['지둥이', '일어나면', '방화벽의', '반사경이', '위로', '들리게', '되고요', '.'],
    }
    output = postprocessor.to_ner_output(text, prepare_example(example), is_target=False)
    assert output == ['-', '-', '-', '-', '-', '-', '-', '-']

def test_corrupted_format():
    text = '지둥이 일어나면 방화벽의 | 전문용어 ] 반사경이 위로 들리게 되고요 .'
    example = {
    	"text": "지둥이 일어나면 방화벽의 반사경이 위로 들리게 되고요 .",
    	"words": ['지둥이', '일어나면', '방화벽의', '반사경이', '위로', '들리게', '되고요', '.'],
    }
    output = postprocessor.to_ner_output(text, prepare_example(example), is_target=False)
    assert output == ['-', '-', '-', '-', '-', '-', '-', '-']

def test_not_identical_sentence():
    text = '지둥이 일어나면 [ 방화벽의 | 전문용어 ] 셔터가 위로 들리게 되고요 .'
    example = {
    	"text": "지둥이 일어나면 방화벽의 반사경이 위로 들리게 되고요 .",
    	"words": ['지둥이', '일어나면', '방화벽의', '반사경이', '위로', '들리게', '되고요', '.'],
    }
    output = postprocessor.to_ner_output(text, prepare_example(example), is_target=False)
    assert output == ['-', '-', '-', '-', '-', '-', '-', '-']