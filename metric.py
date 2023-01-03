from nervaluate import Evaluator as NEREvaluator
from more_itertools import flatten

NAVER_NER_TAGS = ['인물', '분야', '인공물', '단체', '장소', '문화', '날짜', '시간', '숫자', '사건', '동물', '식물', '물질', '전문용어']

def naver_ner_f1(targets, predictions):
    targets = [_to_iob2_labels(t) for t in targets]
    predictions = [_to_iob2_labels(p) for p in predictions]
    evaluator = NEREvaluator(true=targets, pred=predictions, tags=NAVER_NER_TAGS, loader='list')
    results, _ = evaluator.evaluate()
    precision = results['strict']['precision']
    recall = results['strict']['recall']
    f1 = (precision*recall) / (precision+recall) if precision+recall != 0 else 0
    return {
        'f1': f1
    }

def _to_iob2_labels(labels):
    iob2_labels = []
    for label in labels:
        if label == '-':
            new_label = 'O'
        else:
            tag, suffix = label.split('_')
            new_label = f'{suffix}-{tag}'
            new_label = new_label.upper()
        iob2_labels.append(new_label)
    return iob2_labels
