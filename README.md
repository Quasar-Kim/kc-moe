# Kc-MoE
Kc-MoE는 [ST-MoE](https://arxiv.org/abs/2202.08906) 모델을 [KcBERT](https://github.com/Beomi/KcBERT) 데이터셋에 훈련시킨 언어 모델입니다. Mixture-of-Experts(MoE) 레이어를 적용해 파라미터 개수를 늘리면서도 계산량은 비슷하게 유지한 것이 특징인 모델입니다.

## 체크포인트 다운로드
1M step에서의 T5X 체크포인트입니다. 사용하는 방법은 `how-to-train.md`를 참조해주세요.

- [Google Cloud Console에서 보기](https://console.cloud.google.com/storage/browser/kc-moe/checkpoint)
- gsutil에서는 `gs://kc-moe/checkpoint`로 사용하시면 됩니다.

## 데이터셋
Kc-MoE는 온라인 뉴스 댓글을 수집해 Beomi님이 공개하신 KcBERT v2022 데이터셋에 span corruption objective을 이용해 훈련되었습니다. 이 데이터셋에 훈련한 덕분에 기존의 정제된 데이터에 훈련된 모델과 비교했을 때, 인터넷상의 비격식적인 데이터 처리에 더 효과적입니다.

사용된 데이터는 parquet 형식으로 [kaggle에서](https://www.kaggle.com/datasets/quasarkim/kcmoe-pretrain-finetune-dataset) 받으실 수 있습니다.

## 토크나이저
Kc-MoE는 T5와 동일하게 [sentencepiece](https://github.com/google/sentencepiece) 토크나이저를 사용합니다. 전체 토큰 수는 32,000개로 데이터셋의 10%에 대해 unigram 알고리즘을 이용했습니다. 또 훈련 시 각 문장에 대해 mecab 토크나이저를 사용해 mecab이 인식하는 토큰보다 더 큰 토큰이 생기는 것을 방지하는 [morpheme-aware subword tokenization](https://arxiv.org/abs/2010.02534)를 사용했습니다.

사용된 토크나이저 모델은 `vocab` 디렉토리에 있습니다. `100extra`로 끝나는 모델은 span corruption에 사용하기 위해서 100개의 `<unusedxxx>` 토큰을 추가시킨 모델로 토큰 수가 32,100개입니다. Kc-MoE 모델은 실제론 이 토크나이저 모델을 사용합니다.

## 하이퍼파라미터
[Flaxformer](https://github.com/google/flaxformer)에서 제공하는 ST-MoE의 `base` 아키텍쳐를 이용했습니다. 사용한 하이퍼파라미터는 다음과 같습니다.

| variant     | Num. Experts | Num. Heads | Num. Encoders/Decoders | d_ff | d_model | Similar Dense Model |
|-------------|--------------|------------|------------------------|------|---------|---------------------|
| Kc-MoE-base | 16           | 12         | 12/12                  | 2048 | 768     | T5-base             |

학습 진행 시 input sequence length는 512(2^8), batch size는 256(2^8)을 이용해 한 step당 2^17개의 토큰에 대해 학습합니다. 총 1M(약 2^20) step만큼 훈련시켰으므로 사용된 총 토큰 수는 2^37 ≈ 140B개 입니다. 데이터셋이 약 10B개의 토큰으로 이루어져 있고, corruption ratio가 15%기 때문에 동일 데이터는 약 2번정도 학습되었습니다.

## 성능

NSMC와 KorNLI 두 벤치마크 셋에 대해 성능을 측정한 결과입니다.

| model                | NSMC (acc) | KorNLI (acc) |
|----------------------|------------|--------------|
| KcBERT-base          | 89.62      | 74.85        |
| KcELECTRA-v2022-base | 91.97      | 82.12        |
| Kc-MoE               | 91.86      | 80.47        | 

## Acknowledgement
이 모델은 Google의 TRC 프로그램을 통해서 TPU를 무료로 지원받아 훈련되었습니다.