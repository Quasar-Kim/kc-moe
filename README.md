# Kc-MoE

## 모델 처음부터 훈련시키기

### 리소스 준비하기
- tpu vm 하나
- 일반 vm 하나
- gcs bucket 하나(셋다 같은 리전에)

### step 1. 데이터셋 준비하기
준비할 데이터셋: `kcbert_cleaned`

- 일반 vm에 ssh
- 레포지토리 클론, cd
- poetry 설치
- `poetry shell`
- `poetry install --no-root`
- 데이터셋 준비하기: `cd dataset/kcbert_cleaned` -> `tfds build --data_dir=gs://kc-moe/dataset/tfds`

### step 2. 토크나이저 훈련시키기
- config 파일 `config/sentencepiece`에서 고르기
- 훈련시키기: `python train_tokenizer.py` => vocab 디렉토리에 생김 (이름, 이름.100extra)
- 토큰 추가해야 하면 add_missing_tokens.py 참조
- 버킷에 업로드하기

### step 3. 모델 설정 작성하기
- 두개를 모아야 함: model, task => run
- 오버라이딩 해야 할 설정:
  * TRAIN_STEPS
  * INITIAL_CEHCKPOINT_PATH(finetuning/계속하기 용)
  * MODEL_DIR
  * BATCH_SIZE
  * NUM_MODEL_PARTITIONS
- 오버라이딩 할 수도 있는 설정: NUM_MODEL_PARTITIONS

예시 참조하시오

### step 4. 모델 훈련하기
- tpu vm에 ssh
- 환경설정: `source <(curl -s https://raw.githubusercontent.com/Quasar-Kim/kc-moe/main/tpu-setup.sh)`
  => poetry 설치, 레포지토리 클론, dependency 설치 다 알아서 함
- 훈련시키기: `poetry run python -m t5x.train --gin_file="config/run/kcmoe/pretrain_xl.gin`
 => 오류없이 훈련이 되는지 보기
- 잘 되면 nohup, & 이용
- tensorboard 열어보고 그래프 보고있기