# 모델 처음부터 훈련시키기
Kc-MoE 모델을 T5X를 이용해 TPU에서 처음부터 훈련시켜 봅시다.

## 필요한 프로그램
- gcloud cli
- tensorboard

## 리소스 준비하기
Google Cloud에서 다음 리소스를 준비해주세요. 모든 리소스는 **동일한 리전**에 있는 편이 좋습니다.
1. TPU v3-8 하나
 * 이미지: tpu-vm-base
 * 예시 이름: `v3-train`
2. VM 하나
 * 토크나이저 훈련, 데이터 준비용으로 사용합니다.
 * 토크나이저 훈련 시 메모리가 최소 64GB 이상 필요합니다.
 * 데이터 준비용시에는 메모리를 4GB 정도로 축소해도 괜찮습니다.
 * 예시 이름: `workspace`
3. GCS Bucket 하나
 * 예시 이름: `kc-moe-train`

## step 1. 데이터셋 준비하기
`kcbert_cleaned` 데이터셋을 준비해봅시다.

1. 위에서 만든 VM에 ssh합니다
```
$ gcloud compute ssh workspace
```
2. 다음 명령을 통해 데이터셋을 GCS Bucket에 준비시킵니다.
```sh
# 사전 준비
sudo apt update -y
sudo apt upgrade -y

# 레포지토리 클론
git clone https://github.com/quasar-kim/kc-moe
cd kc-moe

# poetry 설치
curl -sSL https://install.python-poetry.org | python3 - 
export PATH="~/.local/bin:$PATH"

# poetry로 필요한 패키지 설치
poetry install --no-root

# TFDS를 이용해 데이터 준비시키기
cd dataset/kcbert_cleaned
tfds build --data_dir=gs://kc-moe-train/dataset/tfds
```

### step 2. 토크나이저 훈련시키기
`config/sentencepiece` 안에는 다양한 토크나이저 훈련에 사용할 수 있는 gin 설정 파일들이 있습니다.
이중 실제로 사용된 토크나이저 설정은 `morpheme_aware_unigram.gin`입니다. 

```sh
# 토크나이저 훈련
# 훈련에는 약 10시간정도가 소요됩니다.
poetry run python train_tokenizer.py --gin_file=config/sentencepiece/morpheme_aware_unigram.gin

# 훈련된 토크나이저 모델은 vocab/ 디렉토리에 저장됩니다
# 이걸 버킷에 올립시다.
gsutil -m cp -r ./vocab gs://kc-moe-train

# 이제 VM을 꺼도 됩니다.
```

### step 3. 모델 설정 작성하기
모델 훈련 설정도 모두 gin 파일로 작성되어 있습니다. 모델 훈련과 관련된 설정은 `model`, `task` 두가지입니다. `model`에서 훈련시킬 모델을, `task`에서 훈련에 사용할 seqio task(어떤 데이터를 사용할 지)를 골라야 합니다. 그리고 다음 옵션들을 직접 설정해줘야 합니다.

 * TRAIN_STEPS: 몇 step만큼 훈련시킬 것인지
 * MODEL_DIR: 모델 체크포인트 및 로그를 저장할 경로
 * BATCH_SIZE: 배치 사이즈
 * NUM_MODEL_PARTITIONS: model parallelism의 dimension

`config/run`에는 이게 모두 완료된 gin 파일들이 들어 있습니다. Kc-MoE 모델을 훈련할 경우에는 `base_pretrain.gin` 파일에서 `MODEL_DIR`만 바꾸면 훈련을 바로 시작할 수 있습니다. 이 설정은 step 4에서 훈련 시작 시 flag를 통해서 바꿀 수 있으니 지금은 다음 단계로 넘어갑시다.

### step 4. 모델 훈련하기
TPU VM에 ssh합시다.
```sh
gcloud alpha compute tpus tpu-vm ssh v3-train --zone <INSERT-TPU-REGION>
```

TPU VM에서 훈련을 시작해봅시다.
```sh
# 환경설정 스크립트 실행
# 필요한 프로그램 설치, 레포지토리 클론, dependency 설치 수행
source <(curl -s https://raw.githubusercontent.com/Quasar-Kim/kc-moe/main/tpu-setup.sh)

# 여기서 아까 step 3에서 수정한 파일의 변경사항을 반영해주세요

# 훈련이 잘 되나 확인해봅시다
# gin.MODEL_DIR flag를 이용해서 MODEL_DIR 설정을 덮어쓰고 T5X를 이용해 훈련을 시작합니다
# 몇분 후 몇 step인지 알려주는 로그가 뜨고 있으면 ctrl-c를 눌러 정지해주세요.
poetry run python -m t5x.train --gin_file="config/run/base_pretrain.gin" --gin.MODEL_DIR=\"gs://kc-moe-train/t5x/kc-moe\" --gin_search_paths=./flaxformer

# nohup을 이용해서 ssh연결을 끊어도 훈련이 계속되도록 해 놓읍시다
# 로그는 모두 run.log에 기록됩니다
nohup poetry run python t5x.train --gin_file="config/run/base_pretrain.gin" --gin.MODEL_DIR=\"gs://kc-moe-train/t5x/kc-moe\" --gin_search_paths=./flaxformer &> run.log &

# 이제 ssh 연결을 끊어도 됩니다.
```

훈련 진행 상황은 tensorboard를 이용하면 확인할 수 있습니다. `MODEL_DIR`을 logdir로 해서 tensorboard를 실행시키면 진행 상황을 확인할 수 있습니다.
```sh
# Google Cloud 상이 아닌 로컬 컴퓨터에서 실행해도 됩니다
tensorboard --logdir gs://kc-moe-train/t5x/kc-moe
```

1M step까지 훈련을 시키려면 약 8일이 걸립니다.

# 모델 finetuning 시키기
훈련된 모델을 NSMC에 finetuning 시켜봅시다.

## step 1. task 고르기
T5X에서는 Seqio Task를 이용해서 작업을 표현합니다. 여기서 작업은 데이터를 불러오는 것부터 전처리 및 후처리, 매트릭 계산까지 모델을 제외한 훈련 전반을 정의합니다. 

여기서는 NSMC task에 대해 모든게 설정되어 있는 gin 설정 파일인 `config/run/base_nsmc.gin`을 이용하도록 하겠습니다.
이 설정 파일에서 덮어써야 할 설정은 두가지입니다.

* MODEL_DIR: 앞과 동일
* INITIAL_CHECKPOINT_PATH: 미리 훈련된 checkpoint

역시 t5x 실행 시 flag를 통해서 덮어쓰도록 하겠습니다.

## step 2. 모델 훈련하기
TPU VM에 ssh한 후 모델을 훈련시켜봅시다.
```sh
# kc-moe 디렉토리에서 실행해야 합니다
# 약 5분안에 훈련이 완료되므로 nohup은 사용하지 않았습니다.
poetry run python -m t5x.train --gin_file="config/run/base_nsmc.gin" --gin.MODEL_DIR=\"gs://kc-moe-train/t5x/kc-moe-nsmc\" --gin.INITIAL_CHECKPOINT_PATH=\"gs://kc-moe-train/t5x/kc-moe/checkpoint_1000000\" --gin_search_paths=./flaxformer
```

역시 tensorboard를 이용해서 훈련 진행 상황을 확인할 수 있습니다. time series 또는 scalars의 `eval` 그래프에서 validation 셋에 대한 정확도를 보고 제일 정확도가 높게 나온 체크포인트를 확인해주세요. (주의: checkpoint 주기와 validation 주기가 다르므로 그래프상의 특정 점에 해당하는 체크포인트가 없을 수 있습니다. 반드시 gcs bucket에 원하는 체크포인트가 있는지 확인하세요)

## step 3. 모델 성능 측정하기
훈련이 완료된 모델의 성능을 측정해봅시다.
```sh
# TPU VM에서 실행해야 합니다
# gin.CHECKPOINT_PATH를 위에서 성능이 제일 좋게 나왔던 체크포인트로 설정해주세요
poetry run python -m t5x.eval --gin_file=config/run/base_eval.gin --gin.CHECKPOINT_PATH=\"gs://kc-moe-train/t5x/kc-moe-nsmc/checkpoint_1000050\" --gin.MIXTURE_OR_TASK_NAME=\"nsmc\" --gin_search_paths=./flaxformer
```

콘솔에 뜨는 로그를 통해서 accuracy를 확인할 수 있습니다.