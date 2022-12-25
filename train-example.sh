export GOOGLE_CLOUD_BUCKET_NAME=kc-moe
export TFDS_DATA_DIR=gs://$GOOGLE_CLOUD_BUCKET_NAME/dataset/tfds
export MODEL_DIR=gs://$GOOGLE_CLOUD_BUCKET_NAME/t5x/$(date +%Y%m%d)

python3.9 t5x/train.py \
  --gin_file="t5x/examples/t5/t5_1_1/examples/base_wmt_from_scratch.gin" \
  --gin.MODEL_DIR=\"$MODEL_DIR\" \
  --tfds_data_dir=$TFDS_DATA_DIR