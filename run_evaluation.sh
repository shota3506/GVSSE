#!/usr/bin/env sh

GPU=0
CONFIG=./config/gru.ini
VOCAB=./vocab/glove.840B.300d.vocab.pkl
IMG2VEC=./features/val2017.resnet50.2048d.pth
VAL_JSON=$HOME/dataset/coco/annotations/captions_val2017.json
SENTENCE_ENCODER=./save/gru/sentence_encoder-30.pth
NAME=gru

python evaluate.py \
    --gpu ${GPU} \
    --config ${CONFIG} \
    --vocab ${VOCAB} \
    --img2vec ${IMG2VEC} \
    --val_json ${VAL_JSON} \
    --sentence_encoder ${SENTENCE_ENCODER} \
    --name ${NAME}
