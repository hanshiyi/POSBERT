#!/bin/bash
GLUE_DIR=../glue_data
TASK_NAME=$1
MODEL=$2
NUM_EPOCHS=5.0
if [ $MODEL = POS ]
    then
    RUN_FILE=run_glue.py
elif [ $MODEL = BERT ]
    then
    RUN_FILE=run_ori_glue.py
else
    echo error: model not specified
    exit 1
fi
python ./$RUN_FILE \
	    --model_type bert \
	    --model_name_or_path bert-base-uncased \
		--task_name $TASK_NAME \
        --do_train \
		--do_eval \
		--do_lower_case \
		--data_dir $GLUE_DIR/$TASK_NAME \
		--max_seq_length 128 \
		--per_gpu_eval_batch_size=8   \
		--per_gpu_train_batch_size=8   \
		--learning_rate 2e-5 \
		--num_train_epochs $NUM_EPOCHS \
		--output_dir ../GLUE_models/$MODEL/$TASK_NAME/ \
        --save_steps 0
