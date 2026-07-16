nohup torchrun --nproc_per_node=8 --master_port=29500 \
    trainer/train_pretrain.py \
    --epochs 2 \
    --batch_size 160 \
    --accumulation_steps 1 \
    --max_seq_len 512 \
    --learning_rate 5e-4 \
    --num_workers 16 \
    --dtype float16 \
    --use_compile 0 \
    --save_dir ../out \
    --data_path ../dataset/pretrain_t2t.jsonl \
    > train.log 2>&1 &
