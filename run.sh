python ./vae.py \
    --name EXP_20 \
    --batch-size 100 \
    --test-batch-size 1000 \
    --epochs 500 \
    --lr 0.01 \
    --encoder-units 200 \
    --decoder-units 200 \
    --decoder-weight-decay 0.1 \
    --latent-dim 16 \
    --custom-init \
    --weight-init-std 0.01 \
    --bias-init-std 0.01 \
    --z-std-prior 1 \
    --pre-normalization \

