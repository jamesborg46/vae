python ./vae.py \
    --name EXP_17 \
    --batch-size 100 \
    --test-batch-size 1000 \
    --epochs 500 \
    --lr 0.01 \
    --encoder-units 500 \
    --decoder-units 500 \
    --decoder-weight-decay 0.0 \
    --latent-dim 16 \
    --custom-init \
    --weight-init-std 0.01 \
    --bias-init-std 0.0 \
    --z-std-prior 1.0 \
    --pre-normalization \

