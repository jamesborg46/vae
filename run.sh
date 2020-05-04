python ./vae.py \
    --name EXP_21 \
    --batch-size 100 \
    --test-batch-size 1000 \
    --epochs 500 \
    --lr 0.0001 \
    --encoder-units 500 \
    --decoder-units 500 \
    --decoder-weight-decay 0.1 \
    --latent-dim 16 \
    --z-std-prior 1 \
    --pre-normalization \

