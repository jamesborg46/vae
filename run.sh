python ./vae.py \
    --name EXP_3_PRE_NORMALIZATION_DEFAULT_INITS \
    --batch-size 128 \
    --test-batch-size 1000 \
    --epochs 500 \
    --lr 0.01 \
    --encoder-units 500 \
    --decoder-units 500 \
    --decoder-weight-decay 0.1 \
    --latent-dim 32 \
    --weight-init-std 0.01 \
    --bias-init-std 0.0 \
    --pre-normalization

