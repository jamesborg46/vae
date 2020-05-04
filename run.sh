python ./vae.py \
    --name EXP_13 \
    --batch-size 128 \
    --test-batch-size 1000 \
    --epochs 500 \
    --lr 0.0003 \
    --encoder-units 100 \
    --decoder-units 100 \
    --decoder-weight-decay 1 \
    --latent-dim 8 \
    --weight-init-std 0.01 \
    --bias-init-std 0.0 \
    --z-std-prior 0.1 \
    --sigmoidal-mean

