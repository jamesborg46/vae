python ./vae.py \
    --name EXP_12 \
    --batch-size 128 \
    --test-batch-size 1000 \
    --epochs 500 \
    --lr 0.0003 \
    --encoder-units 500 \
    --decoder-units 500 \
    --decoder-weight-decay 0.01 \
    --latent-dim 8 \
    --weight-init-std 0.001 \
    --bias-init-std 0.0 \
    --z-std-prior 0.1 \
    --sigmoidal-mean

