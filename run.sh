python ./vae.py \
    --name EXP_9 \
    --batch-size 128 \
    --test-batch-size 1000 \
    --epochs 500 \
    --lr 0.003 \
    --encoder-units 500 \
    --decoder-units 500 \
    --decoder-weight-decay 0.1 \
    --latent-dim 8 \
    --weight-init-std 0.1 \
    --bias-init-std 0.0 \
    --z-std-prior 1 \
    --sigmoidal-mean

