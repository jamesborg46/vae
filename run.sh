python ./vae.py \
    --name EXP_25 \
    --batch-size 100 \
    --test-batch-size 1000 \
    --epochs 500 \
    --lr 0.01 \
    --encoder-units 500 \
    --decoder-units 500 \
    --decoder-weight-decay 0.1 \
    --latent-dim 16 \
    --custom-init \
    --weight-init-std 0.01 \
    --bias-init-std 0.01 \
    --decoder-type "gaussian" \
    --sigmoidal-mean \
    --z-std-prior 1 \

