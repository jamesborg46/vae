python ./vae.py \
    --name EXP_6_NO_NORM_SIGM_WTIH_VISUALS \
    --batch-size 128 \
    --test-batch-size 1000 \
    --epochs 500 \
    --lr 0.001 \
    --encoder-units 500 \
    --decoder-units 500 \
    --decoder-weight-decay 0.5 \
    --latent-dim 32 \
    --weight-init-std 0.01 \
    --bias-init-std 0.0 \
    --z-std-prior 0.1 \
    --sigmoidal-mean

