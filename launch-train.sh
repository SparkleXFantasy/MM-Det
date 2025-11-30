CUDA_VISIBLE_DEVICES=0 python train.py \
    --data-root ./data/DVF_recons \
    --classes youtube stablevideodiffusion \
    --cache-mm \
    --mm-root ./data/DVF_mm_representations \
    --expt MMDet_try_01 