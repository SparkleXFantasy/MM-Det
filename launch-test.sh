CUDA_VISIBLE_DEVICES=0 python test.py \
    --classes videocrafter1 zeroscope opensora sora pika stablediffusion stablevideo \
    --ckpt weights/MM-Det/current_model.pth \
    --mm-root ./data/DVF_mm_representations \
    --cache-mm \
    --sample-size -1