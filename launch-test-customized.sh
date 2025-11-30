CUDA_VISIBLE_DEVICES=0 python test_customized.py \
    --data-root $YOUR_CUSTOMIZED_DATA_ROOT \
    --classes $YOUR_CUSTOMIZED_CLASS_NAME \
    --ckpt weights/MM-Det/current_model.pth \
    --mm-root $YOUR_CUSTOMZIED_MM_REPRESENTATION_ROOT \
    --cache-mm \
    --sample-size -1


# e.g., if the dataset root is as follows:
# -- my_data_root 
#   | -- my_dataset_A
#      | -- 0_real
#         | -- X.mp4
#            ...
#      |  -- 1_fake
#         | -- Y.mp4
#            ...
# (1) Follow the instruction at dataset/readme.md to get $RECONSTRUCTION_DATASET_ROOT and $MM_REPRESENTATION_ROOT
# (2) The test script is set as 
# CUDA_VISIBLE_DEVICES=0 python test_customized.py \
#     --data-root $RECONSTRUCTION_DATASET_ROOT \
#     --classes my_dataset_A \
#     --ckpt weights/MM-Det/current_model.pth \
#     --mm-root $MM_REPRESENTATION_ROOT \
#     --cache-mm \
#     --sample-size -1
