## Customized Preparation

For customized videos on training and inference, the data should be organized into the [reconstruction dataset structure](#reconstruction-dataset) first. We provide our pretrained VQVAE for reconstruction. Please download the [weights](https://drive.google.com/drive/folders/1RRNS8F7ETZWrcBu8fvB3pM9qHbmSEEzy?usp=sharing) at `vqvae/model.pt` and put it at `./weights/`. The reconstruction process is as follows.

1. Prepare the original videos in the following structure. 

```
-- $VIDEO_DATA_ROOT
  | -- dataset A
    | -- 0_real
      | -- {video_1}.mp4/avi/...
      ...
      | -- {video_N}.mp4/avi/...
    | -- 1_fake
      | -- {video_1}.mp4/avi/...
      ...
      | -- {video_N}.mp4/avi/...
  | -- dataset B
    | -- 0_real
      | -- {video_1}.mp4/avi/...
      ...
      | -- {video_N}.mp4/avi/...
    | -- 1_fake
      | -- {video_1}.mp4/avi/...
      ...
      | -- {video_N}.mp4/avi/...
  ...
```

2. Run the following bash to convert videos into frame suquence and generate reconstructed frames.
```bash
python prepare_reconstructed_dataset.py -d $VIDEO_DATA_ROOT -o $RECONSTRUCTION_DATASET_ROOT
```

3. For customized datasets, we provide a procedure for caching MMFR.

  Prepare the dataset as [the reconstructed data structure](#reconstruction-dataset), where the data root is denoted as `$RECONSTRUCTION_DATASET_ROOT`. Then, run the following script to conduct inference on frames based on our finetuned LLaVA.

```bash
python prepare_mm_representations.py --cached-data-root $RECONSTRUCTION_DATASET_ROOT --output-dir $MM_REPRESENTATION_ROOT
```
  The results will be saved as follows, where the MMFR of each class is cached and saved as a pth file.
```
| -- $MM_REPRESENTATION_ROOT
  | -- dataset A
    | -- 0_real
      | -- mm_representation.pth
    | -- 1_fake
      | -- mm_representation.pth
  | -- dataset B
    | -- 0_real
      | -- mm_representation.pth
    | -- 1_fake
      | -- mm_representation.pth
  ...
```

In each pth file, the MMFR for every frame is saved as:
```
{
  '{video_id}_{frame_id}.jpg': {
    'visual': visual_feature,
    'textual': textual_feature
  },
  ...
}
```
