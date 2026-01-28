# [NeurIPS 2024] On Learning Multi-Modal Forgery Representation for Diffusion Generated Video Detection

This repository is the official implementation of [MM-Det](https://arxiv.org/abs/2410.23623) [NeurIPS 2024 Poster].

[![arxiv](https://img.shields.io/badge/arXiv-2310.23623-b31b1b.svg)](https://arxiv.org/abs/2410.23623)

<table class="center">
    <tr>
    <td><img src="assets/overview_of_main_method.png" width="600">></td>
    </tr>
</table>


## Contents

- [\[NeurIPS 2024\] On Learning Multi-Modal Forgery Representation for Diffusion Generated Video Detection](#neurips-2024-on-learning-multi-modal-forgery-representation-for-diffusion-generated-video-detection)
  - [Contents](#contents)
  - [Environment](#environment)
  - [Diffusion Video Forensics Dataset](#diffusion-video-forensics-dataset)
    - [Full Version](#full-version)
    - [Tiny Version](#tiny-version)
  - [Preparation](#preparation)
    - [Pre-trained Weights](#pre-trained-weights)
    - [Reconstruction Dataset](#reconstruction-dataset)
    - [Multi-Modal Forgery Representation](#multi-modal-forgery-representation)
  - [Evaluation](#evaluation)
    - [Data Structure](#data-structure)
    - [Inference](#inference)
      - [Inference on DVF](#inference-on-dvf)
      - [Inference on customized datasets](#inference-on-customized-datasets)
  - [Training](#training)
    - [LMM Fine-tuning (Not necessary)](#lmm-fine-tuning-not-necessary)
    - [Overall Training](#overall-training)
      - [Preliminary Setup](#preliminary-setup)
      - [Training](#training-1)
  - [Acknowledgement](#acknowledgement)
  - [Citation](#citation)

## Environment

1. Install basic packages
```bash
conda create -n MM_Det python=3.10
conda activate 
pip install -r requirements.txt
cd LLaVA
pip install -e .
```
2. For training cases, install additional packages
```bash
cd LLaVA
pip install --upgrade pip
pip install -e ".[train]"
pip install flash-attn==2.5.8 --no-build-isolation
```

## Diffusion Video Forensics Dataset

### Full Version
We release Diffusion Video Forensics (DVF) as the benchmark for forgery video detection. 

The full version of DVF can be downloaded via links: [BaiduNetDisk](https://pan.baidu.com/s/14d-_jLB_yUwKzOosrMHMvg?pwd=296c)(Code: 296c). [google driver](https://drive.google.com/drive/folders/1NxCvJVPSxV2Mib5NaNj5Cf2WnnjrqpMb?usp=drive_link)


### Tiny Version

We also release a tiny version of DVF for quickstart, in which each dataset contains 10 videos, with each video no more than 100 frames. This tiny version can be downloaded via [BaiduNetDisk](https://pan.baidu.com/s/1FeI9OH_7rqTaTd-ldPCAIg?pwd=77x3) (Code:77x3), [google driver](https://drive.google.com/drive/folders/1dtZXrIkbuo_6nkhetFR008KAaCz8JzBL?usp=sharing). We also provide the corresponding reconsturction dataset and MM representations for evaluation in the above link. More information for evaluation can be found at [here](#evaluation).

<table class="center">
    <tr>
    <td><img src="assets/dvf_dataset_statistics.png"></td>
    </tr>
</table>

## Preparation

### Pre-trained Weights
We provide the [weights](https://huggingface.co/sparklexfantasy/llava-7b-1.5-rfrd) for our fine-tuned large multi-modal model, which is based on llava-v1.5-Vicuna-7b from [LLaVA](https://github.com/haotian-liu/LLaVA). The overall weights for MM-Det without the LMM can be obtained via [weights](https://drive.google.com/drive/folders/1RRNS8F7ETZWrcBu8fvB3pM9qHbmSEEzy?usp=sharing) at `MM-Det/current_model.pth`. Please download and put the weights at `./weights/`.

### Reconstruction Dataset

For the full version of DVF, we provide a ready reconstruction dataset at [huggingface](https://huggingface.co/datasets/sparklexfantasy/DVF) [BaiduNetDisk](https://pan.baidu.com/s/1a0sWzGXfkBfblV1wZ70qsQ?pwd=l8h4) (Code: l8h4).

### Multi-Modal Forgery Representation

For the full version of DVF, we provide a ready dataset for cached MMFR at [huggingface](https://huggingface.co/datasets/sparklexfantasy/DVF) [BaiduNetDisk](https://pan.baidu.com/s/1kuybGikCfxs8CnTnxWI-gQ?pwd=m6uy) (Code: m6uy). Since the representation is fixed during training and inference, it is recommended to cache the representation before the overall training to reduce time cost. 

## Evaluation


### Data Structure
For evaluation on the tiny version of DVF, put all files of [the tiny version](#tiny-version) into `./data`. Using follow unzipping commands. 
```
    zip -FF DVF_recons_tiny.zip --out DVF_recons_full.zip
    for file in DVF_recons_full.zip DVF_tiny.zip mm_representations_tiny.zip; do unzip "$file"; done
```
Then, the data structure will be as follows:
```
-- data
  | -- DVF_tiny
  | -- DVF_recons_tiny    # $RECONSTRUCTION_DATASET_ROOT
  | -- mm_representations_tiny  # $MM_REPRESENTATION_ROOT
```

For evaluation on the full version of DVF, download the data at [Reconstruction Dataset](#reconstruction-dataset) and [Multi-Modal Forgery Representation](#multi-modal-forgery-representation). Then put them into `./data`. The data structure is organized as follows:

```
-- data
  | -- DVF
  | -- DVF_recons   # $RECONSTRUCTION_DATASET_ROOT
  | -- mm_representations  # $MM_REPRESENTATION_ROOT
```

For evaluation on customized dataset, details of data preparation can be found at [dataset/readme.md](dataset/readme.md).

### Inference

Make sure the pre-trained weights are organized at `./weights`. 

#### Inference on DVF
Please set `$RECONSTRUCTION_DATASET_ROOT` and `$MM_REPRESENTATION_ROOT` as the data roots provided at [Data Structure](#data-structure) in `launch-test.sh`.
`--cache-mm` is recommended for save the computational and memory cost of LMM branch. Then run `launch-test.sh` for testing on 7 datasets respectively.

```bash
python test.py \
--classes videocrafter1 zeroscope opensora sora pika stablediffusion stablevideo \
--ckpt ./weights/MM-Det/current_model.pth \
--data-root $RECONSTRUCTION_DATASET_ROOT \
--cache-mm \
--mm-root $MM_REPRESENTATION_ROOT\
--sample-size -1 \
## --lmm-ckpt llava-7b-1.5-rfrd ## In case one cannot download via the huggingface interface and then can load from the local. 
```

Since the entire evaluation is time-costing, `sample-size` can be specified (e.g., 1,000) to reduce time by conducting inference only on limited (1,000) videos. To finish the entire evaluation, please set `sample-size` as `-1`.

#### Inference on customized datasets

Make sure the data preparation is done at [dataset/readme.md](dataset/readme.md).
Please set `$YOUR_CUSTOMIZED_DATA_ROOT` `$YOUR_CUSTOMIZED_CLASS_NAME` and `$MM_REPRESENTATION_ROOT` in `launch-test-customized.sh`.
Then run `launch-test-customized.sh`.

## Training

### LMM Fine-tuning (Not necessary)
Our LMM branch is built upon [LLaVA](https://github.com/haotian-liu/LLaVA), with [llava-v1.5-Vicuna-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) set as the base model. Our fine-tuned LMM weights can be achieved [here](https://huggingface.co/sparklexfantasy/llava-7b-1.5-rfrd)(#pretrained-weights). It is recommended to start the overall training directly using our pretrained LMM weights, otherwise the fine-tuning result may not be steady.

We directly conduct the visual instruction tuning stage in [LLaVA](https://github.com/haotian-liu/LLaVA#train) on a gemini-generated instruction dataset [RFRD](placeholder). For more information on customized LMM fine-tuning, please refer to [LLaVA](https://github.com/haotian-liu/LLaVA).

### Overall Training

#### Preliminary Setup
Our ST backbone is based on Hybrid-ViT at [pytorch-image-models](https://github.com/huggingface/pytorch-image-models). Our model is based on `vit_base_resnet50_224_in21k`. To start training from a pretrained model, you can get the pretrained weights at [pytorch-image-models](https://github.com/huggingface/pytorch-image-models). We also provide a direct [link](https://drive.google.com/drive/folders/1RRNS8F7ETZWrcBu8fvB3pM9qHbmSEEzy?usp=sharing) at `ViT/vit_base_r50_s16_224.orig_in21k`. Please put downloaded weights at `./weights/`.

#### Training
Run the following scripts `launch-train.bash` for an overall training on our model. It is recommended to first cache Multi-Modal Forgery Representations at `$MM_REPRESENTATION_ROOT`. In this case, `--cache-mm` is specified, and LMM branch will not be loaded to save huge computational costs and memory usage. 

```bash
python train.py \
--data-root $RECONSTRUCTION_DATASET_ROOT \
--classes youtube stablevideodiffusion \
--cache-mm \
--mm-root $MM_REPRESENTATION_ROOT \
--expt $EXPT_NAME \
```

## Acknowledgement

We express our sincere appreciation to the following projects.

- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- [pytorch-vqvae](https://github.com/ritheshkumar95/pytorch-vqvae)
- [Stable Diffusion](https://github.com/comfyanonymous/ComfyUI)
- [VideoCrafter1](https://github.com/AILab-CVC/VideoCrafter)
- [Zeroscope](https://huggingface.co/cerspense/zeroscope_v2_576w)
- [OpenSora](https://github.com/hpcaitech/Open-Sora)
- [Stable Video Diffusion](https://github.com/Stability-AI/generative-models).

## Citation

```
@inproceedings{ on-learning-multi-modal-forgery-representation-for-diffusion-generated-video-detection,
  author = { Xiufeng Song and Xiao Guo and Jiache Zhang and Qirui Li and Lei Bai and Xiaoming Liu and Guangtao Zhai and Xiaohong Liu },
  title = { On Learning Multi-Modal Forgery Representation for Diffusion Generated Video Detection },
  booktitle = { Proceeding of Thirty-eighth Conference on Neural Information Processing Systems },
  address = { Vancouver, Canada },
  month = { December },
  year = { 2024 },
}
```
