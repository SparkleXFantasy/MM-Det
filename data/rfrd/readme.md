# Rich Forgery Reasoning Dataset

To finetune large multi-modal models on forgery detection, we construct a text-image paired description dataset called Rich Forgery Reasoning Dataset. This dataset contains 38k conversations on 1921 real frames and 3579 fake frames, with each description generated from Gemini v1.5 pro for forgery analysis. Real videos are from Youtube, while fake ones are generated by [Stable Video Diffusion](https://github.com/Stability-AI/generative-models). The original frames can be extracted from `youtube` and `stablevideodiffusion` in DVF.

Based on these descriptions, we build up an instruction-based dataset for fine-tuning. This dataset contains up to 38k conversations on 5,500 images, which can be achieved at [rfrd](https://drive.google.com/drive/folders/1buMlUlmxpBfmmF00eXMkzvdJKIXEApvw?usp=sharing).


In `rfrd`, the `rfrd.json` contains the overall descriptions, while `rfrd_train.json` is a subset which only contains all training data without validation data to avoid data leakage.

#### Data Structure

Each element contains an image with a conversation based on forgery detection and reasoning. Please set the data root as `$RECONSTRUCTION_DATASET_ROOT` in [Reconstruction Dataset](#reconstruction-dataset), then the path in `image` refers to the corresponding image.

```json
[
    {
      "id": "311b69de83b528413133ec406b8963d5ca9aa3274dff9d2f64c76adf82edf66d",
      "image": "stablevideodiffusion/1_fake/original/WoXBVCpy2C4_9.jpg",
      "conversations": [
          {
              "from": "human",
              "value": "<image>\nAs an experienced forgery inspection expert, please investigate whether the image is genuine or not. Investigate the legitimacy of the visual assets"
          },
          {
              "from": "gpt",
              "value": "The image shows a person in a red shirt who appears to be lying down on a table.  The subject's arm appears to be unnaturally stretched out, and the angle of the hand holding the flask is awkward."
          },
          {
              "from": "human",
              "value": "provide a further analysis on where it is real or fake."
          },
          {
              "from": "gpt",
              "value": "The image is likely a spoof image because the unnatural pose of the person in the red shirt suggests that the image has been manipulated.  It is possible that the image was created by combining different images to create a humorous or surreal scenario."
          }
      ]
    },
    ...
]
```
