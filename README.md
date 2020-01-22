# End-to-End Trainable Fully-Convolutional Siamese Networks for Video Object Segmentation with Bounding Box

This is an implementation of SiamVOS in pytorch.

Compared to conventional semi-supervised video object segmentation methods,
SiamVOS requires only a bounding box of the target for video object segmentation.


## Environment setup
All the code has been tested on Ubuntu18.04, python3.6, Pytorch1.2.0, anaconda2019.03, CUDA 10.0, GTX TITAN x GPU

- Clone the repository
```
git clone https://github.com/nijkah/SiamVOS.git && cd SiamVOS
```

- Setup python environment
```
conda create -n SiamVOS python=3.6
conda activate SiamVOS
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt
```
**NOTICE** imgaug library has an issue on latest numpy=1.8.1,
so pip install git+https://github.com/aleju/imgaug rather than pip install imgaug
For more information see [this](https://github.com/aleju/imgaug/issues/537)

- Download data

[DAVIS-2017](https://davischallenge.org/davis2017/code.html) [Youtube-VOS](https://youtube-vos.org/dataset/vos/) [GyGO](https://github.com/ilchemla/gygo-dataset)

and set paths in **cfg.py**

and you can download the pre-trained deeplab model from
[here](https://drive.google.com/file/d/0BxhUwxvLPO7TeXFNQ3YzcGI4Rjg/view).
Put this in the **data** folder.

- train the model
```
cd scripts
python train_siamvos.py
```

- evaluate the model
```
python evaluate_siamvos.py
```
You can download the trained SiamVOS model from
[here](https://drive.google.com/file/d/1tJELZ_IsP-JK8qyR2AtgeAYiCMtMzoh_/view?usp=sharing).
Put this in the **data/snapshots** folder.

## Results
|         Model         | DAVIS2016 mean IoU | DAVIS2017  |
|:---------------------:|:------------------:|:----------:|
| SiamMask [(paper)](https://arxiv.org/abs/1812.05050)      |        71.7        |    51.1    |
| SiamVOS               |        74.3        |    53.2    |


