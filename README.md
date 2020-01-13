# End-to-End Trainable Fully-Convolutional Siamese Networks for Video Object Segmentation with Bounding Box

This is an implementation of SiamVOS in pytorch.

Compared to conventional semi-supervised video object segmentation methods,
SiamVOS requires only a bounding box of the target for video object segmentation.


## Environment setup
All the code has been tested on Ubuntu18.04, python3.6, Pytorch1.2.0, CUDA 9.0, GTX TITAN x GPU

- Clone the repository
```
git clone https://github.com/nijkah/SiamVOS.git && cd SiamVOS
```

- Setup python environment
```
conda create -n SiamVOS python=3.6
source activate SiamVOS
conda install pytorch=1.2.0 cuda90 -c pytorch
pip install -r requirments.txt
```

- Download data
```
cd data
sh download_datasets.sh
cd ..
```
and you can download the pre-trained deeplab model from
[here](https://drive.google.com/file/d/0BxhUwxvLPO7TeXFNQ3YzcGI4Rjg/view).
Put this in the 'data' folder.

- train the model
```
cd train
python train.py
```

- evaluate the model
```
cd evaluatation
python evaluate.py
```
You can download the trained SiamVOS model from
[here](https://drive.google.com/file/d/1tJELZ_IsP-JK8qyR2AtgeAYiCMtMzoh_/view?usp=sharing).
Put this in the 'data/snapshots' folder.

## Results
|         Model         | DAVIS2016 mean IoU | DAVIS2017  |
|:---------------------:|:------------------:|:----------:|
| SiamMask [(paper)](https://arxiv.org/abs/1812.05050)      |        71.7        |    51.1    |
| SiamVOS               |        74.3        |    53.2    |


