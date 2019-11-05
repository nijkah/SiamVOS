#!/bin/bash

#wget https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip
#unzip DAVIS-data.zip

mkdir ECSSD
cd ECSSD
wget http://www.cse.cuhk.edu.hk/~leojia/projects/hsaliency/data/ECSSD/images.zip
wget http://www.cse.cuhk.edu.hk/~leojia/projects/hsaliency/data/ECSSD/ground_truth_mask.zip
unzip images.zip
unzip ground_truth_mask.zip

cd ..

mkdir MSRA10K
cd MSRA10K
wget http://mftp.mmcheng.net/Data/MSRA10K_Imgs_GT.zip
unzip MSRA10K_Imgs_GT.zip
mkdir images
mkdir annotations
mv MSRA10K_Imgs_GT/Imgs/*.jpg images
mv MSRA10K_Imgs_GT/Imgs/*.png annotations 

cd ..
