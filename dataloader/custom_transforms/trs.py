import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F

import os, math, random

import cv2
import imgaug as ia
from imgaug import augmenters as iaa

import numpy as np
from glob import glob

from tools.utils import *

def flip(I,flip_p):
    if flip_p>0.5:
        return np.fliplr(I)
    else:
        return I

def aug_batch(img_ref, gt_ref, img_search, gt_search, no_crop=False):

    sometimes_search = lambda aug: iaa.Sometimes(0.5, aug)
    sometimes_deform = lambda aug: iaa.Sometimes(0.9, aug)
    h, w = img_search.shape[:2]
    height = 480 if h <= w else 'keep-aspect-ratio'
    width = 480 if w < h else 'keep-aspect-ratio'

    seq_ref = iaa.Sequential(
        [
            iaa.Resize({'height':height, 'width':width}),
            iaa.CropToFixedSize(400, 400),
            ]
        )

    seq_search = iaa.Sequential(
        [
            iaa.Resize({'height':height, 'width':width}),
            iaa.CropToFixedSize(400, 400),
            sometimes_search(iaa.Affine(
                scale={"x": (2**(-1/8), 2**(1/8)), "y": (2**(-1/8), 2**(1/8))},
                translate_percent={"x": (-15, 15), "y": (-10, 10)}, # translate by -20 to +20 percent (per axis)
                rotate=(-25, 25), # rotate by -45 to +45 degrees
                shear=(-15, 15), # shear by -16 to +16 degrees
                cval=(0, 0), # if mode is constant, use a cval between 0 and 255
                mode='edge' # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            iaa.Add((-10, 10), per_channel=0.5),
        ]
        )
    
    seq_deform  = iaa.Sequential(
        [
            sometimes_deform(iaa.Affine(
                scale={"x": (0.98, 1.02), "y": (0.98, 1.02)},
                translate_px={"x": (-5, 5), "y": (-5, 5)}, # translate by -20 to +20 percent (per axis)
                rotate=(-10, 10), # rotate by -45 to +45 degrees
                shear=(-8, 8), # shear by -16 to +16 degrees
                order=0, # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            #sometimes2(iaa.CoarseDropout(0.2, size_percent=(0.1, 0.5)
            #))
        ], random_order=True
        )

    scale=1
    dim = int(scale*400)
    flip_p = random.uniform(0, 1)

    img_ref_o = flip(img_ref, flip_p)
    gt_ref_o = flip(gt_ref,flip_p)
    img_search_o = flip(img_search,flip_p)
    gt_search_o = flip(gt_search,flip_p)

    oh, ow = img_ref.shape[:2]

    # Process Reference Image
    bb = scale_box(gt_ref)
    if bb[2] != 0 and bb[3] != 0:
        for i in range(5):
            seq_ref_det = seq_ref.to_deterministic()
            gt_ref= ia.SegmentationMapsOnImage(gt_ref_o, shape=gt_ref.shape)
            gt_ref_map = seq_ref_det.augment_segmentation_maps([gt_ref])[0]
            mask_ref  = gt_ref_map.get_arr_int().astype('uint8')
            mask_ref = seq_deform.augment_segmentation_maps([gt_ref_map])[0].get_arr_int().astype(float)

            #scale_p = random.uniform(0.8, 1.2)
            bb = scale_box(mask_ref)
            #bb = cv2.boundingRect(template_mask.astype('uint8'))
            if bb[2] >= 10 and bb[3] >= 10:
                img_ref = seq_ref_det.augment_image(img_ref_o)
                mask_ref = np.zeros([*mask_ref.shape[:2], 1])
                mask_ref[bb[1]:bb[1]+bb[3]+1, bb[0]:bb[0]+bb[2]+1, 0] = 1
                break
        else:
            img_ref = cv2.resize(img_ref_o, (dim, dim))
            mask_ref = np.expand_dims(cv2.resize(gt_ref_o, (dim, dim)), 2)
        img_ref= cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB).astype(float)/255.
    else:
        img_ref = np.zeros([dim, dim, 3]).astype('uint8')
        mask_ref = np.zeros([dim, dim, 1])

    
    # Augment Search Image
    if not no_crop:
        img_search_o = crop_and_padding(img_search_o, gt_search_o, (dim, dim)).astype('uint8')
        gt_search_o = crop_and_padding(gt_search_o, gt_search_o, (dim, dim)).astype('uint8')

    for i in range(10):
        seq_search_det = seq_search.to_deterministic()

        gt_search = ia.SegmentationMapsOnImage(gt_search_o, shape=gt_search_o.shape)
        gt_search_map = seq_search_det.augment_segmentation_maps([gt_search])[0]
        gt_search = gt_search_map.get_arr().astype('uint8')
        bb = cv2.boundingRect(gt_search)
        if bb[2] > 10 and bb[3] > 10:
            img_search = seq_search_det.augment_image(img_search_o)
            break
    else:
        img_search = cv2.resize(img_search_o, (dim, dim))
        gt_search = np.expand_dims(cv2.resize(gt_search_o, (dim, dim)), 2).astype(float)
        
    # Get previous (mask and frame) == Deform (mask and image)
    for i in range(5):
        seq_deform_det = seq_deform.to_deterministic()
        p_mask = seq_deform_det.augment_segmentation_maps([gt_search_map])[0].get_arr().astype(float)
        bb = cv2.boundingRect(p_mask.astype('uint8'))
        if bb[2] > 10 and bb[3] > 10:
            img_prev = seq_deform_det.augment_image(img_search.copy())
            img_prev = seq_deform.augment_image(img_prev)
            break
    else:
        img_prev = img_search.copy()
        p_mask = gt_search.copy()

    img_search = cv2.cvtColor(img_search, cv2.COLOR_BGR2RGB).astype(float)/255.
    img_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2RGB).astype(float)/255.
 
    aug_p = random.uniform(0, 1)
    iter = random.randint(1, 5)
    if aug_p > 0.5:
        aug = np.expand_dims(cv2.dilate(p_mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5)), iterations=iter), 3)
    else:
        aug = np.expand_dims(cv2.erode(p_mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5)), iterations=iter), 3)
    p_mask = np.zeros([dim, dim, 1])
    if flip_p <= 0.8:
        p_mask[np.where(aug==1)] = 1
    else:
        bb = cv2.boundingRect(aug.astype('uint8'))
        p_mask[bb[1]:bb[1]+bb[3]+1, bb[0]:bb[0]+bb[2]+1, 0] = 1

    #gt_search = gt_search.astype(float)
    
    return img_search, p_mask, img_prev, img_ref, mask_ref, gt_search


