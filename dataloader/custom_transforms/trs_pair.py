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


def outS(i):
    """Given shape of input image as i,i,3 in deeplab-resnet model, this function
    returns j such that the shape of output blob of is j,j,21 (21 in case of VOC)"""
    j = int(i)
    #j = int((j+1)/2)+1
    j = int((j+1)/2)
    #j = int(np.ceil((j+1)/2.0))
    #j = int((j+1)/2)
    return j

def resize_label_batch(label, size):
    label_resized = np.zeros((size,size,1,label.shape[3]))
    #interp = nn.Upsample(size=(size, size), mode='bilinear')
    labelVar = torch.from_numpy(label.transpose(3, 2, 0, 1))
    #label_resized[:, :, :, :] = interp(labelVar).data.numpy().transpose(2, 3, 1, 0)
    label_resized[:, :, :, :] = F.interpolate(labelVar, size=(size, size), 
            mode='bilinear', align_corners=True).data.numpy().transpose(2, 3, 1, 0)
    label_resized[label_resized>0.3]  = 1
    label_resized[label_resized != 0]  = 1

    return label_resized

def flip(I,flip_p):
    if flip_p>0.5:
        return np.fliplr(I)
    else:
        return I

def aug_pair(img_template, img_search, gt_template, gt_search, no_crop=False):

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    sometimes2 = lambda aug: iaa.Sometimes(0.9, aug)
    h, w = img_search.shape[:2]
    height = 480 if h <= w else 'keep-aspect-ratio'
    width = 480 if w < h else 'keep-aspect-ratio'

    seq_ref = iaa.Sequential(
        [
            iaa.Resize({'height':height, 'width':width}),
            iaa.CropToFixedSize(400, 400),
            ]
        )

    seq = iaa.Sequential(
        [
            iaa.Resize({'height':height, 'width':width}),
            iaa.CropToFixedSize(400, 400),
            sometimes(iaa.Affine(
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
    
    seq2 = iaa.Sequential(
        [
            sometimes2(iaa.Affine(
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

    img_template= flip(img_template,flip_p)
    gt_template= flip(gt_template,flip_p)
    img_search= flip(img_search,flip_p)
    gt_search= flip(gt_search,flip_p)

    oh, ow = img_template.shape[:2]
    # process template
    #bb = cv2.boundingRect(gt_template)
    bb = scale_box(gt_template)
    if bb[2] != 0 and bb[3] != 0:
        for i in range(10):
            seq_ref_det = seq_ref.to_deterministic()
            #template = crop_and_padding(img_template, gt_template, (dim, dim))
            template = seq_ref_det.augment_image(img_template)
            gt_template_a = ia.SegmentationMapOnImage(gt_template, shape=gt_template.shape, nb_classes=2)
            gt_template_map = seq_ref_det.augment_segmentation_maps([gt_template_a])[0]
            template_mask = gt_template_map.get_arr_int().astype('uint8')
            template_mask= seq2.augment_segmentation_maps([gt_template_map])[0].get_arr_int().astype(float)

            #scale_p = random.uniform(0.7, 1.3)
            #template = cv2.resize(img_template, (dim, dim))
            #template_mask = np.expand_dims(cv2.resize(gt_template, (dim, dim)), 2)
            #bb = scale_box(template_mask, scale_p)
            bb = cv2.boundingRect(template_mask.astype('uint8'))
            template_mask= np.zeros([*template_mask.shape[:2], 1])
            if bb[2] >= 5 and bb[3] >= 5:
                template_mask[bb[1]:bb[1]+bb[3]+1, bb[0]:bb[0]+bb[2]+1, 0] = 1
                break
        else:
            template = cv2.resize(img_template, (dim, dim))
            template_mask = np.expand_dims(cv2.resize(gt_template, (dim, dim)), 2)
    else:
        template = np.zeros([dim, dim, 3]).astype('uint8')
        template_mask = np.zeros([dim, dim, 1])

    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB).astype(float)/255.
    
    # Augment Search Image
    if not no_crop:
        img_search_o = crop_and_padding(img_search, gt_search, (dim, dim)).astype('uint8')
        gt_search_o = crop_and_padding(gt_search, gt_search, (dim, dim)).astype('uint8')
    else:
        img_search_o = img_search.copy()
        gt_search_o = gt_search.copy().squeeze()

    for i in range(20):
        seq_det = seq.to_deterministic()
        img_search = seq_det.augment_image(img_search_o)
        img_search = cv2.cvtColor(img_search, cv2.COLOR_BGR2RGB).astype(float)/255.

        gt_search = ia.SegmentationMapOnImage(gt_search_o, shape=gt_search_o.shape, nb_classes=2)
        gt_search_map = seq_det.augment_segmentation_maps([gt_search])[0]
        gt_search = gt_search_map.get_arr_int()
        bb = cv2.boundingRect(gt_search.astype('uint8'))
        if bb[2] > 10 and bb[3] > 10:
            break
    else:
        img_search = cv2.resize(img_search_o, (dim, dim))
        img_search = cv2.cvtColor(img_search, cv2.COLOR_BGR2RGB).astype(float)/255.
        gt_search = cv2.resize(gt_search_o, (dim, dim))

        
    for i in range(10):
        mask = seq2.augment_segmentation_maps([gt_search_map])[0].get_arr_int().astype(float)
        bb = cv2.boundingRect(gt_search.astype('uint8'))
        if bb[2] > 10 and bb[3] > 10:
            break
    else:
        mask = gt_search.copy()
 
    fc = np.zeros([dim, dim, 1])
    aug_p = random.uniform(0, 1)
    iter = random.randint(1, 5)
    if aug_p > 0.5:
        aug = np.expand_dims(cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5)), iterations=iter), 3)
    else:
        aug = np.expand_dims(cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5)), iterations=iter), 3)
    if flip_p <= 0.8:
        fc[np.where(aug==1)] = 1
    else:
        bb = cv2.boundingRect(mask.astype('uint8'))
        fc[bb[1]:bb[1]+bb[3]+1, bb[0]:bb[0]+bb[2]+1, 0] = 1

    gt_search= np.expand_dims(gt_search, 2)
    gt = np.expand_dims(gt_search, 3)
    #label = resize_label_batch(gt.astype(float), dim//2)
    label = gt.astype(float)
    #label = resize_label_batch(gt.astype(float), ow)
    label = label.squeeze(3)

    #print(img_search.shape)
    #print(fc.shape)
    #print(template.shape)
    #print(template_mask.shape)
    #print(label.shape)

    return img_search, fc, template, template_mask, label


def aug_mask_nodeform(img_template, img_search, gt_template, gt_search, p_mask):

    sometimes = lambda aug: iaa.Sometimes(0.8, aug)

    seq = iaa.Sequential(
        [
            sometimes(iaa.Affine(
                scale={"x": (2**(-1/8), 2**(1/8)), "y": (2**(-1/8), 2**(1/8))},
                translate_px={"x": (-8, 8), "y": (-8, 8)}, # translate by -20 to +20 percent (per axis)
                cval=(0, 0), # if mode is constant, use a cval between 0 and 255
                mode='edge' # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            iaa.Add((-10, 10), per_channel=0.5),
        ], random_order=True
        )
    
    scale=1
    dim = int(scale*328)
    
    # Create Template Image
    flip_p = random.uniform(0, 1)
    img_template= flip(img_template,flip_p)
    gt_template= flip(gt_template,flip_p)

    img_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2RGB)
    target = crop_and_padding(img_template, gt_template, (dim, dim))
    mask = crop_and_padding(gt_template, gt_template, (dim, dim))

    seq_det = seq.to_deterministic()
    target = seq_det.augment_image(target).astype(float)/255.
    mask_map = ia.SegmentationMapOnImage(mask, shape=mask.shape, nb_classes=2)
    mask= seq_det.augment_segmentation_maps([mask_map])[0].get_arr_int()
    bb = cv2.boundingRect(mask.astype('uint8'))
    template_mask= np.zeros(mask.shape)
    template_mask[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]] = 1

    # Create Search Image
    flip_p = random.uniform(0, 1)
    img_search = flip(img_search,flip_p)
    gt_search = flip(gt_search,flip_p)
    p_mask = flip(p_mask, flip_p)
    
    img_search = cv2.cvtColor(img_search, cv2.COLOR_BGR2RGB)
    img_search_o = crop_and_padding(img_search, p_mask, (dim, dim))
    gt_search_o = crop_and_padding(gt_search, p_mask, (dim, dim))
    p_mask_o = crop_and_padding(p_mask, p_mask, (dim, dim))

    if len(np.unique(gt_search_o)) == 1:
        img_search_o = crop_and_padding(img_search, gt_search, (dim, dim))
        gt_search_o = crop_and_padding(gt_search, gt_search, (dim, dim))
        p_mask = gt_search
    for i in range(10):
        seq_det = seq.to_deterministic()
        img_search = seq_det.augment_image(img_search_o).astype(float)/255.

        gt_searchmap = ia.SegmentationMapOnImage(gt_search_o, shape=gt_search.shape, nb_classes=2)
        gt_search = seq_det.augment_segmentation_maps([gt_searchmap])[0].get_arr_int().astype(float)
        mask_map = ia.SegmentationMapOnImage(p_mask_o, shape=p_mask.shape, nb_classes=2)
        p_mask= seq_det.augment_segmentation_maps([mask_map])[0].get_arr_int().astype(float)
        if bb[2] > 10 and bb[3] > 10:
            break

    
    p_mask = np.expand_dims(cv2.dilate(p_mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)), iterations=2), 2)
    template_mask= np.expand_dims(template_mask, 2)

    #image = np.dstack([img_search, p_mask])
    gt_search = np.expand_dims(gt_search, 2)
    gt_search = np.expand_dims(gt_search, 3)
    label = resize_label_batch(gt_search, dim//2)
    label = label.squeeze(3)

    return img_search, fc, template, template_mask, label

