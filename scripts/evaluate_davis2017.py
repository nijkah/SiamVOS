import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn

from collections import OrderedDict
import os
import sys
import json
import time
sys.path.append('..')

from models import siamvos
from dataloader.datasets import DAVIS_eval
from tools.utils import *
from cfg import Config

cfg = Config()

DAVIS_PATH= cfg.DAVIS17_PATH
# THIS IS BEST 36 + 1
#SAVED_DICT_PATH = '/data/hakjin-workspace/snapshots/stm_2_sequential-6200.pth'
# THIS IS SECOND-BEST 40 + 1
#SAVED_DICT_PATH = '/data/hakjin-workspace/snapshots/stm_2_sequential-6600.pth'
#SAVED_DICT_PATH = '/data/hakjin-workspace/snapshots/stm_2_sequential-3700.pth'
SAVED_DICT_PATH = '../data/trained_SiamVOS_new.pth'
palette = Image.open(DAVIS_PATH+ 'Annotations/480p/bear/00000.png').getpalette()
SAVE_PATH = '../data/eval/'

def infer_SO(frames, masks, num_frames):
    # t, n, 2, h, w
    t, n, o, h, w = masks.size()
    predicts = torch.zeros(t, n, o+1, h, w)

    mask_prev = masks[0]
    bb = cv2.boundingRect(mask_prev.squeeze().data.numpy().astype('uint8'))
    box_ref = torch.zeros(mask_prev.size())
    box_ref[:, :, bb[1]:bb[1]+bb[3]+1, bb[0]:bb[0]+bb[2]+1] = 1

    img_ref = frames[0]
    img_prev = img_ref.clone()
    mask_prev = box_ref.clone()

    with torch.no_grad():
        for t in range(num_frames):
            img = frames[t]
            output, _ = model(img.cuda(), mask_prev.cuda(), img_prev.cuda(), img_ref.cuda(), box_ref.cuda())
            pred = F.softmax(output, dim=1)[:, 1].data.cpu()
            predicts[t, :, 1] = pred
            predicts[t, :, 0] = 1- pred

            img_prev = img.clone()
            mask_prev = torch.argmax(output, dim=1, keepdim=True).float()
    return predicts

def infer_MO(frames, masks, num_frames, num_objects):
    if num_objects == 1:
        predicts = infer_SO(frames, masks, num_frames)
        return predicts

    # t, n, o, h, w
    t, n, o, h, w = masks.size()
    predicts = torch.zeros(t, n, o+1, h, w)

    img_ref = frames[0]

    with torch.no_grad():
        for o in range(num_objects):
            mask_prev = masks[0, :, o:o+1]
            bb = cv2.boundingRect(mask_prev.squeeze().data.numpy().astype('uint8'))
            box_ref = torch.zeros(mask_prev.size())
            box_ref[:, :, bb[1]:bb[1]+bb[3]+1, bb[0]:bb[0]+bb[2]+1] = 1

            img_prev = img_ref.clone()
            mask_prev = box_ref.clone()

            for t in range(num_frames):
                img = frames[t]
                output, _ = model(img.cuda(), mask_prev.cuda(), img_prev.cuda(), img_ref.cuda(), box_ref.cuda())
                pred = F.softmax(output, dim=1)[:, 1].data.cpu()
                predicts[t, :, o+1:o+2] = pred

                img_prev = img.clone()
                mask_prev = torch.argmax(output, dim=1, keepdim=True).float()
        
        for t in range(num_frames):
            # for background
            bg = torch.max(predicts[t, :, 1:], dim = 1, keepdim=True)[0]
            #bg = torch.mean(predicts[t, :, 1:], dim = 1, keepdim=True)
            predicts[t, :, 0:1] = 1- bg

            predicts[t] = F.softmax(predicts[t], dim=1)

    return predicts


def eval(model, testLoader, name='siamvos_best9'):
    model.eval()
    for seq, (frames, masks, info) in enumerate(testLoader):
        seq_name = info['name']
        num_frames = info['num_frames']
        num_objects = info['num_objects']

        tt = time.time()
        predicts = infer_MO(frames, masks, num_frames, num_objects)
        print('{} | num_objects: {}, FPS: {}'.format(seq_name, num_objects, num_frames/(time.time()-tt)))

        # Save results for quantitative eval 
        if testLoader.MO:
            folder = os.path.join(SAVE_PATH, 'MO_'+name)
        else:
            folder = os.path.join(SAVE_PATH, 'SO_'+name)

        test_path = os.path.join(folder, seq_name)
        if not os.path.exists(test_path):
            os.makedirs(test_path)

        for t in range(num_frames):
            pred = predicts[t,0].numpy()
            # make hard label
            pred = np.argmax(pred, axis=0).astype(np.uint8)
            #E = ToLabel(E)

            img_pred = Image.fromarray(pred)
            img_pred.putpalette(palette)
            img_pred.save(os.path.join(test_path, '{:05d}.png'.format(t)))


if __name__ == '__main__':
    model = siamvos.build_siamvos(2)
    state_dict = torch.load(SAVED_DICT_PATH)
    model.load_state_dict(state_dict['model'])
    model = model.cuda()
    model.eval()

    testLoader = DAVIS_eval(root=DAVIS_PATH, imset='2017/val.txt',multi_object=True)
    eval(model, testLoader)
