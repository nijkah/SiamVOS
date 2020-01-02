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
from dataloader.datasets_triple import DAVIS_eval
from tools.utils import *

DAVIS_PATH= '/media/datasets/DAVIS/'
#SAVED_DICT_PATH = '../data/trained_siamvos.pth'
SAVED_DICT_PATH = '../data/snapshots/triplet_GC-26000.pth'
palette = Image.open(DAVIS_PATH+ 'Annotations/480p/bear/00000.png').getpalette()

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

def infer_MO_enhanced(frames, masks, num_frames, num_objects):
    if num_objects == 1:
        predicts = infer_SO(frames, masks, num_frames)
        return predicts

    # t, n, o, h, w
    t, n, o, h, w = masks.size()
    predicts = torch.zeros(t, n, o+1, h, w)

    img_ref = frames[0]

    with torch.no_grad():
        mask_prevs = []
        box_refs = []
        img_prevs = []
        mask_prevs = []
        for o in range(num_objects):
            mask_prev = masks[0, :, o:o+1]
            bb = cv2.boundingRect(mask_prev.squeeze().data.numpy().astype('uint8'))
            box_ref = torch.zeros(mask_prev.size())
            box_ref[:, :, bb[1]:bb[1]+bb[3]+1, bb[0]:bb[0]+bb[2]+1] = 1
            img_prev = img_ref.clone()
            mask_prev = box_ref.clone()

            mask_prevs.append(mask_prev)
            box_refs.append(box_ref)
            img_prevs.append(img_prev)

        mask_prevs = torch.cat(mask_prevs, dim=0)
        box_refs = torch.cat(box_refs, dim=0)
        img_prevs = torch.cat(img_prevs, dim=0)
        img_refs = img_prevs.clone()

        for t in range(num_frames):
            img = frames[t]
            imgs = torch.cat([img]*num_objects, dim=0)
            output, _ = model(imgs.cuda(), mask_prevs.cuda(), img_prevs.cuda(), img_refs.cuda(), box_refs.cuda())
            #print(output.shape, mask_prevs.shape)
            #a = input('stop')
            pred = F.softmax(output, dim=1).data.cpu()[:, 1:]
            #print('out', pred.shape)
            bg = torch.prod(1-pred, dim=0)

            #predicts[t, :, o+1:o+2] = pred
            predicts[t, :, 1:] = pred.transpose(1, 0)
            #for o in range(num_objects):
            #    predicts[t, :, o+1:o+2] = pred[o]
            predicts[t, :, 0] = bg
            #bg = torch.max(predicts[t, :, 1:], dim = 1, keepdim=True)[0]
            #predicts[t, :, 0:1] = 1- bg

            img_prevs = imgs.clone()
            #mask_prevs = torch.argmax(output, dim=1, keepdim=True).float()
            #mask_prevs = torch.argmax(predicts[t], dim=1, keepdim=True).float()
            mask_prevs = (pred>0.5).float()
            #mask_prevs = torch.zeros(mask_prevs.shape)
            #mask_prevs =  (predicts[t]>0.50001).float().transpose(0, 1)
            print(predicts[t].shape, mask_prevs.shape)
            #for o in range(num_objects):
            #    mask_prevs[o] = (argmax_out==(o+1)).float()
                #predicts[t, :, o+1:o+2] = pred[o,1:2]
        
        #for t in range(num_frames):
            # for background
        #    bg = torch.max(predicts[t, :, 1:], dim = 1, keepdim=True)[0]
            #bg = torch.mean(predicts[t, :, 1:], dim = 1, keepdim=True)
        #    predicts[t, :, 0:1] = 1- bg

        #    predicts[t] = F.softmax(predicts[t], dim=1)

    return predicts

def eval(model, testLoader, name='siamvos'):
    model.eval()
    for seq, (frames, masks, info) in enumerate(testLoader):
        #all_F, all_M = all_F[0], all_M[0]
        seq_name = info['name']
        num_frames = info['num_frames']
        num_objects = info['num_objects']

        tt = time.time()
        predicts = infer_MO(frames, masks, num_frames, num_objects)
        print('{} | num_objects: {}, FPS: {}'.format(seq_name, num_objects, num_frames/(time.time()-tt)))

        # Save results for quantitative eval ######################
        if testLoader.MO:
            folder = 'results/MO_'+name
        else:
            folder = 'results/SO_'+name    

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
    model = siamvos.build_siamvos(2, True)
    state_dict = torch.load(SAVED_DICT_PATH)
    model.load_state_dict(state_dict['model'])
    #model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()
    #testLoader = DAVIS_eval(root=DAVIS_PATH, imset='2016/val.txt',multi_object=False)
    testLoader = DAVIS_eval(root=DAVIS_PATH, imset='2017/val.txt',multi_object=True)
    eval(model, testLoader)
