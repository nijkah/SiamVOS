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
#SAVED_DICT_PATH = '../data/snapshots/fuse1-10000.pth'
#SAVED_DICT_PATH = '../../EXO-Tracker/triple-44000.pth'
SAVED_DICT_PATH = '../data/snapshots/siam_bptt-10.pth'
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
    #return torch.cat([1-predicts, predicts], dim=1)

def eval(model, testLoader):
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
            folder = 'results/MO_bptt'
        else:
            folder = 'results/SO_bptt'    

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

"""
def test_model(model, vis=False, save=True, name='triple'):
    dim = 328
    model.eval()
    #model.train()
    with open(os.path.join(DAVIS_PATH, 'ImageSets/480p', 'val.txt')) as f:
        files = f.readlines()
    val_seqs = sorted(list(set([i.split('/')[3] for i in files])))
    dumps = OrderedDict()


    pytorch_list = []
    tiou = 0
    for seq in val_seqs:
        seq_path = os.path.join(im_path, seq)
        img_list = [os.path.join(seq, i)[:-4] for i in sorted(os.listdir(seq_path))]

        seq_iou = 0
        for idx, i in enumerate(img_list):

            img_original = cv2.imread(os.path.join(im_path,i+'.jpg'))
            img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
            img_temp = img_original.copy().astype(float)/255.
            #img_temp = cv2.resize(img_temp, (400, 400))
            oh, ow = img_temp.shape[:2]
            
            gt_original = cv2.imread(os.path.join(gt_path,i+'.png'),0)
            #gt_original =cv2.resize(gt_original, (400, 400))
            gt_original[gt_original==255] = 1   

            if idx == 0:
                bb = cv2.boundingRect(gt_original)
                template = img_temp.copy()
                #template = crop_and_padding(img_temp, gt_original, (dim, dim))
                box = np.zeros([oh, ow, 1])
                if bb is not None:
                    box[bb[1]:bb[1]+bb[3]+1, bb[0]:bb[0]+bb[2]+1, 0] = 1
                    template = np.expand_dims(template, 0).transpose(0,3,1,2)
                    template = torch.FloatTensor(template).cuda()
                    box = np.expand_dims(box, 0).transpose(0,3,1,2)
                    dummy = torch.FloatTensor(box.copy())
                    box = torch.FloatTensor(box).cuda()
                previous = gt_original.copy()
                bb = cv2.boundingRect(previous)
                previous = np.zeros(gt_original.shape).astype('uint8')
                previous[bb[1]:bb[1]+bb[3]+1, bb[0]:bb[0]+bb[2]+1]= 1
                img_prev = template
            
            search_region = img_temp.copy() 
            mask = previous.copy()
            #if mask.sum() == 0:
            #    mask = np.ones(previous.shape)
            #inp = np.dstack([search_region, mask])
            image = torch.FloatTensor(np.expand_dims(search_region,0).transpose(0,3,1,2)).cuda()
            mask = torch.FloatTensor(mask[np.newaxis, :, :, np.newaxis].transpose(0,3,1,2)).cuda()


            #output = model(template, torch.FloatTensor(np.expand_dims(inp, 0).transpose(0,3,1,2)).cuda(gpu0))
            output, _ = model(image, mask, img_prev, template, box, dummy.cuda())
            #pred_c = output.data.cpu().numpy()
            #pred_c = output.data.cpu().numpy()
            #pred_c[pred_c>0.5] = 1
            #pred_c[pred_c!=1] = 0
            #pred_c = scipy.misc.imresize(pred_c.astype('uint8'), (321, 321))
            #pred_c = scipy.misc.imresize(pred_c.astype('uint8').squeeze(), (321, 321))
            #print(output.shape)
            #pred_c = F.upsample(output, scale_factor=2).data.cpu().numpy()
            #pred_c = F.interpolate(output, size=(dim,dim), mode='bilinear', align_corners=True).data.cpu().numpy()
            pred_c = output.data.cpu().numpy()
            pred_c = pred_c.squeeze(0).transpose(1,2,0)
            #pred_c = np.argmax(pred_c,axis = 2)

            #bb = list(cv2.boundingRect(previous.astype('uint8')))
            #bb = list(cv2.boundingRect(gt_original.astype('uint8')))
            #pred = crop_and_padding(gt_original, previous, (dim, dim))
            #pred = crop_and_padding(gt_original, gt_original, (dim, dim))
            #pred = restore_mask(pred_c, bb, img_original.shape)
            pred = np.argmax(pred_c, axis=2).astype('uint8')
            #pred[pred>0.5]=1
            #pred[pred!=1]=0
                        
            plt.ion()
            if vis:
                plt.subplot(2, 2, 1)
                plt.imshow(img_original)
                plt.subplot(2, 2, 2)
                plt.title('previous mask')
                plt.imshow(previous)
                plt.subplot(2, 2, 3)
                plt.title('gt - pred')
                bg = np.zeros([*gt_original.shape[:2], 3])
                bg[:, :, 0] = gt_original*255
                bg[:,:, 1 ] = pred*255

                plt.imshow(bg.astype('uint8'))
                plt.subplot(2, 2, 4)
                output = output.data.cpu().numpy().squeeze()
                output = np.argmax(output, 0)
                plt.imshow(output.astype('uint8'))
                #plt.subplot(2, 2, 4)
                #plt.title('prediction')
                #plt.imshow(pred)
                plt.show()
                plt.pause(.0001)
                plt.clf()
            
            previous = pred
            img_prev = image

            
            iou = get_iou(previous, gt_original.squeeze(), 0)

            if save:
                save_path = 'Result_'+name
                folder = os.path.join(save_path, i.split('/')[0])
                if not os.path.isdir(folder):
                    os.makedirs(folder)
                cv2.imwrite(os.path.join(save_path, i+'.png'),previous*255)
            seq_iou += iou

        print(seq, seq_iou/len(img_list))
        dumps[seq] = seq_iou/len(img_list)
        tiou += seq_iou/len(img_list)
    #miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    #print 'pytorch',iter,"Mean iou = ",np.sum(miou)/len(miou)
    print('total:', tiou/len(val_seqs))
    model.train()
    dumps['t mIoU'] = tiou/len(val_seqs)
    with open('dump_'+name+'bptt.json', 'w') as f:
        json.dump(dumps, f, indent=2)

    return tiou/len(val_seqs)
"""

if __name__ == '__main__':
    model = siam_deeplab_up.build_siam_Deeplab(2)
    #state_dict = torch.load('data/snapshots/DAVIS16-20000.pth')
    state_dict = torch.load(SAVED_DICT_PATH)
    model.load_state_dict(state_dict['model'])
    model = model.cuda()
    model.eval()
    #testLoader = DAVIS_eval(root=DAVIS_PATH, imset='2016/val.txt',multi_object=False)
    testLoader = DAVIS_eval(root=DAVIS_PATH, imset='2017/val.txt',multi_object=True)
    eval(model, testLoader)
