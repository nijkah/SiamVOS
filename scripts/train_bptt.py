import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import os
import sys
import json
import time
import random
import argparse
sys.path.append('..')

from models import siamvos
from dataloader.datasets_triple import DAVIS_eval
from tools.utils import *
from scripts.evaluate_siamvos import test_model

DAVIS_PATH= '/media/datasets/DAVIS/'
SAVED_DICT_PATH = '../data/snapshots/triplet_GC-26000.pth'
MODEL_DIR = '../data/losses'
palette = Image.open(DAVIS_PATH+ 'Annotations/480p/bear/00000.png').getpalette()

def get_box(mask):
    bb = cv2.boundingRect(mask.squeeze().data.numpy().astype('uint8'))
    box_ref = torch.zeros([1, 1, *mask.size()])
    box_ref[:, :, bb[1]:bb[1]+bb[3]+1, bb[0]:bb[0]+bb[2]+1] = 1

    return box_ref

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
            folder = 'results/MO'
        else:
            folder = 'results/SO'    

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

def bptt_hsm(data, hidden, target, model, criterion, bptt_len, bptt_step):
    hidden_v = hidden.detach()
    data_v = data.detach()
    hsm = { -1 : hidden.detach() }
    intervals = list(enumerate(range(0, data.size(0), bptt_step)))
    # Record states at selective intervals and flag the need for grads.
    # Note we don't need to forward the last interval as we'll do it below.
    # This loop is most of the extra computation for this approach.
    for f_i,f_v in intervals[:-1]:
        output,hidden_v = model(data_v[f_v:f_v+args.bptt_step], hidden_v)
        hsm[f_i] = hidden_v.detach()   

    save_grad=None
    loss = 0
    for b_i, b_v in reversed(intervals):
        output,h = model(data[b_v:b_v+args.bptt_step], hsm[b_i-1])
        iloss = criterion(output.view(-1, ntokens), 
            targets[b_v:b_v+args.bptt_step].view(-1))
        if b_v+args.bptt_step >= data.size(0):
            # No gradient from the future needed.
            # These are the hidden states for the next sequence.
            hidden = h
            iloss.backward()
        else:
            variables=[iloss]
            grad_variables=[None]   # scalar = None
            # Associate stored gradients with state variables for 
            # multi-variable backprop
            for l in h:
                variables.append(l)
                g = save_grad.popleft()
                grad_variables.append(g)
            torch.autograd.backward(variables, grad_variables)

        if b_i > 0:
            # Save the gradients left on the input state variables
            save_grad = collections.deque()
            for l in hsm[b_i-1]:
                # If this fails, could be a non-leaf, in which case exclude;
                # its grad will be handled by a leaf
                assert(l.grad is not None)  
                save_grad.append(l.grad)
        loss += iloss.data[0]

    av = 1/(args.batch_size*args.bptt)
    loss *= av
    for g in model.parameters():
        g.grad.data.mul_(av)

def train(model, trainLoader, args):
    model.train()
    model.freeze_bn()

    losses = {'loss_seq':[], 'loss_t':[]}

    #optimizer = optim.SGD([{'params': model.get_1x_lr_params_NOscale(), 'lr': args.lr},
    #                       {'params': model.get_10x_lr_params(), 'lr': 10*args.lr} ],
    #                        lr = args.lr , momentum = 0.9,weight_decay = args.wtDecay)
    #optimizer = optim.Adam([{'params': model.get_1x_lr_params_NOscale(), 'lr': args.lr},
    #                        {'params': model.get_10x_lr_params(), 'lr': 10*args.lr} ],
    #                         lr = args.lr)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    iters_per_epoch = len(trainLoader)
    writer = SummaryWriter()

    for epoch in range(0, args.num_epochs):
        # training
        print("Train epoch {}".format(epoch))
        for i, (frames, gts, info) in enumerate(trainLoader):
            optimizer.zero_grad()
            seq_name = info['name']
            num_frames = info['num_frames']
            num_objects = info['num_objects']

            if (args.bptt_len < num_frames):
                start_frame = random.randint(0, num_frames - args.bptt_len)
                frames = frames[start_frame:start_frame+args.bptt_len, ...]
                gts = gts[start_frame:start_frame+args.bptt_len, ...]

            tt = time.time()
            t, n, c, h, w = gts.shape
            predicts = torch.zeros(t, n, c+1, h, w)

            num_bptt = gts.shape[0]
            loss_t = 0
            counter_t = 0
            
            img_ref = frames[0]

            for o in range(c):
                box_ref = get_box(gts[0][:,o:o+1].squeeze())
                mask_prev = box_ref.clone()
                img_prev = img_ref.clone()
                loss = 0
                counter = 0
                for t in range(0, num_bptt):
                    img = frames[t]
                    gt = gts[t].float()
                    gt = gt[:, o:o+1]

                    output, loss_t = model(img.cuda(), mask_prev.cuda(), img_prev.cuda(), img_ref.cuda(), box_ref.cuda(), gt.cuda())
                    loss = loss + loss_t
                    counter += 1
                    output = output.detach()

                    pred = F.softmax(output, dim=1)[:, 1].data.cpu()
                    predicts[t, :, o+1:o+2] = pred.detach()

                    img_prev = img.clone()
                    mask_prev = torch.argmax(output, dim=1, keepdim=True).float()

                    if (t+1) % args.bptt_step == 0:
                        optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        optimizer.step()
                        if t < num_bptt - 2:
                            loss = 0
                            counter = 0
                if loss > 0:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                print('{} loss: {}'.format(seq_name, loss/counter))
                losses['loss_seq'].append(float((loss/counter).data.cpu().numpy()))
                loss_t = loss_t + loss.detach()
                counter_t += counter
            
            for t in range(num_bptt):
                # for background
                bg = torch.max(predicts[t, :, 1:], dim = 1, keepdim=True)[0]
                predicts[t, :, 0:1] = 1- bg
                predicts[t] = F.softmax(predicts[t], dim=1)

            if (i+1) % args.display_interval == 0:
                writer.add_scalar('Train/BCE', loss_t/counter_t, i + epoch*iters_per_epoch)
                print('loss: {}'.format(loss_t/counter_t))
                losses['loss_t'].append(float((loss_t/counter_t).data.cpu().numpy()))
                
            if epoch % 10 == 0 and epoch > 0:
                save_name = "{}/{}.pth".format(MODEL_DIR, epoch)

            if epoch % 3 == 0:
                folder = 'debugging_bptt/MO_'+str(epoch)
                test_path = os.path.join(folder, seq_name)
                if not os.path.exists(test_path):
                    os.makedirs(test_path)

                for t in range(num_bptt):
                    pred = predicts[t,0].numpy()
                    # make hard label
                    pred = np.argmax(pred, axis=0).astype(np.uint8)
                    #E = ToLabel(E)

                    img_pred = Image.fromarray(pred)
                    img_pred.putpalette(palette)
                    img_pred.save(os.path.join(test_path, '{:05d}.png'.format(t)))

        with open('../data/losses/'+args.name+'-'+str(epoch)+'.json', 'w') as f:
            json.dump(losses , f)

        if epoch % 5 == 0 and epoch > 0 :
            iou = test_model(model, save=True, name=args.name)
            torch.save({'iter':iter,
                        'model':model.state_dict(),
                        'optimizer':optimizer.state_dict()
                        },'../data/snapshots/'+args.name+'-'+str(epoch)+'.pth')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet-DeepLab on segmentation datasets in pytorch using VOC12\
        pretrained initialization')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate')
    parser.add_argument('--batchSize', '-b', type=int, default=7, help='Number of samples per batch')
    parser.add_argument('--wtDecay', type=float, default=0.0005, help='Weight decay during training')
    parser.add_argument('--gpu', type=int, default=0, help='GPU number')
    parser.add_argument('--name', type=str, default='siam_bptt', help='save name')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--bptt_len', type=int, default=12, help='bptt length')
    parser.add_argument('--bptt_step', type=int, default=4, help='bptt step')
    parser.add_argument('--display_interval', type=int, default=10, help='display interval')

    args = parser.parse_args()

    model = siamvos.build_siamvos(2, True)
    #state_dict = torch.load('data/snapshots/DAVIS16-20000.pth')
    state_dict = torch.load(SAVED_DICT_PATH)
    model.load_state_dict(state_dict['model'])
    model = model.cuda()
    trainLoader = DAVIS_eval(root=DAVIS_PATH, imset='2017/train.txt', multi_object=True)
    train(model, trainLoader, args)
