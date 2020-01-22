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
from torch.utils.data import DataLoader, ConcatDataset
sys.path.append('..')

from models import siamvos
#from dataloader.datasets_triple import DAVIS_eval
from dataloader.datasets_eval import DAVIS_train, YVOS_train
from scripts.evaluate_siamvos import test_model

from tools.utils import *
from cfg import Config

cfg = Config()

DAVIS_PATH= cfg.DAVIS17_PATH
YVOS_PATH= os.path.join(cfg.DATASET_PATH, 'Youtube-VOS')
MODEL_DIR = '../data/losses'
SAVED_DICT_PATH = '../data/trained_SiamVOS_new.pth'
palette = Image.open(DAVIS_PATH+ 'Annotations/480p/bear/00000.png').getpalette()

def mask2box(mask, scale=1.0):
    np_mask = mask.data.cpu().numpy()
    if len(np_mask.shape) == 4:
        np_mask = np.squeeze(np_mask, 1)

    #np_boxes = np.zeros(mask.shape, dtype=np.float32)
    ts_boxes = torch.zeros(mask.shape, dtype=torch.float32)
    for b in range(np_mask.shape[0]):
        all_ys, all_xs  = np.where(np_mask[b] >= 0.49)

        if all_ys.size == 0 or all_xs.size == 0:
            # if no pixel, return whole
            ymin, ymax = 0, np_mask.shape[1]
            xmin, xmax = 0, np_mask.shape[2]
        else:
            ymin, ymax = np.min(all_ys), np.max(all_ys)
            xmin, xmax = np.min(all_xs), np.max(all_xs)
        """
        # apply scale
        # y = (ymax + ymin) / 2.
        # x = (xmax + xmin) / 2.
        orig_h = ymax - ymin + 1
        orig_w = xmax - xmin + 1

        ymin = np.maximum(-5, ymin - (scale - 1) / 2. * orig_h)  
        ymax = np.minimum(np_mask.shape[1]+5, ymax + (scale - 1) / 2. * orig_h)    
        xmin = np.maximum(-5, xmin - (scale - 1) / 2. * orig_w)  
        xmax = np.minimum(np_mask.shape[2]+5, xmax + (scale - 1) / 2. * orig_w)  

        # final ywhw
        y = (ymax + ymin) / 2.
        x = (xmax + xmin) / 2.
        h = ymax - ymin + 1
        w = xmax - xmin + 1

        ts_boxes[b][y:y+h, x:x+w] = 1
        """
        ts_boxes[b][ymin:ymax, xmin:xmax] = 1
        
    return ts_boxes


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

        if epoch % 10 == 0 and epoch > 0 :
            torch.save({'iter':iter,
                        'model':model.state_dict(),
                        'optimizer':optimizer.state_dict()
                        },'../data/snapshots/'+args.name+'-'+str(epoch)+'.pth')

def train_sequential(model, db_train, args):
    model.train()

    numerics = {'loss_seq':[], 'loss_t':[], 'acc': []}

    optimizer = optim.SGD([{'params': model.get_1x_lr_params_NOscale(), 'lr': args.lr},
                           {'params': model.get_10x_lr_params(), 'lr': 10*args.lr} ],
                            lr = args.lr , momentum = 0.9,weight_decay = args.wtDecay)
    #optimizer = optim.Adam([{'params': model.get_1x_lr_params_NOscale(), 'lr': args.lr},
    #                        {'params': model.get_10x_lr_params(), 'lr': 10*args.lr} ],
    #                         lr = args.lr)
    #optimizer = optim.Adam(model.parameters(), lr = args.lr)

    writer = SummaryWriter()
    trainLoader = DataLoader(db_train, batch_size=args.batchSize, shuffle=True)
    iters_per_epoch = len(trainLoader)
    iter = 0
    best = 0
    
    if args.iter > 0:
        sdict = torch.load('../data/snapshots/'+ args.name+'_sequential-'+str(args.iter)+'.pth')
        model.load_state_dict(sdict['model'])
        optimizer.load_state_dict(sdict['optimizer'])
        iter = sdict['iter']

    for epoch in range(0, args.num_epochs):
        # training
        print("Train epoch {}".format(epoch))
        for i, (frames, gts, info) in enumerate(trainLoader):
            optimizer.zero_grad()
            seq_name = info['name']
            num_frames = info['num_frames']

            tt = time.time()

            loss_t = 0
            counter_t = 0

            #frames -> (n, t, c, h, w)
            #gts -> (n, t, h, w)
            img_refs = frames[:, 0]
            box_refs = mask2box(gts[:, 0]).unsqueeze(1)

            mask_prevs = box_refs.clone()
            img_prevs = img_refs.clone()
            
            loss_seq = 0
            for t in range(args.batch_frames):
                img_t = frames[:, t]
                gt_t = gts[:, t].float().unsqueeze(1)

                output, loss = model(img_t.cuda(), mask_prevs.cuda(), img_prevs.cuda(), img_refs.cuda(), box_refs.cuda(), gt_t.cuda())

                loss_t = float(loss.data.cpu().numpy())

                numerics['loss_t'].append(loss_t)
                print('iter = ',iter, ', loss = ', loss_t)
                loss_seq += loss_t

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                mask_prevs = torch.argmax(output.detach(), dim=1, keepdim=True).float()
                #pred_c = output.data.cpu().numpy()
                #pred_c = pred_c.squeeze(0).transpose(1,2,0)
                #pred = np.argmax(pred_c, axis=2).astype('uint8')
                img_prevs = img_t

                #lr = lr_poly(args.lr, iter, args.maxIter,0.9)
                #optimizer.param_groups[0]['lr'] = lr
                #optimizer.param_groups[1]['lr'] = 10*lr

                
            print('seq loss: {}'.format(loss_seq/args.batch_frames))
            numerics['loss_seq'].append(loss_seq/args.batch_frames)

            if iter % 100 == 0 and iter!=0:
                print('taking snapshot ...')
                torch.save({'iter':iter,
                            'model':model.state_dict(),
                            'optimizer':optimizer.state_dict()
                        },'../data/snapshots/'+args.name+'_sequential-'+str(iter)+'.pth')
                iou = test_model(model, save=True, name=args.name)
                numerics['acc'].append(iou)
                numerics['iters'] = iter
                with open('../data/losses/'+args.name+'_sequential-'+str(iter)+'.json', 'w') as f:
                    json.dump(numerics, f)

                if best < iou:
                    iou = best
                    torch.save({'iter':iter,
                                'model':model.state_dict(),
                                'optimizer':optimizer.state_dict()
                            },'../data/snapshots/'+args.name+'_sequential-best.pth')

            iter+=1
            
            """
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
            """






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet-DeepLab on segmentation datasets in pytorch using VOC12\
        pretrained initialization')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--batchSize', '-b', type=int, default=6, help='Number of samples per batch')
    parser.add_argument('--wtDecay', type=float, default=0.0005, help='Weight decay during training')
    parser.add_argument('--gpu', type=int, default=0, help='GPU number')
    parser.add_argument('--name', type=str, default='SiamVOS_seq', help='save name')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_frames', type=int, default=6, help='batch per each sequence')
    parser.add_argument('--iter', type=int, default=0, help='batch per each sequence')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


    model = siamvos.build_siamvos(2)
    state_dict = torch.load(SAVED_DICT_PATH)
    model.load_state_dict(state_dict['model'])
    model = model.cuda()
    #testLoader = DAVIS_eval(root=DAVIS_PATH, imset='2016/val.txt',multi_object=False)
    db_davis = DAVIS_train(root=DAVIS_PATH, batch_frames=args.batch_frames)
    db_yvos= YVOS_train(root=YVOS_PATH, batch_frames=args.batch_frames)
    db_train = ConcatDataset([db_davis,db_yvos])
    train_sequential(model, db_train, args)
    #train_sequential(model, db_yvos, args)
