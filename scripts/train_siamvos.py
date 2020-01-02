#48000
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim

import sys
import os
import argparse
import time
import json
sys.path.append('..')

from models import siamvos
from dataloader.datasets_triple import DAVIS2017, YTB_VOS, GyGO
from scripts.evaluate_siamvos import test_model
from tools.utils import *

DATASET_PATH = '/media/datasets/'
DAVIS_PATH = os.path.join(DATASET_PATH, 'DAVIS/')
VOS_PATH = os.path.join(DATASET_PATH, 'Youtube-VOS/')
GYGO_PATH = os.path.join(DATASET_PATH, 'GyGO-Dataset')
SAVED_DICT_PATH = os.path.join(DATASET_PATH, 'MS_DeepLab_resnet_trained_VOC.pth')

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    start = time.time()

    model = siamvos.build_siamvos(2, gc=True)
    saved_state_dict = torch.load(SAVED_DICT_PATH)

    model_dict = model.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)

    model.load_state_dict(model_dict)

    """
    saved_state_dict = torch.load('../data/snapshots/fuse1-13000.pth')
    model.load_state_dict(saved_state_dict)
    """

    model.cuda()

    db_davis_train = DAVIS2017(train=True,root=DAVIS_PATH, aug=True)
    db_ytb_train = YTB_VOS(train=True, root=VOS_PATH, aug=True)
    db_GyGO= GyGO(root=GYGO_PATH, aug=True)
    db_train = ConcatDataset([db_davis_train, db_ytb_train, db_GyGO])
    print(len(db_train))

    train_loader = DataLoader(db_train, batch_size=args.batchSize, shuffle=True)

    optimizer = optim.SGD([{'params': model.get_1x_lr_params_NOscale(), 'lr': args.lr},
                           {'params': model.get_10x_lr_params(), 'lr': 10*args.lr} ],
                            lr = args.lr, momentum = 0.9,weight_decay = args.wtDecay)
    #optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = base_lr, momentum = 0.9,weight_decay = weight_decay)

    lr = args.lr
    optimizer.zero_grad()


    losses = []
    acc = []
    best = 0
    numerics = {'loss':losses, 'acc': acc}
    iter = 0
    print(len(train_loader))

    if args.iter > 0 :
        sdict = torch.load('../data/snapshots/' + args.name+'-'+str(args.iter)+'.pth')
        model.load_state_dict(sdict['model'])
        optimizer.load_state_dict(sdict['optimizer'])
        iter = sdict['iter']

    start_t = time.time()
    for epoch in range(0, 21):
        for ii, sample in enumerate(train_loader):
            start_e = time.time()
            iter += 1
            if iter == args.maxIter:
                break

            img_search, p_mask, img_prev, img_ref, mask_ref, label = sample[0].cuda(), sample[1].cuda(), sample[2].cuda(), sample[3].cuda(), sample[4].cuda(), sample[5].cuda()

            out, loss = model(img_search, p_mask, img_prev, img_ref, mask_ref,  label)
            numerics['loss'].append(float(loss.data.cpu().numpy()))

            print('iter = ',iter, 'of',args.maxIter,'completed, loss = ', (loss.data.cpu().numpy()))

        
            if iter % 5 == 0:
                vis_2(img_search[0], p_mask[0], img_ref[0], mask_ref[0], label[0], out[0], iter)

            start_t = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('(poly lr policy) learning rate',lr)

            lr = lr_poly(args.lr, iter, args.maxIter,0.9)

            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = 10*lr

            if iter % 1000 == 0 and iter!=0:
                print('taking snapshot ...')
                torch.save({'iter':iter,
                            'model':model.state_dict(),
                            'optimizer':optimizer.state_dict()
                           },'../data/snapshots/'+args.name+'-'+str(iter)+'.pth')
                iou = test_model(model, save=True, name=args.name)
                numerics['acc'].append(iou)
                numerics['iters'] = iter
                with open('../data/losses/'+args.name+'-'+str(iter)+'.json', 'w') as f:
                    json.dump(numerics, f)
                if best < iou:
                    torch.save({'iter':iter,
                                'model':model.state_dict(),
                                'optimizer':optimizer.state_dict()
                            },'../data/snapshots/'+args.name+'-'+str(iter)+'.pth')
                    best = iou
            start_t = time.time()


    end = time.time()
    print('time taken ', end-start)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet-DeepLab on segmentation datasets in pytorch using VOC12\
        pretrained initialization')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--batchSize', '-b', type=int, default=7, help='Number of samples per batch')
    parser.add_argument('--wtDecay', type=float, default=0.0005, help='Weight decay during training')
    parser.add_argument('--gpu', type=int, default=0, help='GPU number')
    parser.add_argument('--maxIter', type=int, default=81000, help='Maximum number of iterations')
    parser.add_argument('--name', type=str, default='triplet_GC_re', help='save name')
    parser.add_argument('--iter', type=int, default=0, help='continue training')

    args = parser.parse_args()

    if not os.path.exists('../data/snapshots'):
        os.makedirs('../data/snapshots')
    if not os.path.exists('../data/losses'):
        os.makedirs('../data/losses')

    main(args)

