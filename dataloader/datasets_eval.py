import torch
import torch.utils.data as data

import os, math, random
from os.path import *
import json
import numpy as np

from glob import glob
import cv2
from PIL import Image

class DAVIS_train(data.Dataset):

    def __init__(self, root, imset='2017/train.txt', batch_frames=5, resize=(1280, 720), resolution='480p'):
        self.root = root
        self.batch_frames = batch_frames
        self.resize = resize
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob(os.path.join(self.image_dir, _video, '*.jpg')))
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)


    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        num_objects = self.num_objects[video]

        target_object = random.randint(1, num_objects)
        
        #                                 t,                  h,  w,       c
        #raw_frames = np.empty((self.batch_frames,)+self.shape[video]+(3,), dtype=np.float32)
        #raw_masks = np.empty((self.batch_frames,)+self.shape[video], dtype=np.uint8)
        raw_frames = np.empty((self.batch_frames,)+(self.resize[1], self.resize[0], 3,), dtype=np.float32)
        raw_masks = np.empty((self.batch_frames,)+(self.resize[1], self.resize[0], ), dtype=np.uint8)

        st_f = random.randint(0, info['num_frames']-self.batch_frames)
        #for i, f in enumerate(valid_frames[st_f:st_f+self.batch_frames]):
        for i, f in enumerate(range(st_f, st_f+self.batch_frames)):
            img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            raw_frames[i] = np.array(Image.open(img_file).convert('RGB').resize(self.resize, resample=Image.BILINEAR))/255.

            mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  #allways return first frame mask
            raw_mask = np.array(Image.open(mask_file).convert('P').resize(self.resize, resample=Image.BILINEAR), dtype=np.uint8)

            raw_masks[i] = (raw_mask == target_object).astype(np.uint8)

        ## make One-hot channel is object index
        #oh_masks = np.zeros((len(mask_frames),)+self.shape[video]+(num_objects,), dtype=np.uint8)
        #for o in range(num_objects):
        #    oh_masks[:,:,:,o] = (raw_masks == (o+1)).astype(np.uint8)
        
        # (t, h, w, c) - >  (t, n, c, h, w)
        th_frames = torch.from_numpy(np.transpose(raw_frames, (0, 3, 1, 2)).copy()).float()
        #th_masks = torch.unsqueeze(torch.from_numpy(np.transpose(raw_masks, (0, 3, 1, 2)).copy()).long(), 1)
        th_masks = torch.from_numpy(raw_masks).long()
        
        return th_frames, th_masks, info

class YVOS_train(data.Dataset):

    def __init__(self, root, imset='train', batch_frames=5, resize=(1280, 720)):
        self.root = root
        self.imset = imset
        self.batch_frames = batch_frames
        self.resize = resize
        self.mask_dir = os.path.join(root, imset, 'Annotations')
        self.image_dir = os.path.join(root, imset, 'JPEGImages')
        split_file = os.path.join(root, imset, 'meta.json')
        self.seq_info = dict()
        with open(split_file) as f:
                data = json.load(f)
                seqs = sorted(list(data['videos'].keys()))
                for seq in seqs:
                    sdict = data['videos'][seq]['objects']
                    objects = sdict.keys()
                    ob_info = dict((ob, sdict[ob]['frames']) for ob in objects)
                    self.seq_info[seq] = ob_info

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        videos = os.listdir(self.image_dir)
        for _video in sorted(videos):
            self.videos.append(_video)
            frames = glob(os.path.join(self.image_dir, _video, '*.jpg'))
            self.num_frames[_video] = len(frames)

            self.num_objects[_video] = len(self.seq_info[_video])
            self.shape[_video] = np.array(Image.open(os.path.join(self.mask_dir, _video, frames[0])).convert("P")).shape

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        while True:
            video = self.videos[index] % self.videos
            seq_info = self.seq_info[video]

            valid_objects = []
            for o in seq_info:
                frames = seq_info[o]
                if self.batch_frames <= len(frames):
                    valid_objects.append(o)
            
            if len(valid_objects) != 0:
                break
            index = random.randint(0, len(self.videos))

        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        num_objects = self.num_objects[video]

        target_object = random.choice(valid_objects)
        valid_frames = self.seq_info[video][target_object]

        # t, h, w, c
        #raw_frames = np.empty((self.batch_frames,)+self.shape[video]+(3,), dtype=np.float32)
        #raw_masks = np.empty((self.batch_frames,)+self.shape[video], dtype=np.uint8)
        raw_frames = np.empty((self.batch_frames,)+(self.resize[1], self.resize[0], 3,), dtype=np.float32)
        raw_masks = np.empty((self.batch_frames,)+(self.resize[1], self.resize[0], ), dtype=np.uint8)

        #for f in range(self.num_frames[video]):
        st_f = random.randint(0, len(valid_frames)-self.batch_frames)
        for i, f in enumerate(valid_frames[st_f:st_f+self.batch_frames]):
            img_file = os.path.join(self.image_dir, video, f+'.jpg')
            raw_frames[i] = np.array(Image.open(img_file).convert('RGB').resize(self.resize, resample=Image.BILINEAR))/255.

            mask_file = os.path.join(self.mask_dir, video, f+'.png')  #allways return first frame mask
            raw_mask = np.array(Image.open(mask_file).convert('P').resize(self.resize, resample=Image.BILINEAR), dtype=np.uint8)
            raw_masks[i] = (raw_mask == int(target_object)).astype(np.uint8)

        ## make One-hot channel is object index
        #oh_masks = np.zeros((len(mask_frames),)+self.shape[video]+(num_objects,), dtype=np.uint8)
        #for o in range(num_objects):
        #    oh_masks[:,:,:,o] = (raw_masks == (o+1)).astype(np.uint8)
        
        # (t, h, w, c) - >  (t, n, c, h, w)
        th_frames = torch.from_numpy(np.transpose(raw_frames, (0, 3, 1, 2)).copy()).float()
        #th_masks = torch.unsqueeze(torch.from_numpy(np.transpose(raw_masks, (0, 3, 1, 2)).copy()).long(), 1)
        th_masks = torch.from_numpy(raw_masks).long()
        
        return th_frames, th_masks, info


class DAVIS_eval(data.Dataset):

    def __init__(self, root, imset='2017/val.txt', resolution='480p', multi_object=False):
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob(os.path.join(self.image_dir, _video, '*.jpg')))
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)

        self.MO = multi_object

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        if self.MO:
            num_objects = self.num_objects[video]
        else:
            num_objects = 1
        info['num_objects'] = num_objects

        
        #                                 t,                  h,  w,       c
        raw_frames = np.empty((self.num_frames[video],)+self.shape[video]+(3,), dtype=np.float32)
        raw_masks = np.empty((self.num_frames[video],)+self.shape[video], dtype=np.uint8)
        for f in range(self.num_frames[video]):
            img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            raw_frames[f] = np.array(Image.open(img_file).convert('RGB'))/255.

            try:
                mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  #allways return first frame mask
                raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
            except:
                mask_file = os.path.join(self.mask_dir, video, '00000.png')
                raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)

            if self.MO:
                raw_masks[f] = raw_mask
            else:
                raw_masks[f] = (raw_mask != 0).astype(np.uint8)

            
        # make One-hot channel is object index
        oh_masks = np.zeros((self.num_frames[video],)+self.shape[video]+(num_objects,), dtype=np.uint8)
        for o in range(num_objects):
            oh_masks[:,:,:,o] = (raw_masks == (o+1)).astype(np.uint8)


        """
        # padding size to be divide by 32
        nf, h, w, _ = oh_masks.shape
        new_h = h + 32 - h % 32 new_w = w + 32 - w % 32
        # print(new_h, new_w)
        lh, uh = (new_h-h) / 2, (new_h-h) / 2 + (new_h-h) % 2
        lw, uw = (new_w-w) / 2, (new_w-w) / 2 + (new_w-w) % 2
        lh, uh, lw, uw = int(lh), int(uh), int(lw), int(uw)
        pad_masks = np.pad(oh_masks, ((0,0),(lh,uh),(lw,uw),(0,0)), mode='constant')
        pad_frames = np.pad(raw_frames, ((0,0),(lh,uh),(lw,uw),(0,0)), mode='constant')
        info['pad'] = ((lh,uh),(lw,uw))

        th_frames = torch.unsqueeze(torch.from_numpy(np.transpose(pad_frames, (3, 0, 1, 2)).copy()).float(), 0)
        th_masks = torch.unsqueeze(torch.from_numpy(np.transpose(pad_masks, (3, 0, 1, 2)).copy()).long(), 0)
        """

        # (t, h, w, c) - >  (t, n, c, h, w)
        th_frames = torch.unsqueeze(torch.from_numpy(np.transpose(raw_frames, (0, 3, 1, 2)).copy()).float(), 1)
        th_masks = torch.unsqueeze(torch.from_numpy(np.transpose(oh_masks, (0, 3, 1, 2)).copy()).long(), 1)
        
        return th_frames, th_masks, info

class YVOS_eval(data.Dataset):

    def __init__(self, root, imset='val'):
        self.root = root
        self.imset = imset
        self.mask_dir = os.path.join(root, imset, 'Annotations')
        self.image_dir = os.path.join(root, imset, 'JPEGImages')
        split_file = os.path.join(root, imset, 'meta.json')
        self.seq_info = dict()
        with open(split_file) as f:
                data = json.load(f)
                seqs = sorted(list(data['videos'].keys()))
                for seq in seqs:
                    sdict = data['videos'][seq]['objects']
                    objects = sdict.keys()
                    ob_info = list((ob, sdict[ob]['frames']))
                    self.seq_info[seq] = ob_info

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        videos = os.listdir(self.image_dir)
        for _video in sorted(videos):
            self.videos.append(_video)
            frames = glob(os.path.join(self.image_dir, _video, '*.jpg'))
            self.num_frames[_video] = len(frames)

            self.num_objects[_video] = len(self.seq_info[_video])
            self.shape[_video] = np.array(Image.open(os.path.join(self.mask_dir, _video, frames[0])).convert("P")).shape

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['num_objects'] = self.num_objects[video]
        info['objects_info'] = self.seq_info[video]

        # t, h, w, c
        raw_frames = np.empty((self.num_frames[video],)+self.shape[video]+(3,), dtype=np.float32)
        for f in range(self.num_frames[video]):
            img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            raw_frames[f] = np.array(Image.open(img_file).convert('RGB'))/255.

        mask_frames = os.path.join(self.mask_dir, video)
        raw_masks = np.empty((len(mask_frames),)+self.shape[video], dtype=np.uint8)
        for f in mask_frames:
            mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  
            raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)

        # make One-hot channel is object index
        oh_masks = np.zeros((len(mask_frames),)+self.shape[video]+(num_objects,), dtype=np.uint8)
        for o in range(num_objects):
            oh_masks[:,:,:,o] = (raw_masks == (o+1)).astype(np.uint8)

        
        # (t, h, w, c) - >  (t, n, c, h, w)
        th_frames = torch.unsqueeze(torch.from_numpy(np.transpose(raw_frames, (0, 3, 1, 2)).copy()).float(), 1)
        th_masks = torch.unsqueeze(torch.from_numpy(np.transpose(oh_masks, (0, 3, 1, 2)).copy()).long(), 1)
        
        return th_frames, th_masks, info

