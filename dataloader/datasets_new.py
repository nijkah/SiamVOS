import torch
import torch.utils.data as data

import os, math, random
from os.path import *
import json
import numpy as np

from glob import glob
import cv2
from PIL import Image
from dataloader.custom_transforms.trs_utils import ImgAugBatch
from configs import Config as cfg

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

class DAVIS(data.Dataset):
    def __init__(self, train=False, root = '', aug=False):
        self.aug = aug
        self.cfg = cfg()

        if train:
            seqs_file = 'train.txt'
        else:
            seqs_file = 'val.txt'

    def __getitem__(self, index):

        index = index % self.size
        seq = self.seq_id_list[index]
        index_list = [i for i, x in enumerate(self.seq_id_list) if x == seq]
        index_list.remove(index)
        search_index = random.choice(index_list)

        i_index = index_list.index(search_index)
        candidates = index_list[i_index-3:i_index] + index_list[i_index+1:i_index+4]
        mask_index = random.choice(candidates)

        img_ref = cv2.imread(self.image_list[index])
        img_search = cv2.imread(self.image_list[search_index])

        gt_ref = np.expand_dims(np.array(Image.open(self.gt_list[index])), axis=2)
        gt_search= np.expand_dims(np.array(Image.open(self.gt_list[search_index])), axis=2)
        mask = np.expand_dims(np.array(Image.open(self.gt_list[mask_index])), axis=2)
        gt_ref[gt_ref==255] = 1
        gt_search[gt_search==255] = 1
        mask[mask==255] = 1
        
        if self.aug:
            img_search, p_mask, img_prev, img_ref, mask_ref, gt = aug_batch(img_ref, gt_ref, img_search, 
                    gt_search, no_crop=True)
            
        # hwc
        img_search = img_search.transpose(2, 0, 1)
        p_mask = p_mask.transpose(2, 0, 1)
        img_prev = img_prev.transpose(2, 0, 1)
        img_ref = img_ref.transpose(2, 0, 1)
        mask_ref = mask_ref.transpose(2, 0, 1)
        gt = gt.transpose(2, 0, 1)

        img_search = torch.from_numpy(img_search.astype(np.float32))
        p_mask = torch.from_numpy(p_mask.astype(np.float32))
        img_prev = torch.from_numpy(img_prev.astype(np.float32))
        img_ref = torch.from_numpy(img_ref.astype(np.float32))
        mask_ref = torch.from_numpy(mask_ref.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        return img_search, p_mask, img_prev, img_ref, mask_ref, gt

    def __len__(self):
        return self.size

class DAVIS2016(DAVIS):
    def __init__(self, train=False, root = '', aug=False):
        super(DAVIS2016, self).__init__(train=train, root = root, aug=aug)
        self.cfg = cfg()
        if train:
            seqs_file = 'train.txt'
        else:
            seqs_file = 'val.txt'

        with open(join(root, 'ImageSets/480p', seqs_file)) as f:
            files = f.readlines()

        self.image_list = []
        self.gt_list = []
        self.seq_id_list = []
        for f in files:
            im, gt = f.split()
            seq = im.split('/')[-2]
            img = join(root, im[1:])
            gt = join(root, gt[1:])
            self.image_list += [img]
            self.gt_list += [gt]
            gt_im = np.array(Image.open(gt))
            if len(np.unique(gt_im)) == 1:
                continue
            self.image_list += [img]
            self.gt_list += [gt]
            self.seq_id_list += [seq]

        self.size = len(self.image_list)
        self.frame_size = cv2.imread(self.image_list[0]).shape
       
        assert (len(self.image_list) == len(self.gt_list))


class DAVIS2017(data.Dataset):
    def __init__(self, train=False, root = '', aug=False):
        self.aug = aug
        self.cfg = cfg()

        if train:
            seqs_file = 'train.txt'
        else:
            seqs_file = 'val.txt'

        seq_list = sorted(np.loadtxt(join(root, 'ImageSets/2017', seqs_file), dtype=str).tolist())

        image_root = os.path.join(root, 'JPEGImages/480p')
        gt_root = os.path.join(root, 'Annotations/480p')

        self.image_list = []
        self.gt_list = []
        self.seq_id_list = []
        for seq in seq_list:
            files = sorted(os.listdir(join(image_root, seq)))
            for i in range(len(files)):
                img = join(image_root, seq, files[i])
                gt = join(gt_root, seq, files[i][:-4]+'.png')
                self.image_list += [img]
                self.gt_list += [gt]
                self.seq_id_list += [seq]

        self.size = len(self.image_list)
        self.frame_size = cv2.imread(self.image_list[0]).shape
       
        assert (len(self.image_list) == len(self.gt_list))

    def __getitem__(self, index):

        flag = True 

        index = index % self.size
        seq = self.seq_id_list[index]
        index_list = [i for i, x in enumerate(self.seq_id_list) if x == seq]

        for index in index_list:
            gt_ref_o = np.array(Image.open(self.gt_list[index]))
            if len(np.unique(gt_ref_o)) == 1:
                continue

            search_center = random.choice(index_list)
            i_index = index_list.index(search_center)
            index_list.remove(search_center)
            candidates = index_list[i_index-2:i_index] + index_list[i_index+1:i_index+3]
            random.shuffle(candidates)

            for search_index in candidates:
                gt_o = np.array(Image.open(self.gt_list[search_index]))
                if len(np.unique(gt_o)) == 1:
                    continue

                labels_ref = np.unique(gt_ref_o).tolist()
                labels = np.unique(gt_o).tolist()
                labels = list(set(labels_ref).intersection(set(labels)))
                if len(labels) == 0:
                    continue
                random.shuffle(labels)
                for l in labels:
                    if l == 0 :
                        continue
                    gt_ref = gt_ref_o.copy()
                    gt = gt_o.copy()
                    gt_ref[gt_ref!=l] = 0
                    gt[gt!=l] = 0
                    gt_ref[gt_ref==l] = 1
                    gt[gt==l] = 1
                    if (np.unique(gt_ref) == np.unique(gt)).all():
                        img_ref  = Image.open(self.image_list[index])
                        img = Image.open(self.image_list[search_index])
                        break
                else:
                    continue
                break
            else:
                continue
            break
        aug = ImgAugBatch(cfg=self.cfg)
        gt, gt_ref = Image.fromarray(gt), Image.fromarray(gt_ref)

        if self.aug:
            img_search, p_mask, img_prev, img_ref, mask_ref, gt = aug(img_ref, gt_ref, img, gt)
            
        # hwc
        img_search = img_search.transpose(2, 0, 1)
        p_mask = p_mask.transpose(2, 0, 1)
        img_prev = img_prev.transpose(2, 0, 1)
        img_ref = img_ref.transpose(2, 0, 1)
        mask_ref = mask_ref.transpose(2, 0, 1)
        gt = gt.transpose(2, 0, 1)

        img_search = torch.from_numpy(img_search.astype(np.float32))
        p_mask = torch.from_numpy(p_mask.astype(np.float32))
        img_prev = torch.from_numpy(img_prev.astype(np.float32))
        img_ref = torch.from_numpy(img_ref.astype(np.float32))
        mask_ref = torch.from_numpy(mask_ref.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        return img_search, p_mask, img_prev, img_ref, mask_ref, gt
    
    def __len__(self):
        return self.size

class DAVIS_small(DAVIS2017):
    def __init__(self, train=False, root = '', aug=False):
        super(DAVIS2017, self).__init__()
        self.aug = aug
        self.cfg = cfg()

        if train:
            seqs_file = 'train.txt'
        else:
            seqs_file = 'val.txt'
        
        meta_file = 'davis_semantics.json'
        cat_file= 'categories.json'
        with open(os.path.join(root, cat_file)) as f:
            cat_info = json.load(f)
        with open(os.path.join(root, meta_file)) as f:
            meta_info = json.load(f)
        labels = [l for l in cat_info.keys() if cat_info[l]['super_category'] in self.cfg.DAVIS_SMALL_CATEGORIES]

        valid_seqs = []
        for seq in meta_info:
            instances = [meta_info[seq][id] for id in meta_info[seq]]
            are_valid = [True if i in labels else False for i in instances]
            if False in are_valid:
                continue
            valid_seqs.append(seq)

        seq_list = sorted(np.loadtxt(join(root, 'ImageSets/2017', seqs_file), dtype=str).tolist())
        seq_list = [seq for seq in seq_list if seq in valid_seqs]

        image_root = os.path.join(root, 'JPEGImages/480p')
        gt_root = os.path.join(root, 'Annotations/480p')

        self.image_list = []
        self.gt_list = []
        self.seq_id_list = []
        for seq in seq_list:
            files = sorted(os.listdir(join(image_root, seq)))
            for i in range(len(files)):
                img = join(image_root, seq, files[i])
                gt = join(gt_root, seq, files[i][:-4]+'.png')
                self.image_list += [img]
                self.gt_list += [gt]
                self.seq_id_list += [seq]

        self.size = len(self.image_list)
        self.frame_size = cv2.imread(self.image_list[0]).shape
       
        assert (len(self.image_list) == len(self.gt_list))


class YTB_VOS(data.Dataset):
    def __init__(self, train=False, root = '', aug=False):
        self.aug = aug
        self.cfg = cfg() 

        if train:
            seq = 'train'
        else:
            seq = 'valid'

        image_root = join(root, seq, 'JPEGImages')
        gt_root = join(root, seq, 'Annotations')


        self.image_list = []
        self.gt_list = []
        self.seq_id_list = []
        for seq in os.listdir(image_root):
            files = sorted(os.listdir(join(image_root, seq)))
            for i in range(len(files)):
                img = join(image_root, seq, files[i])
                gt = join(gt_root, seq, files[i][:-4]+'.png')
                self.image_list += [img]
                self.gt_list += [gt]
                self.seq_id_list += [seq]

        self.size = len(self.image_list)
        self.frame_size = cv2.imread(self.image_list[0]).shape

        assert (len(self.image_list) == len(self.gt_list))


    def __getitem__(self, index):

        index = index % self.size
        seq = self.seq_id_list[index]
        index_list = [i for i, x in enumerate(self.seq_id_list) if x == seq]

        for index in index_list:
            gt_ref_o = np.array(Image.open(self.gt_list[index]))
            if len(np.unique(gt_ref_o)) == 1:
                continue
            search_center = random.choice(index_list)
            i_index = index_list.index(search_center)
            index_list.remove(search_center)
            candidates = index_list[i_index-2:i_index] + index_list[i_index+1:i_index+3]
            random.shuffle(candidates)

            for search_index in candidates:
                gt_o = np.array(Image.open(self.gt_list[search_index]))
                if len(np.unique(gt_o)) == 1:
                    continue

                labels_ref = np.unique(gt_ref_o).tolist()
                labels = np.unique(gt_o).tolist()
                labels = list(set(labels_ref).intersection(set(labels)))
                if len(labels) == 0 :
                    continue
                random.shuffle(labels)
                for l in labels:
                    if l == 0 :
                        continue
                    gt_ref = gt_ref_o.copy()
                    gt = gt_o.copy()
                    gt_ref[gt_ref!=l] = 0
                    gt[gt!=l] = 0
                    gt_ref[gt_ref==l] = 1
                    gt[gt==l] = 1
                    if (np.unique(gt_ref) == np.unique(gt)).all():
                        img_ref  = Image.open(self.image_list[index])
                        img = Image.open(self.image_list[search_index])
                        break
                else:
                    continue
                break
            else:
                continue
            break
        aug = ImgAugBatch(cfg=self.cfg)
        gt, gt_ref = Image.fromarray(gt), Image.fromarray(gt_ref)

        if self.aug:
            img_search, p_mask, img_prev, img_ref, mask_ref, gt = aug(img_ref, gt_ref, img, gt)
            
        # hwc
        img_search = img_search.transpose(2, 0, 1)
        p_mask = p_mask.transpose(2, 0, 1)
        img_prev = img_prev.transpose(2, 0, 1)
        img_ref = img_ref.transpose(2, 0, 1)
        mask_ref = mask_ref.transpose(2, 0, 1)
        gt = gt.transpose(2, 0, 1)

        img_search = torch.from_numpy(img_search.astype(np.float32))
        p_mask = torch.from_numpy(p_mask.astype(np.float32))
        img_prev = torch.from_numpy(img_prev.astype(np.float32))
        img_ref = torch.from_numpy(img_ref.astype(np.float32))
        mask_ref = torch.from_numpy(mask_ref.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        return img_search, p_mask, img_prev, img_ref, mask_ref, gt
        
    def __len__(self):
        return self.size

class YTB_VOS_small(YTB_VOS):
    def __init__(self, train=False, root = '', aug=False):
        super(YTB_VOS, self).__init__()
        self.aug = aug
        self.cfg = cfg()

        if train:
            seq = 'train'
            meta_file = 'meta.json'
            with open(os.path.join(root, seq, meta_file)) as f:
                meta_info = json.load(f)
        else:
            seq = 'valid'

        image_root = join(root, seq, 'JPEGImages')
        gt_root = join(root, seq, 'Annotations')
        valid_seqs = []
        for seq in meta_info['videos']:
            objects = meta_info['videos'][seq]['objects']
            are_valid = [True if objects[id]['category'] in self.cfg.YTBVOS_SMALL_CATEGORIES else False for id in objects]
            if False in are_valid:
                continue
            valid_seqs.append(seq)
        
        seq_list = os.listdir(image_root)
        seq_list = [seq for seq in seq_list if seq in valid_seqs]

        self.image_list = []
        self.gt_list = []
        self.seq_id_list = []
        for seq in seq_list:
            files = sorted(os.listdir(join(image_root, seq)))
            for i in range(len(files)):
                img = join(image_root, seq, files[i])
                gt = join(gt_root, seq, files[i][:-4]+'.png')
                self.image_list += [img]
                self.gt_list += [gt]
                self.seq_id_list += [seq]

        self.size = len(self.image_list)
        self.frame_size = cv2.imread(self.image_list[0]).shape

        assert (len(self.image_list) == len(self.gt_list))

class GyGO(data.Dataset):
    def __init__(self, train=False, root = '', aug=False):
        self.aug = aug
        self.cfg = cfg()

        if train:
            seqs_file = 'trainval.txt'
        else:
            seqs_file = 'trainval.txt'

        with open(join(root, 'ImageSets/', seqs_file)) as f:
            seqs = f.readlines()

        image_root = os.path.join(root, 'JPEGImages', '480p')
        gt_root = os.path.join(root, 'Annotations', '480p')

        self.image_list = []
        self.gt_list = []
        self.seq_id_list = []
        for seq in seqs:
            seq = seq.strip()
            files = sorted(os.listdir(join(image_root, seq)))
            for i in range(len(files)):
                img = join(image_root, seq, files[i])
                gt = join(gt_root, seq, files[i][:-4]+'.png')
                self.image_list += [img]
                self.gt_list += [gt]
                self.seq_id_list += [seq]
            
        self.size = len(self.image_list)
        self.frame_size = cv2.imread(self.image_list[0]).shape
       
        assert (len(self.image_list) == len(self.gt_list))

    def __getitem__(self, index):

        index = index % self.size
        seq = self.seq_id_list[index]
        index_list = [i for i, x in enumerate(self.seq_id_list) if x == seq]
        index_list.remove(index)
        search_index = random.choice(index_list)

        i_index = index_list.index(search_index)
        candidates = index_list[i_index-3:i_index] + index_list[i_index+1:i_index+4]
        mask_index = random.choice(candidates)

        img_ref = cv2.imread(self.image_list[index])
        img_search = cv2.imread(self.image_list[search_index])

        gt_ref = np.expand_dims(np.array(Image.open(self.gt_list[index]).convert('L')), axis=2)
        gt_search= np.expand_dims(np.array(Image.open(self.gt_list[search_index]).convert('L')), axis=2)
        gt_ref[gt_ref<128] = 0
        gt_ref[gt_ref!=0] = 1
        gt_search[gt_search<128] = 0
        gt_search[gt_search!=0] = 1

        if self.aug:
            img_search, p_mask, img_prev, img_ref, mask_ref, gt = aug_batch(img_ref, gt_ref, img_search, 
                    gt_search, no_crop=True)
            
        # hwc
        img_search = img_search.transpose(2, 0, 1)
        p_mask = p_mask.transpose(2, 0, 1)
        img_prev = img_prev.transpose(2, 0, 1)
        img_ref = img_ref.transpose(2, 0, 1)
        mask_ref = mask_ref.transpose(2, 0, 1)
        gt = gt.transpose(2, 0, 1)

        img_search = torch.from_numpy(img_search.astype(np.float32))
        p_mask = torch.from_numpy(p_mask.astype(np.float32))
        img_prev = torch.from_numpy(img_prev.astype(np.float32))
        img_ref = torch.from_numpy(img_ref.astype(np.float32))
        mask_ref = torch.from_numpy(mask_ref.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        return img_search, p_mask, img_prev, img_ref, mask_ref, gt

    def __len__(self):
        return self.size


