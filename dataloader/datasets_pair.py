import torch
import torch.utils.data as data

import os, math, random
from os.path import *
import numpy as np

from glob import glob
import cv2
#from .custom_transforms_pair import aug_pair, aug_mask_nodeform
from dataloader.custom_transforms import aug_batch
from PIL import Image


class DAVIS(data.Dataset):
    def __init__(self, train=False, is_cropped = False, root = '', replicates = 1, aug=False):
        self.is_cropped = is_cropped
        self.replicates = replicates
        self.aug = aug

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

        img_template = cv2.imread(self.image_list[index])
        img_search = cv2.imread(self.image_list[search_index])

        gt_template = np.expand_dims(np.array(Image.open(self.gt_list[index])), axis=2)
        gt_search= np.expand_dims(np.array(Image.open(self.gt_list[search_index])), axis=2)
        mask = np.expand_dims(np.array(Image.open(self.gt_list[mask_index])), axis=2)
        gt_template[gt_template==255] = 1
        gt_search[gt_search==255] = 1
        mask[mask==255] = 1
        
        if self.aug:
            img, mask, target, box, gt = aug_pair(img_template, img_search, gt_template,
                    gt_search, no_crop=True)
            #img, target, gt = aug_mask(img_template, img_search, gt_template, gt_search, mask)
            
        # hwc

        img = img.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        target = target.transpose(2, 0, 1)
        box = box.transpose(2, 0, 1)
        gt = gt.transpose(2, 0, 1)

        img = torch.from_numpy(img.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))
        target = torch.from_numpy(target.astype(np.float32))
        box = torch.from_numpy(box.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        return img, mask, target, box, gt

    def __len__(self):
        return self.size * self.replicates

class DAVIS2016(DAVIS):
    def __init__(self, train=False, is_cropped = False, root = '', replicates = 1, aug=False):
        super(DAVIS2016, self).__init__(train=train, is_cropped = is_cropped, root = root, replicates = replicates, aug=aug)
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
    def __init__(self, train=False, is_cropped = False, root = '', replicates = 1, aug=False):
        self.is_cropped = is_cropped
        self.replicates = replicates
        self.aug = aug

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
        index_list.remove(index)

        while flag:
            if len(index_list) == 0:
                index = random.choice(list(range(self.size)))
                seq = self.seq_id_list[index]
                index_list = [i for i, x in enumerate(self.seq_id_list) if x == seq]
                index_list.remove(index)

            gt_template_o = np.expand_dims(np.array(Image.open(self.gt_list[index])), axis=2)
            if len(np.unique(gt_template_o)) == 1:
                index = random.choice(index_list)
                index_list.remove(index)
                continue

            search_center = random.choice(index_list)
            i_index = index_list.index(search_center)
            index_list.remove(search_center)
            candidates = index_list[i_index-2:i_index] + index_list[i_index+1:i_index+3]

            while len(candidates) > 0:
                search_index = random.choice(candidates)

                gt_search_o = np.expand_dims(np.array(Image.open(self.gt_list[search_index])), axis=2)
                if len(np.unique(gt_search_o)) == 1:
                    candidates.remove(search_index)
                    continue

                labels_template = np.unique(gt_template_o).tolist()
                labels_search = np.unique(gt_search_o).tolist()
                labels = list(set(labels_template).intersection(set(labels_search)))
                if 0 in labels:
                    labels.remove(0)
                while len(labels) > 0:
                    idx = random.choice(labels)
                    gt_template = gt_template_o.copy()
                    gt_search = gt_search_o.copy()
                    gt_template[gt_template!=idx] = 0
                    gt_search[gt_search!=idx] = 0
                    gt_template[gt_template==idx] = 1
                    gt_search[gt_search==idx] = 1
                    bb_template = cv2.boundingRect(gt_template.squeeze())
                    bb_search = cv2.boundingRect(gt_search.squeeze())
                    if bb_search[2] < 30 or bb_search[3] < 30 or bb_template[2] < 30 or bb_template[3] < 30:
                        labels.remove(idx)
                        continue
                    img_template  = cv2.imread(self.image_list[index])
                    img_search = cv2.imread(self.image_list[search_index])
                    flag = False
                    break
                candidates.remove(search_index)


        if self.aug:
            img, mask, target, box, gt = aug_pair(img_template, img_search, gt_template,
                    gt_search, no_crop=True)
            #img, target, gt = aug_mask(img_template, img_search, gt_template, gt_search, mask)

        img = img.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        target = target.transpose(2, 0, 1)
        box = box.transpose(2, 0, 1)
        gt = gt.transpose(2, 0, 1)

        img = torch.from_numpy(img.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))
        target = torch.from_numpy(target.astype(np.float32))
        box = torch.from_numpy(box.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))


        return img, mask, target, box, gt

    
    def __len__(self):
        return self.size * self.replicates

class YTB_VOS(data.Dataset):
    def __init__(self, train=False, is_cropped = False, root = '', replicates = 1, aug=False):
        self.is_cropped = is_cropped
        self.replicates = replicates
        self.aug = aug

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
        flag = True 

        index = index % self.size
        seq = self.seq_id_list[index]
        index_list = [i for i, x in enumerate(self.seq_id_list) if x == seq]
        index_list.remove(index)

        while flag:
            if len(index_list) == 0:
                index = random.choice(list(range(self.size)))
                seq = self.seq_id_list[index]
                index_list = [i for i, x in enumerate(self.seq_id_list) if x == seq]
                index_list.remove(index)

            gt_template_o = np.expand_dims(np.array(Image.open(self.gt_list[index])), axis=2)
            if len(np.unique(gt_template_o)) == 1:
                index = random.choice(index_list)
                index_list.remove(index)
                continue

            search_center = random.choice(index_list)
            i_index = index_list.index(search_center)
            index_list.remove(search_center)
            candidates = index_list[i_index-2:i_index] + index_list[i_index+1:i_index+3]

            while len(candidates) > 0:
                search_index = random.choice(candidates)

                gt_search_o = np.expand_dims(np.array(Image.open(self.gt_list[search_index])), axis=2)
                if len(np.unique(gt_search_o)) == 1:
                    candidates.remove(search_index)
                    continue

                labels_template = np.unique(gt_template_o).tolist()
                labels_search = np.unique(gt_search_o).tolist()
                labels = list(set(labels_template).intersection(set(labels_search)))
                if 0 in labels:
                    labels.remove(0)
                while len(labels) > 0:
                    idx = random.choice(labels)
                    gt_template = gt_template_o.copy()
                    gt_search = gt_search_o.copy()
                    gt_template[gt_template!=idx] = 0
                    gt_search[gt_search!=idx] = 0
                    gt_template[gt_template==idx] = 1
                    gt_search[gt_search==idx] = 1
                    bb_template = cv2.boundingRect(gt_template.squeeze())
                    bb_search = cv2.boundingRect(gt_search.squeeze())
                    if bb_search[2] < 30 or bb_search[3] < 30 or bb_template[2] < 30 or bb_template[3] < 30:
                        labels.remove(idx)
                        continue
                    img_template  = cv2.imread(self.image_list[index])
                    img_search = cv2.imread(self.image_list[search_index])
                    flag = False
                    break
                candidates.remove(search_index)


        if self.aug:
            img, mask, target, box, gt = aug_pair(img_template, img_search, gt_template,
                    gt_search, no_crop=True)
            #img, target, gt = aug_mask(img_template, img_search, gt_template, gt_search, mask)

        img = img.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        target = target.transpose(2, 0, 1)
        box = box.transpose(2, 0, 1)
        gt = gt.transpose(2, 0, 1)

        img = torch.from_numpy(img.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))
        target = torch.from_numpy(target.astype(np.float32))
        box = torch.from_numpy(box.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))


        return img, mask, target, box, gt

    def __len__(self):
        return self.size * self.replicates

class GyGO(data.Dataset):
    def __init__(self, train=False, root = '', replicates = 1, aug=False):
        self.replicates = replicates
        self.aug = aug

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

        img_template = cv2.imread(self.image_list[index])
        img_search = cv2.imread(self.image_list[search_index])

        gt_template = np.expand_dims(np.array(Image.open(self.gt_list[index]).convert('L')), axis=2)
        gt_search= np.expand_dims(np.array(Image.open(self.gt_list[search_index]).convert('L')), axis=2)
        gt_template[gt_template<128] = 0
        gt_template[gt_template!=0] = 1
        gt_search[gt_search<128] = 0
        gt_search[gt_search!=0] = 1
        
        if self.aug:
            img, mask, target, box, gt = aug_pair(img_template, img_search, gt_template,
                    gt_search, no_crop=True)
            #img, target, gt = aug_mask(img_template, img_search, gt_template, gt_search, mask)
            
        # hwc

        img = img.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        target = target.transpose(2, 0, 1)
        box = box.transpose(2, 0, 1)
        gt = gt.transpose(2, 0, 1)

        img = torch.from_numpy(img.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))
        target = torch.from_numpy(target.astype(np.float32))
        box = torch.from_numpy(box.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        return img, mask, target, box, gt

    def __len__(self):
        return self.size * self.replicates


class ECSSD_dreaming(data.Dataset):
    def __init__(self, is_cropped = False, root = '', replicates = 1, aug=False):
        self.is_cropped = is_cropped
        self.replicates = replicates
        self.aug = aug

        root = join(root, 'dreaming')

        self.image_list = []
        self.gt_list = []
        self.mask_list = []
        seqs = sorted(os.listdir(root))
        for seq in seqs:
            files = [i for i in os.listdir(join(root,seq)) if 'jpg' in i]
            imgs = []
            gts = []
            masks = []
            for f in files:
                img = join(root, seq, f)
                gt = join(root, seq, f[:-4]+'.png')
                imgs.append(img)
                gts.append(gt)
                masks.append(join(root, seq, f[:-4]+'bb.png'))
            self.image_list += [imgs]
            self.gt_list += [gts]
            self.mask_list += [masks]

        self.size = len(self.image_list)


    def __getitem__(self, index):

        index = index % self.size
        img_template = cv2.imread(self.image_list[index][0])
        img_search = cv2.imread(self.image_list[index][1])
        

        gt_template = np.expand_dims(np.array(Image.open(self.gt_list[index][0])), axis=2)
        gt_search = np.expand_dims(np.array(Image.open(self.gt_list[index][1])), axis=2)
        mask = np.expand_dims(np.array(Image.open(self.mask_list[index][0])), axis=2)

        gt_template[gt_template==255] = 1
        gt_search[gt_search==255] = 1
        mask[mask==255] = 1

        if len(np.unique(mask)) != 2:
            mask = gt_search.copy()


        if self.aug:
            img, mask, target, box, gt = aug_pair(img_template, img_search, gt_template,
                    gt_search, no_crop=True)
            #img, mask, target, box, gt = aug_mask_nodeform(img_template, img_search, gt_template, gt_search, mask)
            
        img = img.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        target = target.transpose(2, 0, 1)
        box = box.transpose(2, 0, 1)
        gt = gt.transpose(2, 0, 1)

        img = torch.from_numpy(img.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))
        target = torch.from_numpy(target.astype(np.float32))
        box = torch.from_numpy(box.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        return img, mask, target, box, gt

    def __len__(self):
        return self.size * self.replicates

class MSRA10K_dreaming(data.Dataset):
    def __init__(self, is_cropped = False, root = '', replicates = 1, aug=False):
        self.is_cropped = is_cropped
        self.replicates = replicates
        self.aug = aug

        root = join(root, 'dreaming')

        self.image_list = []
        self.gt_list = []
        self.mask_list = []
        seqs = sorted(os.listdir(root))
        for seq in seqs:
            files = [i for i in os.listdir(join(root,seq)) if 'jpg' in i]
            imgs = []
            gts = []
            masks = []
            for f in files:
                img = join(root, seq, f)
                gt = join(root, seq, f[:-4]+'.png')
                imgs.append(img)
                gts.append(gt)
                masks.append(join(root, seq, f[:-4]+'bb.png'))
            self.image_list += [imgs]
            self.gt_list += [gts]
            self.mask_list += [masks]
        

        self.size = len(self.image_list)
       

    def __getitem__(self, index):

        index = index % self.size
        img_template = cv2.imread(self.image_list[index][0])
        img_search = cv2.imread(self.image_list[index][1])
        

        gt_template = np.expand_dims(np.array(Image.open(self.gt_list[index][0])), axis=2)
        gt_search = np.expand_dims(np.array(Image.open(self.gt_list[index][1])), axis=2)
        mask = np.expand_dims(np.array(Image.open(self.mask_list[index][0])), axis=2)

        gt_template[gt_template==255] = 1
        gt_search[gt_search==255] = 1
        mask[mask==255] = 1

        if len(np.unique(mask)) != 2:
            mask = gt_search.copy()


        if self.aug:
            img, mask, target, box, gt = aug_pair(img_template, img_search, gt_template,
                    gt_search, no_crop=True)
            #img, mask, target, box, gt = aug_mask_nodeform(img_template, img_search, gt_template, gt_search, mask)

        img = img.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        target = target.transpose(2, 0, 1)
        box = box.transpose(2, 0, 1)
        gt = gt.transpose(2, 0, 1)

        img = torch.from_numpy(img.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))
        target = torch.from_numpy(target.astype(np.float32))
        box = torch.from_numpy(box.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        return img, mask, target, box, gt
        
    def __len__(self):
        return self.size * self.replicates

