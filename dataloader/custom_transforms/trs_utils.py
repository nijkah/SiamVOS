import torch
import random
import numpy as np
from torchvision import transforms
from tools.utils import scale_box

from PIL import Image, ImageOps, ImageFilter
import imgaug as ia
from imgaug import augmenters as iaa

class ImgAugBatch(object):
    def __init__(self, cfg=None, aug=True, deform=True):
        self.cfg = cfg
        self.aug = aug 
        self.deform = deform 
    
    def __call__(self, img_ref, gt_ref, img, gt):
        img_ref, gt_ref = np.asarray(img_ref), np.asarray(gt_ref)
        img, gt = np.asarray(img), np.asarray(gt)
                
        if self.aug is not None:
            img_ref, gt_ref = self.resize_crop(img_ref, gt_ref)
            img, gt = self.resize_crop(img, gt)
        
        if self.deform is not None:
            img_p = self.get_previous(img)
            p_mask = self.get_previous(gt)
            box_ref = self.get_box(gt_ref)
        
        img = np.array(img).astype(float)
        gt = np.array(gt)[..., np.newaxis].astype(float)
        p_mask = p_mask[..., np.newaxis]

        img /= 255.
        img_p /= 255.
        img_ref /= 255.
        
        return img, p_mask, img_p, img_ref, box_ref, gt
    
    def resize_crop(self, img, gt):
        h, w = img.shape[:2]
        height = self.cfg.RESIZE_DIM if h <= w else 'keep-aspect-ratio'
        width = self.cfg.RESIZE_DIM if w < h else 'keep-aspect-ratio'

        aug = iaa.Sequential([
                iaa.Resize({'height':height, 'width':width}),
                iaa.CropToFixedSize(self.cfg.INPUT_DIM, self.cfg.INPUT_DIM),
                iaa.Fliplr(0.5),
                iaa.GaussianBlur(sigma=(0, 3.0))])
        
        tform_det = aug.to_deterministic()
        gt_map = ia.SegmentationMapOnImage(gt, shape=gt.shape, nb_classes=2)
        cropped_img = tform_det.augment_image(img).astype(float)
        cropped_gt = tform_det.augment_segmentation_maps([gt_map])[0].get_arr_int().astype(float)

        return cropped_img, cropped_gt
    
    def get_previous(self, x):

        tr = iaa.Sequential([
                iaa.Affine(
                    scale={"x": (0.98, 1.02), "y": (0.98, 1.02)},
                    translate_px={"x": (-20, 20), "y": (-20, 20)}, # translate by -20 to +20 percent (per axis)
                    rotate=(-5, 5), # rotate by -45 to +45 degrees
                    shear=(-8, 8), # shear by -16 to +16 degrees
                    order=0, # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    mode='edge' # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )])

        if len(x.shape) == 2 or x.shape[-1] == 1:
            box_p = random.uniform(0, 1)
            if box_p < 0.3:
                box = self.get_box(np.array(x))
                return box.squeeze().astype(float)
            else:
                gt_map = ia.SegmentationMapOnImage(x.astype('uint8'), shape=x.shape, nb_classes=2)
                cropped_gt = tr.augment_segmentation_maps([gt_map])[0].get_arr_int().astype(float)
                return cropped_gt.squeeze().astype(float)
        else:
            x_previous = tr.augment_image(x)

        return np.array(x_previous).astype(float)
    
    def get_box(self, gt):
        bb = scale_box(gt)
        box = np.zeros([*gt.shape[:2], 1]).astype(float)
        if bb[0] == 0 and bb[1] == 0 and bb[2] == 0 and bb[3] == 0:
            return box
        box[bb[1]:bb[1]+bb[3]+1, bb[0]:bb[0]+bb[2]+1, 0] = 1

        return box


class ProcessBatch(object):
    def __init__(self, cfg=None, aug=True, deform=True):
        self.cfg = cfg
        self.aug = aug 
        self.deform = deform 
    
    def __call__(self, img_ref, gt_ref, img, gt):
                
        if self.aug is not None:
            img, gt = self.augment_img(img, gt)
            img_ref, gt_ref = self.augment_img(img_ref, gt_ref)
        
        img_ref, gt_ref = np.array(img_ref).astype(float), np.array(gt_ref).astype(float)

        
        if self.deform is not None:
            img_p = self.get_previous(img)
            p_mask = self.get_previous(gt)
            box_ref = self.get_box(gt_ref)
        
        img = np.array(img).astype(float)
        gt = np.array(gt)[..., np.newaxis].astype(float)
        p_mask = p_mask[..., np.newaxis]

        img /= 255.
        img_p /= 255.
        img_ref /= 255.
        
        return img, p_mask, img_p, img_ref, box_ref, gt


    def augment_img(self, img, gt):
        sample={'image':img, 'label':gt}
        composed_transforms = transforms.Compose([
            RandomHorizontalFlip(),
            RandomScaleCrop(base_size=self.cfg['base_size'], crop_size=self.cfg['crop_size']),
            RandomGaussianBlur()])
        
        transformed_sample = composed_transforms(sample)

        return transformed_sample['image'], transformed_sample['label']

    def get_previous(self, x):
        tr = transforms.RandomAffine(
                    degrees=(-5, 5),
                    translate=(0.05, .05),
                    scale=(0.98, 1.02),
                    shear=(-10, 10, -10, 10))
        x_previous = tr(x)
        if x.mode == "L":
            box_p = random.uniform(0, 1)
            if box_p < 0.3:
                box = self.get_box(np.array(x_previous))
                return box.squeeze()

        return np.array(x_previous).astype(float)
    
    def get_box(self, gt):
        bb = scale_box(gt)
        box = np.zeros([*gt.shape[:2], 1]).astype(float)
        box[bb[1]:bb[1]+bb[3]+1, bb[0]:bb[0]+bb[2]+1, 0] = 1

        return box

            


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}
