import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
import cv2

def get_box(mask, scale=1.0):
    bb = scale_box(mask, scale)
    box = np.zeros([*mask.shape[:2], 1]).astype(float)
    if bb[0] == 0 and bb[1] == 0 and bb[2] == 0 and bb[3] == 0:
        return box
    box[bb[1]:bb[1]+bb[3]+1,bb[0]:bb[0]+bb[2]+1] = 1

    return box


def scale_box(mask, scale=1.0):
    mask = mask.squeeze()
    all_ys, all_xs = np.where(mask >= 0.5)
    if all_ys.size == 0 or all_xs.size == 0:
        #ymin, ymax = 0, mask.shape[0]
        #xmin, xmax = 0, mask.shape[1]
        return 0, 0, 0, 0
    else:
        ymin, ymax = np.min(all_ys), np.max(all_ys)
        xmin, xmax = np.min(all_xs), np.max(all_xs)
    
    h = ymax - ymin + 1
    w = xmax - xmin + 1

    return xmin, ymin, w, h

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def overlay(img, mask, color=[255, 0, 0], transparency=0.6, gray=True):
    im_over = np.zeros(img.shape)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        for c in range(3):
            im_over[:, :, c] = (1 - mask) * img + mask * (color[c]*transparency + (1-transparency)*img)
    else:
        for c in range(3):
            im_over[:, :, c] = (1 - mask) * img[:, :, c] + mask * (color[c]*transparency + (1-transparency)*img[:, :, c])

    im_over[im_over>255] = 255
    im_over[im_over<0] =0 

    return im_over


def vis(img, mask, gt, out, analysis=True):
    plt.ion()
    im = img.data.cpu().numpy().transpose(1, 2, 0)
    im *= 255
    fg = mask.data.cpu().numpy().transpose(1, 2, 0)
    fg[fg>0] = 1
    fg[fg!=1] = 0
    im = overlay(im, fg.squeeze())

    out = out.data.cpu().numpy()
    out = np.argmax(out, 0)
    
    shape = out.shape

    plt.subplot(2, 3, 1).set_title('Search region')
    plt.imshow(im.astype('uint8'))

    gt_pred = np.zeros([shape[0], shape[1], 3])
    gt_pred[:,:,0] = gt.squeeze() * 255
    gt_pred[:,:,2] = out * 255
    #plt.subplot(2, 3, 6).set_title('GT-pred')
    plt.subplot(2, 3, 6)
    plt.text(3, 3, 'GT and Pred', bbox={'facecolor':'white', 'pad':1})
    precision, recall = calculate_precision(out, gt.squeeze()), calculate_recall(out, gt.squeeze())
    plt.xlabel('prec:'+str(precision)+' reca:'+str(recall))
    plt.imshow(gt_pred.astype('uint8'))

    plt.subplot(2, 3, 2).set_title('Ground Truth')
    plt.imshow(gt.squeeze())

    plt.subplot(2, 3, 4).set_title('Prediction')
    plt.imshow(out)

    mask_pred = np.zeros([shape[0], shape[1], 3])
    fg = cv2.resize(fg, (shape[0], shape[1]), cv2.INTER_NEAREST)
    mask_pred[:,:,0] = fg * 255
    mask_pred[:,:,2] = out * 255
    plt.subplot(2, 3, 5).set_title('Mask-pred')
    precision, recall = calculate_precision(out, fg), calculate_recall(out, fg)
    plt.xlabel('prec:'+str(precision)+' reca:'+str(recall))
    plt.imshow(mask_pred.astype('uint8'))


    plt.show()
    plt.pause(0.05)
    plt.clf()

def vis_2(img, mask,target, box,gt, out, iter=0, analysis=True, vis=True):
    plt.ion()
    im = img.data.cpu().numpy().transpose(1, 2, 0)
    im *= 255
    fg = mask.data.cpu().numpy().transpose(1, 2, 0)
    fg[fg>0] = 1
    fg[fg!=1] = 0
    im = overlay(im, fg.squeeze())

    target_im= target.data.cpu().numpy().transpose(1, 2, 0)
    target_im *= 255
    fg2 = box.data.cpu().numpy().transpose(1, 2, 0)
    fg2[fg2>0] = 1
    fg2[fg2!=1] = 0
    target_im = overlay(target_im, fg2.squeeze(), gray=False)

    out = out.data.cpu().numpy()
    out = np.argmax(out, 0)
    
    shape = out.shape

    plt.subplot(2, 3, 1).set_title('Template')
    plt.imshow(target_im.astype('uint8'))

    plt.subplot(2, 3, 2).set_title('Search region')
    plt.imshow(im.astype('uint8'))

    gt_pred = np.zeros([shape[0], shape[1], 3])
    gt= gt.data.cpu().numpy().transpose(1, 2, 0)
    gt= cv2.resize(gt, (shape[0], shape[1]), cv2.INTER_NEAREST)
    gt_pred[:,:,0] = gt.squeeze() * 255
    gt_pred[:,:,2] = out * 255
    #plt.subplot(2, 3, 6).set_title('GT-pred')
    plt.subplot(2, 3, 6)
    plt.text(3, 3, 'GT and Pred', bbox={'facecolor':'white', 'pad':1})
    precision, recall = calculate_precision(out, gt.squeeze()), calculate_recall(out, gt.squeeze())
    plt.xlabel('prec:'+str(precision)+' reca:'+str(recall))
    plt.imshow(gt_pred.astype('uint8'))

    plt.subplot(2, 3, 4).set_title('Ground Truth')
    plt.imshow(gt.squeeze())

    plt.subplot(2, 3, 5).set_title('Prediction')
    plt.imshow(out)

    mask_pred = np.zeros([shape[0], shape[1], 3])
    fg = cv2.resize(fg, (shape[0],  shape[1]), cv2.INTER_NEAREST)
    mask_pred[:,:,0] = fg * 255
    mask_pred[:,:,2] = out * 255
    plt.subplot(2, 3, 3).set_title('Mask-pred')
    precision, recall = calculate_precision(out, fg), calculate_recall(out, fg)
    plt.xlabel('prec:'+str(precision)+' reca:'+str(recall))
    plt.imshow(mask_pred.astype('uint8'))

    if not os.path.exists('debugging'):
        os.makedirs('debugging')
    plt.savefig(os.path.join('debugging', 'debug_'+str(iter%1000)+'.png'))

    if vis:
        plt.show()
        plt.pause(0.05)
        plt.clf()


def compute_direct_coordinate(bb, context_factor=3):
    context = (bb[2]+bb[3])//context_factor

    direct_x = bb[0]-context
    direct_y = bb[1]-context
    direct_w = max(bb[2]+2*context, 1)
    direct_h = max(bb[3]+2*context, 1)

    return direct_x, direct_y, direct_w, direct_h

def compute_padding(direct_coordinate, wh):
    m_w, m_h = wh

    direct_x, direct_y, direct_w, direct_h = direct_coordinate
    padded_left = abs(direct_x) if direct_x < 0 else 0
    padded_up = abs(direct_y) if direct_y < 0 else 0
    padded_right = direct_x+direct_w - m_w + 1 if direct_x+direct_w > m_w  else 0
    padded_bottom = direct_y+direct_h - m_h + 1 if direct_y+direct_h > m_h else 0

    pads = ( (padded_up, padded_bottom), (padded_left, padded_right) )

    return pads

def crop_and_padding(img, mask, wh, context_on=True, context_factor=3):
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, 2)
    if len(img.shape) == 2:
        img = np.expand_dims(img, 2)
    c_c = img.shape[2]
    w, h = wh
    m_h, m_w, _ = mask.shape

    if c_c != 1:
        im_rgb_mean = [np.mean(img[:,:,c]) for c in range(c_c)]

    
    if context_on:
        bb = cv2.boundingRect(mask)
        direct_coordinate = compute_direct_coordinate(bb, context_factor)
        pads = compute_padding(direct_coordinate, (m_w, m_h))

        dbb = [0, 0, 0, 0]
        dbb[0] = direct_coordinate[0]+pads[1][0]
        dbb[1] = direct_coordinate[1]+pads[0][0]
        dbb[2] = direct_coordinate[2]
        dbb[3] = direct_coordinate[3]

    else:
        direct_coordinate = cv2.boundingRect(mask)
        pads = compute_padding(direct_coordinate, (m_w, m_h))
        print(pads)
        dbb = [0, 0, 0, 0]
        dbb[0] = direct_coordinate[0]+pads[1][0]
        dbb[1] = direct_coordinate[1]+pads[0][0]
        dbb[2] = direct_coordinate[2]
        dbb[3] = direct_coordinate[3]
        
    if c_c != 1:
        img = np.stack([np.pad(img[:, :, c], pads, 'constant', constant_values=im_rgb_mean[c]) for c in range(c_c)], axis=2)
    else:
        img = np.pad(img.squeeze(), pads, 'constant', constant_values=0)[:, :, np.newaxis]

    cropped_img = img[dbb[1]:dbb[1]+dbb[3], dbb[0]:dbb[0]+dbb[2], :]
    
    direct_x, direct_y, direct_w, direct_h = direct_coordinate
    fit_to_w = direct_w > direct_h 
    cc_w = w if fit_to_w else int(float(direct_w)/direct_h *h)
    cc_h = int(float(direct_h )/direct_w*w) if fit_to_w else h
    
    pads = ( ((h-cc_h+1)//2, (h-cc_h)//2), ((w-cc_w+1)//2, (w-cc_w)//2))

    if c_c != 1:
        cropped_img = cv2.resize(cropped_img, (cc_w, cc_h))
        padded_img = np.stack([np.pad(cropped_img[:,:,c], pads, 'constant', constant_values=im_rgb_mean[c]) for c in range(c_c)], axis=2 )
    else:
        cropped_img = cv2.resize(cropped_img, (cc_w, cc_h), cv2.INTER_NEAREST)
        padded_img = np.pad(cropped_img, pads, 'constant', constant_values=0)

    return padded_img

def restore_mask(mask, bb, shape, context_factor=3):
    o_h, o_w = shape[0], shape[1]
    m_h, m_w = mask.shape[0], mask.shape[1]
    
    direct_coordinate = compute_direct_coordinate(bb, context_factor)
    direct_x, direct_y, direct_w, direct_h = direct_coordinate

    cropped_left =  0 if direct_x < 0 else direct_x
    cropped_up = 0 if direct_y < 0 else direct_y
    cropped_right = o_w if direct_x+direct_w > o_w  else direct_x+direct_w
    cropped_bottom = o_h if direct_y+direct_h > o_h else direct_y+direct_h

    ###
    fit_to_w = direct_w > direct_h 
    cc_w = m_w if fit_to_w else int(float(direct_w)/direct_h *m_h)
    cc_h = int(float(direct_h )/direct_w*m_w) if fit_to_w else m_h
    
    pads = ( ((m_h-cc_h+1)//2, (m_h-cc_h)//2), ((m_w-cc_w+1)//2, (m_w-cc_w)//2))

    unpadded_mask = mask[pads[0][0]:pads[0][0]+cc_h, pads[1][0]:pads[1][0]+cc_w]
    #reresized_mask = cv2.resize(unpadded_mask.astype('uint8'), (direct_w, direct_h), cv2.INTER_NEAREST)
    reresized_mask = cv2.resize(unpadded_mask, (direct_w, direct_h))

    context = (bb[2]+bb[3])//context_factor
    restored_mask = np.zeros([shape[0], shape[1], 2])
    restored_mask[:, :, 0] = 1
    pads = compute_padding(direct_coordinate,(o_w,o_h))
    restored_mask[cropped_up:cropped_bottom, cropped_left:cropped_right] = \
        reresized_mask[pads[0][0]:direct_h-pads[0][1]+1,pads[1][0]:direct_w-pads[1][1]+1]
    ####
    """
    fit_to_w = w > h
    resized_w = m_w if fit_to_w else int(float(direct_w)/direct_h*m_h)
    resized_h = int(float(direct_h)/direct_w*m_w) if fit_to_w else m_h
    
    pads = ( (m_h-resized_h+1)//2, (m_h-resized_h)//2, (m_w-resized_w+1)/2, (m_w-resized_w)/2)

    unpadded_mask = mask[pads[0]:pads[0]+resized_h, pads[2]:pads[2]+resized_w]
    reresized_mask = cv2.resize(unpadded_mask.astype('uint8'), (direct_w, direct_h), cv2.INTER_NEAREST)

    restored_mask[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]] = reresized_mask[context:context+h, context:context+w]
    """

    return restored_mask

def get_iou(pred, gt, ignore_cls=None):
    if pred.shape != gt.shape:
        print('pred shape', pred.shape, 'gt shape', gt.shape)
    assert (pred.shape == gt.shape)
    #gt = gt.astype(np.float32)
    #pred = pred.astype(np.float32)

    labels = np.unique(gt).tolist()
    if isinstance(ignore_cls, int):
        labels = [label for label in labels if label != ignore_cls]
    if isinstance(ignore_cls, list):
        labels = [label for label in labels if label not in ignore_cls]

    if len(labels) == 0:
        
        if (len(np.unique(pred)) == 1 and np.unique(pred)[0] == ignore_cls):
            return 1
        else:
            return 0

    count = dict()

    for j in labels:
        x = np.where(pred == j)
        p_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        x = np.where(gt == j)
        GT_idx_j = set(zip(x[0].tolist(), x[1].tolist()))

        n_jj = set.intersection(p_idx_j, GT_idx_j)
        u_jj = set.union(p_idx_j, GT_idx_j)

        if len(GT_idx_j) != 0:
            count[j] = float(len(n_jj)) / len(u_jj)

    result_class = list(count.values())

    Aiou = np.sum(result_class[:]) / len(labels)

    return Aiou


def get_general_iou(pred, gt):
    if pred.shape != gt.shape:
        print('pred shape', pred.shape, 'gt shape', gt.shape)
    assert (pred.shape == gt.shape)
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    labels = np.unique(gt).tolist()

    count = dict()

    for j in labels:
        x = np.where(pred == j)
        p_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        x = np.where(gt == j)
        GT_idx_j = set(zip(x[0].tolist(), x[1].tolist()))

        n_jj = set.intersection(p_idx_j, GT_idx_j)
        u_jj = set.union(p_idx_j, GT_idx_j)

        if len(GT_idx_j) != 0:
            count[j] = float(len(n_jj)) / float(len(u_jj))

    result_class = count.values()
    Aiou = np.sum(result_class[:]) / float(len(np.unique(gt)))

    return Aiou

def calculate_precision(pred, gt, save_imgs=0):
    x = np.where(pred == 1)
    pred_idx = set(zip(x[0].tolist(), x[1].tolist()))
    x = np.where(gt == 1)
    gt_idx = set(zip(x[0].tolist(), x[1].tolist()))
    true_positives = set.intersection(pred_idx, gt_idx)

    if len(pred_idx) == 0:
        return 0

    return round(float(len(true_positives)) / len(pred_idx), 3)

def calculate_recall(pred, gt, save_imgs=0):
    x = np.where(pred == 1)
    pred_idx = set(zip(x[0].tolist(), x[1].tolist()))
    x = np.where(gt == 1)
    gt_idx = set(zip(x[0].tolist(), x[1].tolist()))
    true_positives = set.intersection(pred_idx, gt_idx)

    x = np.where(pred == 0)
    pred_idx_neg = set(zip(x[0].tolist(), x[1].tolist()))
    false_negatives = set.intersection(pred_idx_neg, gt_idx)
    if(len(true_positives)+len(false_negatives)) == 0:
        return 0

    return round(float(len(true_positives)) / (len(true_positives)+len(false_negatives)), 3)
