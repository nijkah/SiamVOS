import torch
import numpy as np
import torch.nn.functional as F

def is_box(mask):
    np_mask = mask.data.cpu().numpy().astype('uint8')
    if len(np_mask.shape) == 4:
        np_mask = np.squeeze(np_mask, 1)
    is_box_list = []

    for b in range(np_mask.shape[0]):
        all_ys, all_xs  = np.where(np_mask[b] >= 0.49)

        if all_ys.size == 0 or all_xs.size == 0:
            # if no pixel, return whole
            ymin, ymax = 0, np_mask.shape[1]
            xmin, xmax = 0, np_mask.shape[2]
        else:
            ymin, ymax = np.min(all_ys), np.max(all_ys)
            xmin, xmax = np.min(all_xs), np.max(all_xs)
        h = ymax - ymin + 1
        w = xmax - xmin + 1

        area = h*w
        val_area = np_mask[b].sum()
        if area == val_area:
            is_box_list.append(True)
        else:
            is_box_list.append(False)

    return is_box_list


def get_ROI_grid(roi, src_size, dst_size, scale=1.):
    # scale height and width
    ry, rx, rh, rw = roi[:,0], roi[:,1], scale * roi[:,2], scale * roi[:,3]
    
    # convert ti minmax  
    ymin = ry - rh/2.
    ymax = ry + rh/2.
    xmin = rx - rw/2.
    xmax = rx + rw/2.
    
    h, w = src_size[0], src_size[1] 
    # theta
    theta = torch.zeros(roi.size()[0],2,3)
    theta[:,0,0] = (xmax - xmin) / (w - 1)
    theta[:,0,2] = (xmin + xmax - (w - 1)) / (w - 1)
    theta[:,1,1] = (ymax - ymin) / (h - 1)
    theta[:,1,2] = (ymin + ymax - (h - 1)) / (h - 1)

    #inverse of theta
    inv_theta = torch.zeros(roi.size()[0],2,3)
    det = theta[:,0,0]*theta[:,1,1]
    adj_x = -theta[:,0,2]*theta[:,1,1]
    adj_y = -theta[:,0,0]*theta[:,1,2]
    inv_theta[:,0,0] = w / (xmax - xmin) 
    inv_theta[:,1,1] = h / (ymax - ymin) 
    inv_theta[:,0,2] = adj_x / det
    inv_theta[:,1,2] = adj_y / det
    # make affine grid
    fw_grid = F.affine_grid(theta, torch.Size((roi.size()[0], 1, dst_size[0], dst_size[1]))).cuda()
    bw_grid = F.affine_grid(inv_theta, torch.Size((roi.size()[0], 1, src_size[0], src_size[1]))).cuda()
    return fw_grid, bw_grid, theta


def mask2yxhw(mask, scale=1.0):
    np_mask = mask.data.cpu().numpy()
    if len(np_mask.shape) == 4:
        np_mask = np.squeeze(np_mask, 1)

    np_yxhw = np.zeros((np_mask.shape[0], 4), dtype=np.float32)
    for b in range(np_mask.shape[0]):
        all_ys, all_xs  = np.where(np_mask[b] >= 0.49)

        if all_ys.size == 0 or all_xs.size == 0:
            # if no pixel, return whole
            ymin, ymax = 0, np_mask.shape[1]
            xmin, xmax = 0, np_mask.shape[2]
        else:
            ymin, ymax = np.min(all_ys), np.max(all_ys)
            xmin, xmax = np.min(all_xs), np.max(all_xs)

        # make sure minimum 128 original size
        if (ymax-ymin) < 128:
            res = 128. - (ymax-ymin)
            ymin -= int(res/2)
            ymax += int(res/2)

        if (xmax-xmin) < 128:
            res = 128. - (xmax-xmin)
            xmin -= int(res/2)
            xmax += int(res/2)

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

        yxhw = np.array([y,x,h,w], dtype=np.float32)
        
        np_yxhw[b] = yxhw
        
    return torch.from_numpy(np_yxhw.copy()).float()
