import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio
import cv2

# Set User parameters
DAVIS_PATH = "/media/datasets/DAVIS/"
RESULT_PATH = '../scripts/Result_triplet_GC_re/'
#RESULT_PATH = '../scripts/results/MO_siamvos/'
#SAVE_PATH = "../data/save/siamvos_color_2"
SAVE_PATH = "../data/save/siamvos_color_3"
palette = Image.open(DAVIS_PATH+ 'Annotations/480p/bear/00000.png').getpalette()

# Show results
def overlay_seq(result_path, seq_name=None, data_path=DAVIS_PATH, SAVE_PATH=SAVE_PATH):
    if os.path.isdir(os.path.join(SAVE_PATH, seq_name)):
        print('exist')
    else:
        os.makedirs(os.path.join(SAVE_PATH, seq_name))
    transparency = 0.3
    test_frames = sorted(os.listdir(os.path.join(data_path, 'JPEGImages', '480p', seq_name)))
    plt.ion()

    masks = sorted(os.listdir(os.path.join(result_path)))
    gts = sorted(os.listdir(os.path.join(DAVIS_PATH, 'Annotations', '480p', seq_name)))

    for i, img_p in enumerate(test_frames):
        frame_num = img_p.split('.')[0]
        img = np.array(Image.open(os.path.join(data_path, 'JPEGImages', '480p', seq_name, img_p)))

        mask = Image.open(os.path.join(result_path, masks[i])).convert('RGB')
        mask = np.array(mask)

        if np.isin([0, 255], np.unique(mask)).all():
            mask[mask==255] = 100

        im_over = cv2.addWeighted(img, .9, mask, 1.5, 0)

        imageio.imwrite(os.path.join(SAVE_PATH,seq_name,frame_num+'.jpg'), im_over.astype('uint8'))
        
def overlay_comparison(result_path, seq_name=None, data_path=DAVIS_PATH):
    if os.path.isdir(os.path.join(SAVE_PATH, seq_name)):
        print('exist')
    else:
        os.makedirs(os.path.join(SAVE_PATH, seq_name))
    overlay_color = [0, 255, 0]
    gt_color = [255, 0, 30]
    transparency = 0.6
    test_frames = sorted(os.listdir(os.path.join(data_path, 'JPEGImages', '480p', seq_name)))
    plt.ion()

    masks = sorted(os.listdir(os.path.join(result_path)))
    gts = sorted(os.listdir(os.path.join(DAVIS_PATH, 'Annotations', '480p', seq_name)))

    for i, img_p in enumerate(test_frames):
        if i == 0:
            continue
        frame_num = img_p.split('.')[0]
        img = np.array(Image.open(os.path.join(data_path, 'JPEGImages', '480p', seq_name, img_p)))
        mask = np.array(Image.open(os.path.join(result_path, masks[i-1])))
        gt = np.array(Image.open(os.path.join(DAVIS_PATH, 'Annotations/480p', seq_name, gts[i])))

        mask = mask/np.max(mask)
        gt = gt/np.max(gt)
        im_over = np.ndarray(img.shape)
        im_over[:, :, 0] = (1 - gt) * img[:, :, 0] + gt * (gt_color[0]*transparency + (1-transparency)*img[:, :, 0])
        im_over[:, :, 1] = (1 - mask) * img[:, :, 1] + mask* (overlay_color[1]*transparency+ (1-transparency)*img[:, :, 1])
        im_over[:, :, 2] = (1 - gt) * img[:, :, 2] + gt * (gt_color[2]*transparency + (1-transparency)*img[:, :, 2])

        imageio.imwrite(os.path.join(SAVE_PATH,seq_name,frame_num+'.png'), im_over.astype('uint8'))

if __name__ == "__main__":
    #seqs = os.listdir(os.path.join(segtrack_path, 'JPEGImages/480p'))
    seqs = sorted(os.listdir(RESULT_PATH))
    for seq in seqs:
        result_path = os.path.join(RESULT_PATH,seq)
        print(seq)
        overlay_seq(result_path, seq, DAVIS_PATH, SAVE_PATH=SAVE_PATH)
