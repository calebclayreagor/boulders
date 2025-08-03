#%%
import os, glob
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator

# set paths
pth = os.path.join('..', 'data', 'img', '*', '*.jpg')
msk_pth = os.path.join('..', 'data', 'mask')
plt_pth = os.path.join('..', 'data', 'plot')

# load ViT-H SAM model
sam_pth = os.path.join('..', 'sam', 'sam_vit_h_4b8939.pth')
sam = sam_model_registry['vit_h'](sam_pth).to('cuda')
msk_gen = SamAutomaticMaskGenerator(sam,
            stability_score_thresh = .9,
            points_per_dimension = 64,
            points_per_batch = 16)

# visualize masks
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key = (lambda x: x['area']), reverse = True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((*sorted_anns[0]['segmentation'].shape, 4))
    img[..., 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# SAM inference
for fn in sorted(glob.glob(pth)):
    img = cv2.imread(fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    msk_list = msk_gen.generate(img)

    # join masks
    msk_join = np.zeros_like(msk_list[0]['segmentation'], dtype = np.int64)
    for i, msk in enumerate(msk_list):
        msk_join[msk['segmentation']] = i + 1
    resize_shape = (img.shape[1], img.shape[0])
    msk_join = cv2.resize(msk_join, resize_shape, interpolation = cv2.INTER_NEAREST)
    
    # save
    fn_split = os.path.split(fn)
    fn_split_2 = os.path.split(fn_split[0])
    name_out = fn_split[1].replace('.jpg', '.npy')
    fn_out = os.path.join(msk_pth, fn_split_2[1], name_out)
    os.makedirs(os.path.dirname(fn_out), exist_ok = True)
    np.save(fn_out, msk_join)

    # plot
    fn_out = os.path.join(plt_pth, fn_split_2[1], fn_split[1])
    os.makedirs(os.path.dirname(fn_out), exist_ok = True)
    plt.figure(figsize = (20, 20))
    plt.imshow(img_gray, cmap = 'gray')
    show_anns(msk_list)
    plt.axis('off')
    plt.savefig(fn_out)
    plt.close()

    # clear memory
    del msk_list, msk_join
    torch.cuda.empty_cache()

#%%
