import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

sam_checkpoint = "/Users/usmankhan/Downloads/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "mps"
#device = "cpu"

print('Initializing model.')
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)


image = cv2.imread('/Users/usmankhan/Downloads/train_v1/a_0196.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print('Setting Image')
predictor.set_image(image)

fig = plt.figure(figsize=(8,6))
plt.imshow(image)
plt.axis('on')

coords = []

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print (f'x = {ix}, y = {iy}')

    global coords
    coords.append((ix, iy))

    if len(coords) == 2:
        fig.canvas.mpl_disconnect(cid)

    return coords

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

print(coords)
flat_coords = np.array([coord for sublist in coords for coord in sublist])
print(flat_coords)

masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=flat_coords[None, :],
    multimask_output=False,
)

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks[0], plt.gca())
show_box(flat_coords, plt.gca())
plt.axis('off')
plt.show()

q = input('Do you want to save the mask (Y/N)?')
if q.lower() == 'y':
    # Save mask -- masks[0]
    print('Saving mask.')
