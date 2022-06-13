import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import cv2

flickerDir = '../data/flickrlogos32/images/*'
openlogo = '../data/openlogo/images/*'
gvision = '../data/gvision/images/*'
val = '../data/validation/images/*'

flickr_img_size = []
openlogo_img_size = []
gvision_img_size = []
val_img_size = []
all_imgs = []

# Flickr32
all_files = glob(flickerDir)
for f in all_files:
    im = cv2.imread(f)
    im_h, im_w, channels = im.shape
    flickr_img_size.append(im_h*im_w)

# OpenLogo
all_files = glob(openlogo)
for f in all_files:
    im = cv2.imread(f)
    im_h, im_w, channels = im.shape
    openlogo_img_size.append(im_h*im_w)

# Gvision
all_files = glob(gvision)
for f in all_files:
    im = cv2.imread(f)
    im_h, im_w, channels = im.shape
    gvision_img_size.append(im_h*im_w)

# Validation
all_files = glob(val)
for f in all_files:
    im = cv2.imread(f)
    im_h, im_w, channels = im.shape
    val_img_size.append(im_h*im_w)

all_imgs.append(flickr_img_size)
all_imgs.append(openlogo_img_size)
all_imgs.append(gvision_img_size)
all_imgs.append(val_img_size)

labels = ['Flickr32', 'OpenLogo', 'Google Vision', 'Validation']

font = {'family' : 'normal',
        'size'   : 14,
        'weight' : 'bold'}

plt.rc('font', **font)

# rectangular box plot
bp = plt.boxplot(all_imgs,
            vert=True,          # vertical box alignment
            patch_artist=True,  # fill with color
            labels=labels,     # will be used to label x-ticks
            showfliers=False)

plt.title('Total image size (w x h) in pixels per Dataset')
plt.xlabel('Datasets')
plt.ylabel('Total image size (w x h) in pixels')
plt.show()