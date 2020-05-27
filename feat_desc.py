'''
  File name: feat_desc.py
  Author:
  Date created:
'''

'''
  File clarification:
    Extracting Feature Descriptor for each feature point. You should use the subsampled image around each point feature,
    just extract axis-aligned 8x8 patches. Note that it’s extremely important to sample these patches from the larger 40x40
    window to have a nice big blurred descriptor.
    - Input img: H × W matrix representing the gray scale input image.
    - Input x: N × 1 vector representing the column coordinates of corners.
    - Input y: N × 1 vector representing the row coordinates of corners.
    - Outpuy descs: 64 × N matrix, with column i being the 64 dimensional descriptor (8 × 8 grid linearized) computed at location (xi , yi) in img.
'''
import cv2
import numpy as np
from scipy import signal
from utils import GaussianPDF_2D,patch_mat
from corner_detector import corner_detector
from anms import anms

def feat_desc(img,x,y):
    G = GaussianPDF_2D(0,1,5,5)
    dx, dy = np.gradient(G, axis=(1, 0))
    Ix = signal.convolve2d(img,dx,'same')
    Iy = signal.convolve2d(img,dy,'same')
    Im = np.sqrt(Ix*Ix + Iy*Iy)
    descs = np.zeros((64, x.shape[0]))
    Imag = np.pad(Im, ((20,20),(20,20)),'constant', constant_values=0)
    # print(Imag.shape)
    count = 0
    for i,j in zip(x,y):
        i = int(i + 20)
        j = int(j + 20)
        # print(count, i,j)
        patch40 = Imag[j - 20: j + 20,i - 20:i + 20]
        # //change kiya i aur j to access i,j in image
        # print('pthc40',patch40.shape)
        sub_patches = patch_mat(patch40,5,5)

        flattened = sub_patches.reshape(64, 25)
        max_vals = np.amax(flattened, axis = 1)
        # print('shape',max_vals)
        mean = np.mean(max_vals[max_vals != 0])
        std = np.std(max_vals[max_vals != 0])
        final = (max_vals - mean)/std
        # print('final',np.mean(final), np.std(final))
        # print(final)
        # print(descs[:,count].shape)
        descs[:,count] = final
        count += 1
        # print('count',count)

    return descs

# img = cv2.imread('berkeley1.jpg')
# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cimg=corner_detector(gray)
# x,y,r = anms(cimg,200)
# # print(x.shape)
# descs = feat_desc(gray,x,y)
# print(descs[:,199])

