'''
  File name: mymosaic.py
  Author:
  Date created:
'''

'''
  File clarification:
    Produce a mosaic by overlaying the pairwise aligned images to create the final mosaic image. If you want to implement
    imwarp (or similar function) by yourself, you should apply bilinear interpolation when you copy pixel values.
    As a bonus, you can implement smooth image blending of the final mosaic.
    - Input img_left:H×W×3 matrix representing left image
            img_middle:HxW×3 matrix representing middle image
            img_right:H×W×3 matrix representing right image
            H12:3×3 homography matrix representing warping from frame of left image to frame of middle image
            H32:3×3 homography matrix representing warping from frame or right image to frame of middle image
    - Output img_mosaic: H’×W’×3 representing the stitched image mosaic of three images
'''
import numpy as np
from utils import stitch
import cv2
def mymosaic(img_left, img_middle, img_right, H12, H32):
  # Your Code Here
  img2_new=np.copy(img_middle)
  img2_pad=np.pad(img2_new,((img_left.shape[0],img_left.shape[0]),(img_left.shape[1],img_left.shape[1]),(0,0)),"constant",constant_values=0)
  #padded middle images


  left_on_middle=stitch(img_left,img2_pad,H12,img_middle)
  # cv2.imwrite('left_on_middle_input_cG_affine.jpg',left_on_middle)
  all_three=stitch(img_right,left_on_middle,H32,img_middle)
  img_mosaic=all_three

  return img_mosaic
