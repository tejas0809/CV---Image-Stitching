'''
  File name: corner_detector.py
  Author:
  Date created:
'''

'''
  File clarification:
    Detects corner features in an image. You can probably find free “harris” corner detector on-line,
    and you are allowed to use them.
    - Input img: H × W matrix representing the gray scale input image.
    - Output cimg: H × W matrix representing the corner metric matrix.
'''
import cv2
import numpy as np


def corner_detector(img):
  # Your Code Here
  dest = cv2.cornerHarris(img, 2, 3, 0.04)
  # print(dest)
  return dest

if __name__ == '__main__':
  img = cv2.imread("hill1_cyl.png")
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  cimg = corner_detector(gray)
  img[cimg > 0.01 * cimg.max()] = [0,0,255]
  cv2.imwrite('hill1_cyl_corner.jpg',img)