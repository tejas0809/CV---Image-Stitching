from anms import anms
from corner_detector import corner_detector
from feat_desc import feat_desc
from feat_match import feat_match
from ransac_est_homography import ransac_est_homography
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches
from mymosaic import mymosaic
from utils import plot_image_matching
from test import cylindricalWarp
from utils import removeBlackSides


img = cv2.imread('collegeGreen1.jpg')
img1_cyl, trackimg1 = cylindricalWarp(img)
img1_cyl,trackimg1=removeBlackSides(img1_cyl,trackimg1)
img1_cyl = img1_cyl.astype(np.uint8)
gray= cv2.cvtColor(img1_cyl.astype(np.uint8),cv2.COLOR_BGR2GRAY)


img2 = cv2.imread('collegeGreen2.jpg')
img2_cyl,trackimg2 = cylindricalWarp(img2)
img2_cyl,trackimg2=removeBlackSides(img2_cyl,trackimg2)
img2_cyl = img2_cyl.astype(np.uint8)
gray2= cv2.cvtColor(img2_cyl.astype(np.uint8),cv2.COLOR_BGR2GRAY)


cimg=corner_detector(gray)
cimg[trackimg1 == 0] = 0
# img1_cyl[cimg > 0.01 * cimg.max()] = [0,0,255]
# cv2.imwrite('collegeGreen1_cyl_corner.jpg',img1_cyl)
x1,y1,r=anms(cimg,1000)
print('x1.shape',x1.shape)
desc=feat_desc(gray,x1,y1)

cimg2=corner_detector(gray2)
cimg2[trackimg2 == 0] = 0
# img2_cyl[cimg2 > 0.01 * cimg2.max()] = [0,0,255]
# cv2.imwrite('collegeGreen2_cyl_corner.jpg',img2_cyl)
x2,y2,r2=anms(cimg2,1000)
print('x2.shape',x2.shape)
desc2=feat_desc(gray2,x2,y2)

good=feat_match(desc,desc2)
# print(good.shape)

npgood=np.array(good)
npgood[npgood==-1]=0
print('nonzero',np.count_nonzero(npgood==0))
goodmask=npgood.astype(bool)
npgood=npgood[goodmask]

x1=x1[goodmask]
y1=y1[goodmask]

X1=np.copy(x1)
Y1=np.copy(y1)

X2=np.zeros((x1.shape))
Y2=np.zeros((y1.shape))
X2=x2[npgood]
Y2=y2[npgood]

H12,inlier_ind=ransac_est_homography(X1,Y1,X2,Y2,0.7)
print(H12)

plot_image_matching(X1,Y1,X2,Y2,img1_cyl,img2_cyl)
####################################################################################
img = cv2.imread('collegeGreen3.jpg')
img3_cyl,trackimg3 = cylindricalWarp(img)
img3_cyl,trackimg3=removeBlackSides(img3_cyl,trackimg3)
img3_cyl = img3_cyl.astype(np.uint8)
gray= cv2.cvtColor(img3_cyl.astype(np.uint8),cv2.COLOR_BGR2GRAY)


img2 = cv2.imread('collegeGreen2.jpg')
img2_cyl,trackimg2 = cylindricalWarp(img2)
img2_cyl,trackimg2=removeBlackSides(img2_cyl,trackimg2)
img2_cyl = img2_cyl.astype(np.uint8)
gray2= cv2.cvtColor(img2_cyl.astype(np.uint8),cv2.COLOR_BGR2GRAY)


cimg=corner_detector(gray)
cimg[trackimg3==0]=0
x1,y1,r=anms(cimg,1000)
print('x1.shape',x1.shape)
desc=feat_desc(gray,x1,y1)

cimg2=corner_detector(gray2)
cimg2[trackimg2==0]=0
x2,y2,r2=anms(cimg2,1000)
print('x2.shape',x2.shape)
desc2=feat_desc(gray2,x2,y2)

good=feat_match(desc,desc2)
print(good.shape)

npgood=np.array(good)
npgood[npgood==-1]=0
goodmask=npgood.astype(bool)
npgood=npgood[goodmask]

x1=x1[goodmask]
y1=y1[goodmask]

X1=np.copy(x1)
Y1=np.copy(y1)

X2=np.zeros((x1.shape))
Y2=np.zeros((y1.shape))
X2=x2[npgood]
Y2=y2[npgood]

H32,inlier_ind=ransac_est_homography(X1,Y1,X2,Y2,0.7)
print(H32)

plot_image_matching(X1,Y1,X2,Y2,img3_cyl,img2_cyl)
################################################################################
# img1 = cv2.imread('hill1.jpg')
# img2 = cv2.imread('hill2.jpg')
# img3 = cv2.imread('hill3.jpg')
# img1 = cv2.imread('.png')
# img2 = cv2.imread('hill2_cyl.png')
# img3 = cv2.imread('hill3_cyl.png')

finalOP=mymosaic(img1_cyl,img2_cyl,img3_cyl,H12,H32)
cv2.imwrite("collegeGreen_affine_final_uncarved.jpg",finalOP)

# if __name__ == '__main__':
#     img1 = cv2.imread('hill1_cyl.png')
#     img2 = cv2.imread('hill2_cyl.png')
#     img3 = cv2.imread('hill3_cyl.png')
