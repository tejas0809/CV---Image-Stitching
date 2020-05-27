'''
  File name: anms.py
  Author:
  Date created:
'''

'''
  File clarification:
    Implement Adaptive Non-Maximal Suppression. The goal is to create an uniformly distributed
    points given the number of feature desired:
    - Input cimg: H × W matrix representing the corner metric matrix.
    - Input max_pts: the number of corners desired.
    - Outpuy x: N × 1 vector representing the column coordinates of corners.
    - Output y: N × 1 vector representing the row coordinates of corners.
    - Output rmax: suppression radius used to get max pts corners.
'''
from corner_detector import corner_detector
import numpy as np
import cv2
def anms(cimg, max_pts):
  # Your Code Here
  # cimg[cimg < np.median(cimg)+cimg.std()/3] = 0
  cimg[cimg<0.01*np.max(cimg)]=0
  copy = np.copy(cimg)

  nonzero = len(np.nonzero(copy)[0])

  finalmat = np.zeros((nonzero,3))
  x = np.nonzero(copy)[0]
  y = np.nonzero(copy)[1]
  k = 0
  dummy_ans=np.zeros(cimg.shape)
  for i,j in zip(x,y):
    # print(k)
    currentpoint = np.array([i,j])
    otherpts = np.argwhere((copy > copy[i,j]) * (copy < (1.2 * copy[i,j])))
    if (otherpts.shape[0] == 0):
      finalmat[k,:] = [i,j,np.inf]
    else:
      dist = (otherpts - currentpoint)**2
      dist = np.sum(dist, axis = 1)
      dist = np.sqrt(dist)
      rmin = np.amin(dist)
      finalmat[k,:] = [i,j,rmin]
    k = k + 1
  allmat = finalmat[np.argsort(-finalmat[:,2])]
  X = allmat[:max_pts,0]
  Y = allmat[:max_pts,1]
  rmax = np.amin(allmat[:max_pts,2])

  for x1,y1 in zip(X,Y):
      dummy_ans[int(x1),int(y1)]=cimg[int(x1),int(y1)]

  return Y.astype(int), X.astype(int), rmax

