'''
  File name: ransac_est_homography.py
  Author:
  Date created:
'''

'''
  File clarification:
    Use a robust method (RANSAC) to compute a homography. Use 4-point RANSAC as
    described in class to compute a robust homography estimate:
    - Input x1, y1, x2, y2: N Ã— 1 vectors representing the correspondences feature coordinates in the first and second image.
                            It means the point (x1_i , y1_i) in the first image are matched to (x2_i , y2_i) in the second image.
    - Input thresh: the threshold on distance used to determine if transformed points agree.
    - Outpuy H: 3 Ã— 3 matrix representing the homograph matrix computed in final step of RANSAC.
    - Output inlier_ind: N Ã— 1 vector representing if the correspondence is inlier or not. 1 means inlier, 0 means outlier.
'''
from random import seed
from random import sample
from random import randint
from est_homography import est_homography

import numpy as np

def ransac_est_homography(x1, y1, x2, y2, thresh):
  # Your Code Here
  nRANSAC=2500 #increased
  N=x1.shape[0]
  print("N",N)

  ind=np.stack((x1,y1),-1)
  ind1=ind.reshape(-1,2)
  one=np.ones((ind1.shape[0],1))
  xy1=np.concatenate((ind1,one),axis=1)
  xy1=xy1.reshape((xy1.shape[0],xy1.shape[1],1))

  indi=np.stack((x2,y2),-1)
  indi1=indi.reshape(-1,2)
  onei=np.ones((indi1.shape[0],1))
  xy2=np.concatenate((indi1,onei),axis=1)
  xy2=xy2.reshape((xy2.shape[0],xy2.shape[1],1))
  r_optim = np.inf
  max_cnt=0
  optim_indices=np.zeros((3))
  while nRANSAC:
      # nRANSAC=nRANSAC-1
      rand_indices=np.array(sample(range(0, N), 3))


      H=est_homography(x1[rand_indices],y1[rand_indices],x2[rand_indices],y2[rand_indices])
      if nRANSAC == 2500:
          print(x1[rand_indices])
          print(y1[rand_indices])
          print(x2[rand_indices])
          print(y2[rand_indices])
          print('Hi')
          print(H)
      H=H/H[2,2]
      H_mat = np.repeat(H[ np.newaxis, :, :], x1.shape[0], axis=0)
      ans=H_mat@xy1
      ans=ans/ans[:,2,None]
      r=np.sqrt((ans[:,0,0]-xy2[:,0,0])**2+(ans[:,1,0]-xy2[:,1,0])**2)
      # print(r.shape)


      inlier_ind=np.array([r<thresh],dtype=np.int)
      # print(inlier_ind.shape)
      cur_cnt=np.sum(inlier_ind)
      inlier_bool = inlier_ind[0].astype(bool)
      r_inlier = r[inlier_bool]
      sum_r_inlier = np.sum(r_inlier)
      if(cur_cnt>max_cnt):
          r_optim = sum_r_inlier
          max_cnt=cur_cnt
          optim_indices=np.copy(rand_indices)
      elif (cur_cnt == max_cnt):
          # print('True')
          if sum_r_inlier < r_optim:
              # print("2nd true")
              r_optim = sum_r_inlier
              max_cnt = cur_cnt
              optim_indices = np.copy(rand_indices)
      nRANSAC = nRANSAC - 1



  final_H=est_homography(x1[optim_indices],y1[optim_indices],x2[optim_indices],y2[optim_indices])
  final_H=final_H/final_H[2,2]
  final_H_mat=np.repeat(final_H[np.newaxis, :, :], x1.shape[0], axis=0)
  final_ans=final_H_mat@xy1
  final_ans=final_ans/final_ans[:,2,None]

  final_r=np.sqrt((final_ans[:,0,0]-xy2[:,0,0])**2+(final_ans[:,1,0]-xy2[:,1,0])**2)
  final_inlier_ind=np.array([final_r<thresh],dtype=np.int)
  print(np.sum(final_inlier_ind))
  return final_H, final_inlier_ind[0]

# if __name__ == '__main__':
#     img1 =

