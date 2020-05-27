'''
  File name: feat_match.py
  Author:
  Date created:
'''

'''
  File clarification:
    Matching feature descriptors between two images. You can use k-d tree to find the k nearest neighbour.
    Remember to filter the correspondences using the ratio of the best and second-best match SSD. You can set the threshold to 0.6.
    - Input descs1: 64 × N1 matrix representing the corner descriptors of first image.
    - Input descs2: 64 × N2 matrix representing the corner descriptors of second image.
    - Outpuy match: N1 × 1 vector where match i points to the index of the descriptor in descs2 that matches with the
                    feature i in descriptor descs1. If no match is found, you should put match i = −1.
'''
import cv2
import numpy as np
from annoy import AnnoyIndex

def feat_match(descs1, descs2):
  # Your Code Here
  n1, points1 = descs1.shape[:2]
  n2, points2 = descs2.shape[:2]
  t = AnnoyIndex(n1, metric ="euclidean")

  for i in range(points2):
    t.add_item(i,descs2[:,i])
  t.build(50)

  matches = np.zeros((points1), dtype = int)

  for i in range(points1):
    p_index, dist = t.get_nns_by_vector(descs1[:,i], 2, include_distances=True)
    if dist[0]/dist[1] < 0.6:
      matches[i] =  p_index[0]
    else:
      matches[i] = -1


  return matches

