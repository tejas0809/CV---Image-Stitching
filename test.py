import cv2
import numpy as np
from utils import interp3


def cylindricalWarp(img):
    h_, w_ = img.shape[:2]
    f=(h_+w_)/3
    K = np.array([[f, 0, w_ / 2], [0, f, h_ / 2], [0, 0, 1]])
    print(h_,w_)
    # pixel coordinates
    y_i, x_i = np.indices((h_, w_))
    X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h_ * w_, 3)
    # print(X)
    Kinv = np.linalg.inv(K)
    # print(Kinv)
    X = Kinv.dot(X.T).T  # normalized coords
    #print(X)
    #cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(w_ * h_, 3)
    B = K.dot(A.T).T  # project back to image-pixels plane
    # print(B)
    # back from homog coords
    B = B[:, :-1] / B[:, [-1]]
    Bcopy = np.copy(B)
    B[np.logical_or(np.logical_or((B[:, 0] < 0), (B[:, 0] >= w_)),np.logical_or((B[:, 1] < 0), (B[:, 1] >= h_)))] = 1
    B = B.reshape(h_, w_, -1)
    img[1,1] = [0,0,0]
    out_image = np.zeros((h_,w_,3))
    out_image[:,:,0] = interp3(img[:,:,0],B[:,:,0],B[:,:,1])
    out_image[:, :, 1] = interp3(img[:, :, 1], B[:, :, 0], B[:, :, 1])
    out_image[:, :, 2] = interp3(img[:, :, 2], B[:, :, 0], B[:, :, 1])

    trackB = np.ones(Bcopy.shape[0])
    trackB[(np.logical_and(Bcopy[:,0] >=-50,Bcopy[:,0] <= 10)) | (np.logical_and(Bcopy[:,0] <= w_ + 50, Bcopy[:,0]>=w_ - 10))|(np.logical_and(Bcopy[:,1] >=-50, Bcopy[:,1] <= 10)) | (np.logical_and(Bcopy[:,1] <= h_ + 50 , Bcopy[:,1]>=h_ - 10))] = 0
    # print('trackBshape',trackB.shape)
    trackBnew = trackB.reshape(h_,w_)
    # print(trackBnew.shape)
    # print(np.count_nonzero(trackBnew==0))
    # np.save('trackBnew_hill1',trackBnew)


    return out_image, trackBnew


if __name__ == '__main__':
    img = cv2.imread("input1.png")
    print(img)
    h, w = img.shape[:2]
    img_cyl,_ = get_cyl(img)
    gray = cv2.cvtColor(img_cyl.astype(np.uint8),cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('gray.jpg',gray)
    cv2.imwrite("input1_cyl.jpg", img_cyl)
