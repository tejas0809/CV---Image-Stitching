import cv2
import numpy as np


def removeBlackSides(img1_cyl):
    mid=int(img1_cyl.shape[0]/2)
    print(mid)
    slice=img1_cyl[mid,:]
    j=0
    start=0
    end=0
    for i in range(img1_cyl.shape[1]):
        if(slice[i,0]!=0 or slice[i,1]!=0 or slice[i,2]!=0):
            start=i
            break

    for i in range(img1_cyl.shape[1]-1,start,-1):
        if(slice[i,0]!=0 or slice[i,1]!=0 or slice[i,2]!=0):
            end=i
            break
    # print(start)
    # print(end)
    ans=img1_cyl[:,start:end]
    # cv2.imwrite("TEST.jpg",ans)
    return ans

if __name__=="__main__":
    img = cv2.imread('collegeGreen1.jpg')
    img1_cyl, trackimg1 = cylindricalWarp(img)
    img1_cyl = img1_cyl.astype(np.uint8)
    img1cropped=removeBlackSides(img1_cyl)
    cv2.imwrite("TEST.jpg",img1cropped)
