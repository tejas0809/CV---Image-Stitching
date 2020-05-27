import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import patches
import time
import cv2
def GaussianPDF_1D(mu, sigma, length):
  # create an array
  half_len = length / 2

  if np.remainder(length, 2) == 0:
    ax = np.arange(-half_len, half_len, 1)
  else:
    ax = np.arange(-half_len, half_len + 1, 1)

  ax = ax.reshape([-1, ax.size])
  denominator = sigma * np.sqrt(2 * np.pi)
  nominator = np.exp( -np.square(ax - mu) / (2 * sigma * sigma) )

  return nominator / denominator

def GaussianPDF_2D(mu, sigma, row, col):
  # create row vector as 1D Gaussian pdf
  g_row = GaussianPDF_1D(mu, sigma, row)
  # create column vector as 1D Gaussian pdf
  g_col = GaussianPDF_1D(mu, sigma, col).transpose()

  return signal.convolve2d(g_row, g_col, 'full')

def patch_mat(arr, nrows, ncols):
  """
  Return an array of shape (n, nrows, ncols) where
  n * nrows * ncols = arr.size

  If arr is a 2D array, the returned array should look like n subblocks with
  each subblock preserving the "physical" layout of arr.
  """
  h, w = arr.shape
  assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
  assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
  return (arr.reshape(h // nrows, nrows, -1, ncols)
          .swapaxes(1, 2)
          .reshape(-1, nrows, ncols))



def interp2(v, xq, yq):

   if len(xq.shape) == 2 or len(yq.shape) == 2:
      dim_input = 2
      q_h = xq.shape[0]
      q_w = xq.shape[1]
      xq = xq.flatten()
      yq = yq.flatten()

   h = v.shape[0]
   w = v.shape[1]
   if xq.shape != yq.shape:
      raise Exception('query coordinates Xq Yq should have same shape')

   x_floor = np.floor(xq).astype(np.int32)
   y_floor = np.floor(yq).astype(np.int32)
   x_ceil = np.ceil(xq).astype(np.int32)
   y_ceil = np.ceil(yq).astype(np.int32)

   x_floor[x_floor < 0] = 0
   y_floor[y_floor < 0] = 0
   x_ceil[x_ceil < 0] = 0
   y_ceil[y_ceil < 0] = 0

   x_floor[x_floor >= w-1] = w-1
   y_floor[y_floor >= h-1] = h-1
   x_ceil[x_ceil >= w-1] = w-1
   y_ceil[y_ceil >= h-1] = h-1

   v1 = v[y_floor, x_floor]
   v2 = v[y_floor, x_ceil]
   v3 = v[y_ceil, x_floor]
   v4 = v[y_ceil, x_ceil]

   lh = yq - y_floor
   lw = xq - x_floor
   hh = 1 - lh
   hw = 1 - lw

   w1 = hh * hw
   w2 = hh * lw
   w3 = lh * hw
   w4 = lh * lw

   interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4
   dim_input = 0
   if dim_input == 2:
      return interp_val.reshape(q_h, q_w)
   return interp_val


def plot_image_matching(x1, y1, x2, y2, img1, img2):
   f = plt.figure(figsize=(30, 30))
   ax1 = f.add_subplot(121)
   ax2 = f.add_subplot(122)

   ax1.imshow(img1)
   ax1.axis('off')
   ax2.imshow(img2)
   ax2.axis('off')

   ax1.plot(x1, y1, 'bo', markersize=3)
   ax2.plot(x2, y2, 'bo', markersize=3)
   # print("length after match", len(x1))
   for i in range(len(x1)):
      con = patches.ConnectionPatch(xyA=(x2[i], y2[i]), xyB=(x1[i], y1[i]), coordsA="data", coordsB="data",
                       axesA=ax2, axesB=ax1, color="red")
      ax2.add_artist(con)

   plt.show()

# convert rgb to grayscale

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def genEngMap(I):
  dim = I.ndim
  if dim == 3:
    Ig = rgb2gray(I)
  else:
    Ig = I

  Ig = Ig.astype(np.float64())

  [gradx, grady] = np.gradient(Ig);
  e = np.abs(gradx) + np.abs(grady)
  return e

def cumMinEngHor(e):
  # print('start hor')
  start_time = time.time()
  # Your Code Here
  height = e.shape[0]
  width = e.shape[1]
  k = int(1)
  value_matrix = np.zeros((height + 2, width + 2))  # +1 because for 0th row to be filled in value_matrix it will need to have +0
  value_matrix[0,:] = float('inf')
  value_matrix[-1,:] = float('inf')
  path_matrix = np.zeros((height, width))
  i = np.arange(1, height + 1)
  e1 = np.pad(e, ((1, 1), (1, 1)), 'constant', constant_values=(1))
  c, d = np.meshgrid(np.arange(0, width), np.arange(0, height))
  for j in range(1, width + 1):

    add_term = np.amin(np.array([value_matrix[i - 1, j - 1], value_matrix[i, j - 1], value_matrix[i + 1, j - 1]]).T, axis=1)
    value_matrix[i, j] = e1[i, j] + add_term
    lol = np.vstack((value_matrix[i - 1, j - 1], value_matrix[i, j - 1], value_matrix[i + 1, j - 1]))
    path_matrix[:,j-1] = np.argmin(lol,axis = 0) - 1
  path_matrix[:,0] = 0
  value_matrix = value_matrix[1:height + 1, 1:width + 1]
  My = value_matrix
  Tby = path_matrix
  # print(value_matrix)
  # print('...')
  # print(path_matrix)
  # print('time taken hor: ', time.time() - start_time)
  # return value_matrix, path_matrix
  return My, Tby

def cumMinEngVer(e):
  # print('start ver')
  # start_time = time.time()
  # Your Code Here
  height = e.shape[0]
  width = e.shape[1]
  k = int(1)
  value_matrix = np.zeros((height + 2,width + 2))
  value_matrix[:,0] = float('inf')
  value_matrix[:,-1] = float('inf')
  path_matrix = np.zeros((height,width))
  j = np.arange(1, width + 1)
  e1 = np.pad(e, ((1,1),(1,1)), 'constant', constant_values=(1))
  c,d = np.meshgrid(np.arange(0, width), np.arange(0, height))
  for i in range(1, height + 1):
    add_term = np.amin(np.array([value_matrix[i - 1, j - 1], value_matrix[i-1, j], value_matrix[i-1, j+1]]).T, axis = 1)
    value_matrix[i,j] = e1[i,j] + add_term
    lol = np.vstack((value_matrix[i - 1, j - 1], value_matrix[i - 1, j], value_matrix[i - 1, j + 1]))
    path_matrix[i-1,:] = np.argmin(lol, axis=0) - 1
  path_matrix[0,:] = 0
  value_matrix = value_matrix[1:height + 1,1:width + 1]
  # print('time taken ver: ', time.time() - start_time)
  return value_matrix, path_matrix

def rmHorSeam(I, My, Tby):

  Iy, E,arr = rmVerSeam(np.transpose(I,(1,0,2)), My.T, Tby.T)
  arr=np.flip(arr,1)
  return np.transpose(Iy,(1,0,2)), E,arr



def rmVerSeam(I, Mx, Tbx):
  # Your Code Here

  height, width, _ = I.shape
  del_arr = np.ones((height, width), dtype=np.bool)
  del_indices = []
  x_index = [Mx.shape[0] - 1]
  min_index = np.argmin(Mx[-1])
  del_indices.append(min_index)
  E = Mx[x_index[-1],del_indices[-1]]
  for i_n in range(Mx.shape[0] - 1):
    i = Mx.shape[0] - 2 - i_n
    min_index = min_index + int(Tbx[x_index[-1], del_indices[-1]])
    del_indices.append(min_index)
    x_index.append(i)
  del_indices = np.array(del_indices)
  x_index = np.array(x_index)
  array=np.zeros((Mx.shape[0],2))
  array[:,0]=x_index
  array[:,1]=del_indices
  # print(array)
  array=np.flip(array,0)
  np.save("outfile.npy",array)
  del_arr[x_index, del_indices] = False
  del_arr = np.stack([del_arr] * 3, axis=2)
  Ix = I[del_arr].reshape((height, width - 1,3))
  return Ix, E, array


def removeBlackSides(img1_cyl, trackimg):
    mid=int(img1_cyl.shape[0]/2)
    # print(mid)
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
    ans_track=trackimg[:,start:end]
    # cv2.imwrite("TEST.jpg",ans)
    return ans,ans_track



def stitch(img1,img2,H,img2_orig):
    img2_new=np.copy(img2)
    img2_copy=np.copy(img2)
    left_start=0
    rigt_start=0
    for i in range(img2_orig.shape[0]):
        if(img2_orig[i,0,0]!=0 or img2_orig[i,0,1]!=0 or img2_orig[i,0,2]!=0):
            left_start=i
            break
    for i in range(img2_orig.shape[0]):
        if(img2_orig[i,img2_orig.shape[1]-1,0]!=0 or img2_orig[i,img2_orig.shape[1]-1,1]!=0 or img2_orig[i,img2_orig.shape[1]-1,2]!=0):
            rigt_start=i
            break

    for i in range(img2_orig.shape[0]-1,0,-1):
        if(img2_orig[i,img2_orig.shape[1]-1,0]!=0 or img2_orig[i,img2_orig.shape[1]-1,1]!=0 or img2_orig[i,img2_orig.shape[1]-1,2]!=0):
            rigt_end=i
            break

    for i in range(img2_orig.shape[0]-1,0,-1):
        if(img2_orig[i,0,0]!=0 or img2_orig[i,0,1]!=0 or img2_orig[i,0,2]!=0):
            left_end=i
            break

    # img2_new=np.pad(img2_new1,((img1.shape[0],img1.shape[0]),(img1.shape[1],img1.shape[1]),(0,0)),"constant",constant_values=0)
    xpts=[0,img1.shape[1]-1,img1.shape[1]-1,0]
    ypts=[0,0,img1.shape[0]-1,img1.shape[0]-1]
    xypts=np.vstack((xpts,ypts)).T
    onepts=np.ones((xypts.shape[0],1))
    xy_pts=np.concatenate((xypts,onepts),axis=1)
    xy_pts=xy_pts.reshape((xy_pts.shape[0],xy_pts.shape[1],1))
    xy_pts=xy_pts.astype(int)

    T=np.array([[1,0,img1.shape[1]],[0,1,img1.shape[0]],[0,0,1]])

    TRANS_mat=T@H
    xy_newpts=H@xy_pts
    xy_newpts=xy_newpts/xy_newpts[:,2,None]
    xy_2pts=T@xy_newpts

    xy_2pts=xy_2pts/xy_2pts[:,2,None]
    print('xypts',xypts)
    print('xy_2',xy_2pts)



    xrec0=int(min(xy_2pts[0,0,0],xy_2pts[2,0,0]))
    yrec0=int(min(xy_2pts[0,1,0],xy_2pts[1,1,0]))
    xrec3=xrec0
    yrec1=yrec0
    xrec1=int(max(xy_2pts[1,0,0],xy_2pts[2,0,0]))
    xrec2=xrec1
    yrec2=int(max(xy_2pts[2,1,0],xy_2pts[3,1,0]))
    yrec3=yrec2
    xxrec,yyrec=np.meshgrid(np.arange(xrec0,xrec1+1),np.arange(yrec0,yrec3+1))

    xrec=xxrec.flatten()
    yrec=yyrec.flatten()

    xy1rec = np.vstack((xrec,yrec)).T

    onerec=np.ones((xy1rec.shape[0],1))
    xy_1rec=np.concatenate((xy1rec,onerec),axis=1)
    xy_1rec=xy_1rec.reshape((xy_1rec.shape[0],xy_1rec.shape[1],1))
    xy_1rec=xy_1rec.astype(int)

    TRANS_mat_inv=np.linalg.inv(TRANS_mat)

    H_final=TRANS_mat_inv@T

    points_in_one=TRANS_mat_inv@xy_1rec

    points_in_one=points_in_one/points_in_one[:,2,None]
    points_in_one[points_in_one[:,0,0]<0]=0
    points_in_one[points_in_one[:,1,0]<0]=0
    points_in_one[points_in_one[:,0,0]>img1.shape[1]]=0
    points_in_one[points_in_one[:,1,0]>img1.shape[0]]=0
    points_in_one_mask=points_in_one.astype(bool)

    boolmask=points_in_one_mask[:,0,0]

    points_in_one=points_in_one[boolmask]
    xy_1rec=xy_1rec[boolmask]


    points_in_onex=points_in_one[:,0,0]

    points_in_oney=points_in_one[:,1,0]

    pixel_values_blue=interp2(img1[:,:,0],points_in_onex,points_in_oney)
    pixel_values_green = interp2(img1[:,:,1],points_in_onex,points_in_oney)
    pixel_values_red = interp2(img1[:,:,2],points_in_onex,points_in_oney)
    replace_image = np.dstack((pixel_values_blue,pixel_values_green,pixel_values_red))


    output_image = np.copy(img2_new)
    output_image[xy_1rec[:,1,0],xy_1rec[:,0,0]] = replace_image
    channel1o = output_image[:, :, 0]
    channel2o = output_image[:, :, 1]
    channel3o = output_image[:, :, 2]
    channel1i  = img2[:,:,0]
    channel2i = img2[:,:,1]
    channel3i = img2[:,:,2]
    channel1i[~np.logical_and(np.logical_and(channel1o == 0,channel2o == 0),channel3o == 0)] = channel1o[~np.logical_and(np.logical_and(channel1o == 0,channel2o == 0),channel3o == 0)]
    channel2i[~np.logical_and(np.logical_and(channel1o == 0, channel2o == 0), channel3o == 0)] = channel2o[~np.logical_and(np.logical_and(channel1o == 0, channel2o == 0), channel3o == 0)]
    channel3i[~np.logical_and(np.logical_and(channel1o == 0,channel2o == 0),channel3o == 0)] = channel3o[~np.logical_and(np.logical_and(channel1o == 0,channel2o == 0),channel3o == 0)]
    output_image = np.dstack((channel1i,channel2i,channel3i)).astype(int)

####################################################################################
    #
    # xrecmax=int(max(xy_2pts[1,0,0],xy_2pts[2,0,0]))
    # xrecmax_origspace=xrecmax-img1.shape[1]
    # # yrecmin=int(min(xy_2pts[0,1,0],xy_2pts[1,1,0]))
    # # yrecmin_origspace=yrecmin-img1.shape[0]
    # # yrecmax=int(max(xy_2pts[2,1,0],xy_2pts[3,1,0]))
    # # yrecmax_origspace=yrecmax-img1.shape[0]
    # xrecmin=int(min(xy_2pts[0,0,0],xy_2pts[3,0,0]))
    # xrecmin_origspace=xrecmin-img1.shape[1]
    #
    # if (xrecmax>img1.shape[1] and xrecmin<img1.shape[1]):
    #     img2_subset=img2_orig[:,0:xrecmax_origspace]
    #     e=genEngMap(img2_subset)
    #     Mx, Tbx = cumMinEngVer(e)
    #     Icol, Ecol, array = rmVerSeam(img2_subset, Mx, Tbx)
    #     # print(img2_subset.shape)
    #     # carv(img2_subset,0,1)
    #     # array=np.load("outfile.npy")
    #     # array=np.flip(array,0)
    #     # print("array",array)
    #     for y in range(left_start,left_end):
    #         for x in range(int(array[y,1]),xrecmax_origspace+1):
    #             output_image[y+img1.shape[0],x+img1.shape[1]]=img2_orig[y,x]
    #
    #
    #
    # if (xrecmin<(img1.shape[1]+img2_orig.shape[1]) and xrecmax>(img1.shape[1]+img2_orig.shape[1])):
    #     img2_subset=img2_orig[:,xrecmin_origspace:img2.shape[1]]
    #     e=genEngMap(img2_subset)
    #     Mx, Tbx = cumMinEngVer(e)
    #     Icol, Ecol, array = rmVerSeam(img2_subset, Mx, Tbx)
    #     # print("AAAAArray",array)
    #     # print("right image x")
    #     # print("y in range: 0 to ",img2_orig.shape[0])
    #     print("x in range ",xrecmin_origspace," to ",int(array[0,1])+xrecmin_origspace)
    #     for y in range(rigt_start,rigt_end):
    #         for x in range(xrecmin_origspace,int(array[y,1])+xrecmin_origspace +1):
    #             # print(y,x)
    #             output_image[y+img1.shape[0],x+img1.shape[1]]=img2_orig[y,x]
    #

    # y_top_right=int(xy_2pts[1,1,0])
    # y_top_right_orig_space=y_top_right-img1.shape[0]
    # y_bot_right=int(xy_2pts[2,1,0])
    # y_bot_right_orig_space=y_bot_right-img1.shape[0]
    # y_top_left=int(xy_2pts[0,1,0])
    # y_top_left_orig_space=y_top_left-img1.shape[0]
    # y_bot_left=int(xy_2pts[3,1,0])
    # y_bot_left_orig_space=y_bot_left-img1.shape[0]


    # if(y_top_right>img1.shape[0]):
    #     # print("******************",y_top_right_orig_space)
    #     img2_subset=img2_orig[y_top_right_orig_space:img2_orig.shape[0],:]
    #     e=genEngMap(img2_subset)
    #     My, Tby = cumMinEngHor(e)
    #     Icol, Ecol, array = rmHorSeam(img2_subset, My, Tby)
    #     array=(array).astype(int)
    #     # print(array)
    #     print(y_top_right_orig_space)
    #     # print(array[0,0])
    #     for x in range(0,xrecmax_origspace):
    #         for y in range(0,int(array[x,0])+1):
    #             # print(y,x)
    #             output_image[y+img1.shape[0],x+img1.shape[1]]=img2_orig[y,x]
    #
    # if(y_top_left>img1.shape[0]):
    #     img2_subset=img2_orig[y_top_left_orig_space:img2_orig.shape[0],:]
    #     e=genEngMap(img2_subset)
    #     My, Tby = cumMinEngHor(e)
    #     Icol, Ecol, array = rmHorSeam(img2_subset, My, Tby)
    #     array=(array).astype(int)
    #     print(y_top_left_orig_space)
    #     # print(array[0,0])
    #     for x in range(xrecmin_origspace,img2_orig.shape[1]):
    #         for y in range(0,int(array[x,0])+1):
    #             # print(y,x)
    #             output_image[y+img1.shape[0],x+img1.shape[1]]=img2_orig[y,x]

    # # if(y_bot_right<img1.shape[0]+img2_orig.shape[0]):

    return output_image #output_image to output


if __name__ == "__main__":
   print('demo of the interp2 function')
   x_mesh, y_mesh = np.meshgrid(np.arange(4),np.arange(4))
   print('x, the x meshgrid:')
   print(x_mesh)
   print('y, the y meshgrid:')
   print(y_mesh)
   v = np.arange(16).reshape(4,4)
   print('v, the value located on the coordinates above')
   print(v)
   xq_mesh, yq_mesh = np.meshgrid(np.arange(0,3.5,0.5),np.arange(0,3.5,0.5))
   print('xq_mesh, the query points x coordinates:')
   print(xq_mesh)
   print('yq_mesh, the query points y coordinates:')
   print(yq_mesh)
   interpv = interp2(v,xq_mesh,yq_mesh)
   print('output the interpolated point at query points, here we simply upsample the original input twice')
   print(interpv)

def interp3(v, xq, yq):

	if len(xq.shape) == 2 or len(yq.shape) == 2:
		dim_input = 2
		q_h = xq.shape[0]
		q_w = xq.shape[1]
		xq = xq.flatten()
		yq = yq.flatten()

	h = v.shape[0]
	w = v.shape[1]
	if xq.shape != yq.shape:
		raise Exception('query coordinates Xq Yq should have same shape')

	x_floor = np.floor(xq).astype(np.int32)
	y_floor = np.floor(yq).astype(np.int32)
	x_ceil = np.ceil(xq).astype(np.int32)
	y_ceil = np.ceil(yq).astype(np.int32)

	x_floor[x_floor < 0] = 0
	y_floor[y_floor < 0] = 0
	x_ceil[x_ceil < 0] = 0
	y_ceil[y_ceil < 0] = 0

	x_floor[x_floor >= w-1] = w-1
	y_floor[y_floor >= h-1] = h-1
	x_ceil[x_ceil >= w-1] = w-1
	y_ceil[y_ceil >= h-1] = h-1

	v1 = v[y_floor, x_floor]
	v2 = v[y_floor, x_ceil]
	v3 = v[y_ceil, x_floor]
	v4 = v[y_ceil, x_ceil]

	lh = yq - y_floor
	lw = xq - x_floor
	hh = 1 - lh
	hw = 1 - lw

	w1 = hh * hw
	w2 = hh * lw
	w3 = lh * hw
	w4 = lh * lw

	interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

	if dim_input == 2:
		return interp_val.reshape(q_h, q_w)
	return interp_val
