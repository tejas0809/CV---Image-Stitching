### CIS581: Computer Vision and Computational Photography
### Project 3 Image Stitching

The objective of the project was to implement image stitching. Image stitching or photo stitching is the process of combining multiple photographic images with overlapping fields of view to produce a segmented panorama or high-resolution image.

## File structure
- <b>corner_detector.py</b>
  Used to identify features in an image using Harris corners

- <b>anms.py</b>
  Used to loop through all the feature points, and for each feature point, compare the corner strength (obtained from corner detector) to all the other feature points and keep track of the minimum distance to a larger magnitude feature point. And eventually return the co-ordinates of top N uniformly spread out points.

- <b>feat_desc.py</b>
  After obtaining the 'N' uniformly spread out points of high corner strength from anms, this function describes a feature window around each of those points which could be used to find the same corresponding feature in image 2. This descriptor is built by taking a window of 40 * 40 around the point and then sampling out the pixels with maximum intensity from 5 * 5 regions in that window moving along with a stride of 5. This gives a 8 * 8 descriptor for each point which when linearized gives a 64 length vector for each point.

- <b>feat_match.py</b>
  Used to match the best matching features in other image, given one image. We use Annoy(Approximate Nearest neighbours Oh yeah ) and used the ratio test to get strongly matched points.

- <b>ransac_est_homography.py</b>
  Used to compute the best homography estimate and pull out a minimal set of best features from the matched features by using the Random Sampling Consensus (RANSAC) algorithm. We do this by iterating multiple times, and for each iteration, calculating four random points, finding out the homography using those points and counting the number of inliers that agree with the current homography estimate. After repeated trials, the homography estimate with the largest number of inliers and the minimum least-square estimate is returned.

- <b>est_homography.py</b>
  Used to find out the homography given a set of source and destination points.

- <b>mymosaic.py</b>
  This function stitches the three images together using the method of seam carving. We first stitch the left and middle image and then on this resultant image stitch the right image. The method of seam carving implemented in project 2B (functions put in the utils file) were used to to get the minimum energy seam (both in vertical and horizontal directions) in the overlapping areas along which the two images are then stitched. The seam carving methods removes the abrupt edges formed in the transition from left to middle to right image and smoothens the stitched boundaries. Before padding the left image, we pad the middle image with dimensions of left image and use this same padding when stitiching right image (i.e. the padding is done only once in the enitre mosaicing).

- <b>utils.py</b>
  To modularise the code, we have used the utils.py wherein we have written functions which are called in the aforementioned files. The functions in utils are as follows:
    GaussianPDF_2D - used to generate a 2d gaussian for convolving
    patch_mat - used to vectorize the process of getting the 8 * 8 feature descriptor around a point in the feat\_desc function
    interp2 - used to get intensity values at non-integer pixel values
    plot_image_matching2 - plots two images with red lines indicating the inlier features after ransack and blue lines show outliers
    seam_carving functions - functions from project 2B: genEngMap, cumEngVer/Hor, rmVer/HorSeam used for the seam carving process
    stitch - used to join the images to get the final outpu





## Running the code
- Simply use 'python3 run.py' for running the code.
- Please make sure the required dependencies are installed.
- Images 1,2 and 3 can be changed by changing the name of the images in line 13, 16, 49, 52, 85, 86 and 87 of run.py
- After running corner detector, in ANMS, we filter some of the points with a threshold (which in our case now is 0.01*max(points in cimg) ), which will remove all points with corner strength less than this threshold. This can be changed in line 23 of anms.py
-In anms, we return the top "max_point" points, which can be changed in line 20,24, 56, 60 of run.py
-The threshold for RANSAC can also be changed in line 45 and line 81 of run.py
