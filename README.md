# Markerless-AR
Markerless AR is the process of rendering a 3d object into a real world environment without the need of a [marker](https://docs.opencv.org/3.1.0/d5/dae/tutorial_aruco_detection.html).

# Usage:
AR Using an Image:
```
python arImage.py -s sceneImageName
```
AR Using Video:
```
python arImage.py -s sceneVideoName
```
Note: Make sure the image or video exists within the imgs folder.

# How it Works:
There are a few things that need to be done in order to render 3d objects without a marker:
  * Find Planar Object
  * Feature Matching
  * Find Homography
  * Calculate Camera Intrinsic and Extrinsic Matrices
  * Tie it all together

# 1. Find Planar Object:
  * Given a scene image (or frame if using a video), we must find a planar object (magazine, paper, book, etc.) that we will use to attach our 3d object to like we would with a marker. It seems just as restricting as having to rely on a marker but the main difference is that these planar objects exist naturally in the real world whereas markers do not.

  * Using Canny Edge Detector and then finding the contours, we can get the outline of our planar object which we can then crop out. We then apply a perspective transform in order to get a top-down view of our object in case it has some rotation or translation in the scene. The result of this process will be our query image (planar object), which we will use to perform the feature detection and matching.

# 2. Feature Matching:
  * We want to be able to find and keep track of where our planar object is as we move around in the real world. Using feature matching can help solve this problem. By detecting keypoints in our query and scene image, we can be sure that we only look for parts of the images that contain a lot of information. The reason for this is that we only have to match on those keypoints as opposed to the entire image, which makes finding our planar object a lot easier.
  
  * There are a few different algorithms used for feature detecting such as ORB, SURF, SIFT. In this project we use ORB. Once the keypoints are found, we extract the descriptors. The descriptor is a vector that contains information about the feature point. 
  
  * The matching of feature points is done using a knnMatch which does a search of the nearest neighbor from one set of descriptors to another set. Afterwards, a ratio test is applied to filter out outliers based on distance between the two matches being compared.
  
# 3. Find Homography
  * With the above filtered set of matched feature points, we can now calculate the homography. A homography is a 3x3 transformation matrix that maps the points of one image to another if they are part of the same planar surface:
  
  * If we wanted to, we could refine the homography by applying a warping perspective on the scene using the homography and reapplying the feature matching and descriptor extraction. 
  
# 4. Calculate Camera Intrinsic and Extrinsic Matrices
  * The camera intrinsic matrix (referred to as K) is a 3x3 matrix used to transform 3d camera coordinates to 2d homogeneous image coordinates.  
  ![Alt text](/imgs/K_initial.jpg?raw=true "Camera Intrinsics")  
  The matrix is made up of the focal length (fx and fy), the principal point offsets (cx and cy), and axis skew (s), which in most cases can be set to 0. This gives us:    
  ![Alt text](/imgs/K_final.jpg?raw=true "Camera Intrinsics")  
  This matrix can be calculated based on some assumptions but the focal length must be found with a specific calibration method. You can also obtain the entire matrix by calibrating the camera with a chessboard, which OpenCV provides a built in function just for that.
  
  * The camera extrinsic matrix ([R|t]) is a 3x4 matrix which describes the camera's position in the real world and the direction it is pointing in.  
  ![Alt text](/imgs/Rt.jpg?raw=true "Camera Intrinsics")  
  The matrix has two components: a 3x3 rotation matric (R) and a 3x1 translation matrix (t). This matrix can be extracted from the homography or using K and built in OpenCV functions.
  
# Tie it all together:
  * Once we have all the components found above, we must format it so that OpenGL knows what to do with everything. This is done with the Python wrapper for OpenGL, PyOpenGL.
