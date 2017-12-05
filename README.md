# Markerless-AR
Markerless AR is the process of rendering a 3d object into a real world environment without the need of a marker.

# How it Works:
There are a few things that need to be done in order to render 3d objects without a marker:
  * Find planar object
  * Feature Matching
  * Find Homography
  * Calculate camera intrinsic and extrinsic values
  * Make above information understandable for OpenGL

# 1. Find planar object:
  * Given a scene image (or frame if using a video), we must find a planar object (magazine, paper, book, etc.) that we will use to attach our 3d object to like we would with a marker. It seems just as restricting as having to rely on a marker but the main difference is that these planar objects exist naturally in the real world whereas markers do not.

  * Using Canny Edge Detector and then finding the contours, we can get the outline of our planar object which we can then crop out. We then apply a perspective transform in order to get a top-down view of our object in case it has some rotation or translation in the scene. The result of this process will be our query image (planar object), which we will use to perform the feature detection and matching.

# 2. Feature Matching:
  * We want to be able to find and keep track of where our planar object is as we move around in the real world. Using feature matching can help solve this problem. By detecting keypoints in our query and scene image, we can be sure that we only look for parts of the images that contain a lot of information. The reason for this is that we only have to match on those keypoints as opposed to the entire image, which makes finding our planar object a lot easier.



# Instructions for Use:
  1. Make sure you have the following installed:
      * [WeeChat](https://weechat.org/files/doc/weechat_faq.en.html#compile_osx)
      * [Matrix](https://github.com/torhve/weechat-matrix-protocol-script/blob/master/README.md) 
      * [Terminal-Notifier](https://github.com/julienXX/terminal-notifier)
      
  2. Clone this repo:
  ```     
    git clone https://github.com/omart075/WeeNotifyMatrix.git    
  ``` 
  3. Move script to WeeChat's Python Directory:
  
  ```
    mv weeNotify.py ~/.weechat/python
  ```  
  4. Go into WeeChat's autoload directory for Python:
  
  ```
    cd ~/.weechat/python/autoload
  ```  
  5. Make a link to script from Weechat's Python dir to WeeChat's autoload dir:
  
  ```
    ln -s ../weeNotify.py
  ```
