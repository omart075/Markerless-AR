import math
import cv2
import numpy as np
import argparse
from imutils import contours
import imutils
from skimage.filters import threshold_adaptive
from matplotlib import pyplot as plt

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame, pygame.image
from pygame.locals import *
import pickle
from PIL import Image
from PIL.ExifTags import TAGS
from scipy import linalg


class Camera(object):
    """ Class for representing pin-hole cameras. """

    def __init__(self,P):
        """ Initialize P = K[R|t] camera model. """
        self.P = P
        self.K = None # calibration matrix
        self.R = None # rotation
        self.t = None # translation
        self.c = None # camera center


    def project(self,X):
        """  Project points in X (4*n array) and normalize coordinates. """

        x = np.dot(self.P,X)
        for i in range(3):
          x[i] /= x[2]
        return x

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped

def transformSurface(img):
    image = cv2.imread(img)
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # show the original image and the edge detected image
    #print "STEP 1: Edge Detection"
    # cv2.imshow("Image", image)
    # cv2.imshow("Edged", edged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    # show the contour (outline) of the piece of paper
    #print "STEP 2: Find contours of paper"
    # cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    # cv2.imshow("Outline", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # apply the four point transform to obtain a top-down
    # view of the original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)


    # show the original and scanned images
    #print "STEP 3: Apply perspective transform"
    # cv2.imshow("Original", imutils.resize(orig, height = 650))
    # cv2.imshow("Scanned", imutils.resize(warped, height = 650))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(args['queryImage'], warped)


def drawMatches(img1, kp1, img2, kp2, matches):
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    # Create the output image
    # The rows of the output are the largest between the two images
    # and the columns are simply the sum of the two together
    # The intent is to make this a colour image, so make this 3 channels
    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat[0].queryIdx
        img2_idx = mat[0].trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255,0,0), 1)


    cv2.imwrite("orb.jpg", out)


def orb(img1,img2):
    query = cv2.imread(img1,0) # queryImage
    scene = cv2.imread(img2,0) # trainImage
    h, w = scene.shape[:2]

    # Initiate ORB detector
    orb = cv2.ORB()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(query,None)
    kp2, des2 = orb.detectAndCompute(scene,None)

    # create BFMatcher object
    bf = cv2.BFMatcher()
    #returns list of lists of matches
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    #drawMatches(img1, kp1, img2, kp2, good[:])

    MIN_MATCH_COUNT = 10

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)

        roughM, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()


        '''
        Second pass to refine homography
        '''
        warp = cv2.warpPerspective(scene, roughM, (w, h), flags=cv2.WARP_INVERSE_MAP+cv2.INTER_CUBIC)
        cv2.imwrite("warp.jpg", warp)
        warpedScene = cv2.imread("warp.jpg",0) # queryImage

        # Initiate ORB detector
        orb = cv2.ORB()

        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(query,None)
        kp2, des2 = orb.detectAndCompute(warpedScene,None)

        # create BFMatcher object
        bf = cv2.BFMatcher()
        #returns list of lists of matches
        matches = bf.knnMatch(des1,des2, k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])

        #drawMatches(img1, kp1, img2, kp2, good[:])

        MIN_MATCH_COUNT = 10
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)

            refinedM, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

        resultM = np.matmul(roughM, refinedM)
        '''
        '''

        h,w = query.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,resultM)

        corners = [np.int32(dst)]
        # draws the outline of the query img as it would be found in the scene
        #cv2.polylines(warpedScene,corners,True,255,3, cv2.CV_AA)


    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None


    drawMatches(query,kp1,warpedScene,kp2,good[:])

    return (corners,resultM)

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    cv2.imwrite("3dpoint.jpg", img)


def threeDPoints(corners):
    if corners.any():
        gray = cv2.imread(args['sceneImage'],0)
        h, w = gray.shape[:2]

        calibrations = np.load("calibrate.npz")
        mtx = calibrations['mtx']
        dist = calibrations['dist']
        rvecs = calibrations['rvecs']
        tvecs = calibrations['tvecs']

        flattenR = []
        for x in rvecs:
            for y in x:
                flattenR.append(y)
        flattenR = np.array(flattenR, dtype=np.float32)[:3]

        flattenT = []
        for x in tvecs:
            for y in x:
                flattenT.append(y)
        flattenT = np.array(flattenT, dtype=np.float32)[:3]

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # objp = np.zeros((6*7,3), np.float32)
        # objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        objp = np.zeros((4, 1, 3), np.float32)
        objp[:,:,:2] = np.mgrid[0:2,  0:2].T.reshape(-1,1,2)

        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

        # corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # print corners2

        # Find the rotation and translation vectors.
        rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        draw(gray,corners,imgpts)




def draw_background(imname, sz):
    """  Draw background image using a quad. """

    # load background image (should be .bmp) to OpenGL texture
    bg_image = pygame.image.load(imname).convert()
    bg_image = pygame.transform.scale(bg_image, sz)
    bg_data = pygame.image.tostring(bg_image,"RGBX",1)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # bind the texture
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D,glGenTextures(1))
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,sz[0],sz[1],0,GL_RGBA,GL_UNSIGNED_BYTE,bg_data)
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST)

    # create quad to fill the whole window
    glBegin(GL_QUADS)
    glTexCoord2f(0.0,0.0); glVertex3f(-1.0,-1.0,-1.0)
    glTexCoord2f(1.0,0.0); glVertex3f( 1.0,-1.0,-1.0)
    glTexCoord2f(1.0,1.0); glVertex3f( 1.0, 1.0,-1.0)
    glTexCoord2f(0.0,1.0); glVertex3f(-1.0, 1.0,-1.0)
    glEnd()

    # clear the texture
    glDeleteTextures(1)


def draw_teapot(size):
    """ Draw a red teapot at the origin. """
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_DEPTH_TEST)
    glClear(GL_DEPTH_BUFFER_BIT)

    # draw red teapot
    glMaterialfv(GL_FRONT,GL_AMBIENT,[0,0,0,0])
    glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.5,0.0,0.0,0.0])
    glMaterialfv(GL_FRONT,GL_SPECULAR,[0.7,0.6,0.6,0.0])
    glMaterialf(GL_FRONT,GL_SHININESS,0.25*128.0)
    glutSolidTeapot(size)

def setup():
    """ Setup window and pygame environment. """
    pygame.init()
    display = (800, 747)
    window = pygame.display.set_mode(display,DOUBLEBUF|OPENGL)
    return window

def my_calibration(sz):
    row,col = sz
    fx = 2555*col/2592
    fy = 2586*row/1936
    K = np.diag([fx,fy,1])
    K[0,2] = 0.5*col
    K[1,2] = 0.5*row
    return K

def set_projection_from_camera(K, sz, mtx, aperture):
    """  Set view from a camera calibration matrix. """
    matrix = cv2.calibrationMatrixValues(mtx, sz, aperture[0], aperture[1])

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    fx = K[0,0]
    fy = K[1,1]
    fovy = 2*np.arctan(0.5*sz[1]/fy)*sz[0]/np.pi
    aspect = (sz[0]*fy)/(sz[1]*fx)

    # define the near and far clipping planes
    near = 0.1
    far = 100.0

    # set perspective
    gluPerspective(fovy,aspect,near,far)
    glViewport(0,0,sz[0],sz[1])


def set_modelview_from_camera(Rt):
    """  Set the model view matrix from camera pose. """

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # rotate teapot 90 deg around x-axis so that z-axis is up
    Rx = np.array([[1,0,0],[0,0,-1],[0,1,0]])

    # set rotation to best approximation
    R = Rt[:,:3]
    U,S,V = linalg.svd(R)
    R = np.dot(U,V)
    R[0,:] = -R[0,:] # change sign of x-axis

    # set translation
    t = Rt[:,3]

    # setup 4*4 model view matrix
    M = np.eye(4)
    M[:3,:3] = np.dot(R,Rx)
    M[:3,3] = t

    # transpose and flatten to get column order
    M = M.T
    m = M.flatten()

    # replace model view with the new matrix
    glLoadMatrixf(m)

###############################################################################
###############################################################################


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--queryImage", required=True,
	help="path to input image")
ap.add_argument("-s", "--sceneImage", required=True,
	help="path to input image")
args = vars(ap.parse_args())

'''
Feature matching and pose estimation
'''
query = args["queryImage"]
scene = args["sceneImage"]

# extract receipt from the scene
transformSurface(scene)

gray = cv2.imread(scene,0)
h, w = gray.shape[:2]

# perform feature matching and calculate homography
corners = orb(query,scene)
homography = corners[1]

# R3 = np.cross(homography[:,0], homography[:,1])
# homography = np.insert(homography, 2, R3, axis=1)

'''
'''


'''
Calibration
'''
# format corners for solvePnP
imgpts = corners[0][0]
newpts = []
for x in imgpts:
    pts = (x[0][0], x[0][1])
    newpts.append(pts)
newpts = np.array(newpts, dtype="double")

# threeD = []
# for x in imgpts:
#     pts = np.append(x,[1])
#     pts = [np.matmul(np.linalg.inv(homography), pts)]
#     pts = (pts[0][0], pts[0][1], pts[0][2])
#     threeD.append(pts)
# print threeD
# print newpts

# initialize world coordinates based on corners found with homography
threeD = [(0,0,0), (newpts[1][0],newpts[1][1],0), (newpts[2][0],newpts[2][1],0),(newpts[3][0],newpts[3][1],0)]
threeD = np.array(threeD)

#threeDPoints(np.array(corners[0][0], dtype=np.float32))



calibrations = np.load("calibrate.npz")
mtx = calibrations['mtx'] #camera intrinsic parameters
dist = calibrations['dist']
rvecs = calibrations['rvecs'] #camera extrinsic parameters
tvecs = calibrations['tvecs'] #camera extrinsic parameters

# calulcate rotation and translation vectors
retval, rvecs, tvecs = cv2.solvePnP(threeD, newpts, mtx, dist, flags=cv2.CV_ITERATIVE)

# draw 3d lines showing pose of receipt
(point2D, jacobian) = cv2.projectPoints(threeD, rvecs, tvecs, mtx, dist)

for p in newpts:
    cv2.circle(gray, (int(p[0]), int(p[1])), 20, (0,0,255), -1)

p1 = ( int(newpts[0][0]), int(newpts[0][1]))
p2 = ( int(point2D[0][0][0]), int(point2D[0][0][1]))
cv2.line(gray, p1, p2, (255,0,0), 2)

p1 = ( int(newpts[1][0]), int(newpts[1][1]))
p2 = ( int(point2D[1][0][0]), int(point2D[1][0][1]))
cv2.line(gray, p1, p2, (255,0,0), 2)

p1 = ( int(newpts[2][0]), int(newpts[2][1]))
p2 = ( int(point2D[2][0][0]), int(point2D[2][0][1]))
cv2.line(gray, p1, p2, (255,0,0), 2)

p1 = ( int(newpts[3][0]), int(newpts[3][1]))
p2 = ( int(point2D[3][0][0]), int(point2D[3][0][1]))
cv2.line(gray, p1, p2, (255,0,0), 2)

cv2.imwrite("3dprojection.jpg", gray)

# [R]
rvecs = cv2.Rodrigues(rvecs)[0]

# [R|t] matrix
exMatrix = np.concatenate((rvecs, tvecs), axis=1)

# K[R|t]
projMatrix = np.matmul(mtx, exMatrix)



K = my_calibration((3001, 4011))
cam1 = Camera( np.hstack((K,np.dot(K,np.array([[0],[0],[-1]])) )) )
cam2 = Camera(np.dot(corners[1],cam1.P))

A = np.dot(linalg.inv(K),cam2.P[:,:3])
A = np.array([A[:,0],A[:,1],np.cross(A[:,0],A[:,1])]).T

cam2.P[:,:3] = np.dot(K,A)

Rt = np.dot(linalg.inv(K),cam2.P)
'''
'''



'''
3d rendering
'''

# read Exif header for aperture
img = Image.open(scene)
exif_data = img._getexif()
ret = {}
for tag, value in exif_data.items():
        decoded = TAGS.get(tag, tag)
        ret[decoded] = value

aperture = ret['ApertureValue']

#K = mtx

# make .bmp file of scene for OpenGL
img = Image.open(scene)
img.save( '3dpoint.bmp', 'bmp')

#render object
sz = (800, 747)
window = setup()
draw_background("3dpoint.bmp", sz)
set_projection_from_camera(K,sz, mtx, aperture)
set_modelview_from_camera(Rt)
draw_teapot(0.2)

while True:
    event = pygame.event.poll()
    if event.type in (QUIT,KEYDOWN):
        # pygame.image.save(window, "screenshot.jpeg")
        break
    pygame.display.flip()
    pygame.time.wait(1)
