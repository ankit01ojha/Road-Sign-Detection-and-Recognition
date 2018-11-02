# import the necessary packages
import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt
import glob
from audioOutput import audioOutput

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True ,help = "path to the image")
ap.add_argument("-f", "--file" , required = True ,help = "path to the file where all the queryImages are stored")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

img_hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# lower mask (0-10)
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

# join my masks
mask = mask0+mask1

# set my output img to zero everywhere except my mask
output_img = image.copy()
output_img[np.where(mask==0)] = 0

#HSV 
output_hsv = img_hsv.copy()
output_hsv[np.where(mask==0)] = 0

cv2.imshow('redimage', output_img)
cv2.waitKey(0)

img = cv2.Canny(output_hsv,100,200)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,1000,
                            param1=50,param2=30,minRadius=0,maxRadius=0)

for i in circles[0,:]:
    i[2]=i[2]+4
    cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2)

cv2.imshow('image',image)
cv2.waitKey(0)

x = circles[0][0][0]
y = circles[0][0][1]
r = circles[0][0][2]

startX = int(x-r)
startY = int(y-r)
endX = int(x+r)
endY = int(y+r)
cropped = image[startY:endY, startX:endX]

cv2.imshow('cropped',cropped)
cv2.waitKey(0)

#Image comparision
dim = (250,250)
cropped = cv2.resize(cropped,dim, interpolation = cv2.INTER_AREA)
#img1 = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)   # queryImage
#img1 = cv2.Canny(img1,100,200)
img1 = cropped
good_matches = []
i = 0

for imagePath in glob.glob(args["file"] + "/*.jpeg"):
	img2 = cv2.imread(imagePath) # trainImage
	#img2 = cv2.Canny(img2,100,200)
	print (imagePath)
	method = 'ORB'  # 'SIFT'
	lowe_ratio = 0.77
	magic_number = 0.85

	if method   == 'ORB':
	    finder = cv2.ORB_create()
	elif method == 'SIFT':
	    finder = cv2.xfeatures2d.SIFT_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = finder.detectAndCompute(img1,None)
	kp2, des2 = finder.detectAndCompute(img2,None)

	# BFMatcher with default params
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2, k=2)

	# Apply ratio test
	good = []

	for m,n in matches:
	    if m.distance < lowe_ratio*n.distance:
	        good.append([m])

	img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good, None, flags=2)
	good_matches.append(len(good))
	cv2.imshow('Matched',img3)
	cv2.waitKey(0)

print(good_matches)
matched_image = max(good_matches)
for i in range(0,len(good_matches)):
	if good_matches[i] == matched_image:
		indexOfImage = i
		break
print(indexOfImage)
audioOutput(indexOfImage)