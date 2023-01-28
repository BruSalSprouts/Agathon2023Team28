import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#Reads file and saves to variables
imgL = cv.imread("left.jpg", 0)
imgR = cv.imread("right.jpg", 0)

#Set constants of computation and computes using image varibles
stereo = cv.StereoBM_create(numDisparities=512, blockSize=7)
disparity = stereo.compute(imgL,imgR)

#Prints graph to screen
plt.imshow(disparity,'gray')
plt.show()
