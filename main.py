import cv2
import numpy as np

# Read the image
img = cv2.imread('AgAthon\pples_in_tree.jpg')

# Convert to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#detect number of blobs
detector = cv2.SimpleBlobDetector_create()
 
# Detect blobs.
keypoints = detector.detect(img)
blob_num = len(keypoints)
print(blob_num)

# Define range of red color in HSV
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])

# Threshold the image to get only red colors
mask = cv2.inRange(hsv, lower_red, upper_red)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Define the radius of the enclosing circle
radius = 200

# Iterate through the contours
for cnt in contours:
    # Find the minimum enclosing circle
    (x, y), r = cv2.minEnclosingCircle(cnt)

    # if the radius of the enclosing circle is less than the defined radius
    if r < radius:
        # Draw the circle on the image
        cv2.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 2)

# Draw the contours on the original image
cv2.drawContours(img, contours, -1, (0,0,255), 3)

# Show the final image
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
