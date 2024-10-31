import scipy.ndimage.filters as filters
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


img = cv2.imread("testpiece.png")

operatedImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 


operatedImage = np.float32(operatedImage) 
  
dest = cv2.cornerHarris(operatedImage, 3, 5, 0.04) 
  
dest = cv2.dilate(dest, None) 

img[dest > 0.01 * dest.max()]=[0, 0, 255] 

cv2.imshow('Image with Borders', img) 
cv2.waitKey(0)

