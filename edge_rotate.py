import numpy as np
import matplotlib.pyplot as plt
import cv2
import corner

img = cv2.imread("testpiece.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
# Edge detection
edges = cv2.Canny(img, 100, 200)
# Show image and edges
plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()



def rotate_image_horizontal(img, corners):
    """
     Rotate the puzzle piece to be either horizontal or vertical,
    """
    
    dx = corners[1][0] - corners[0][0]
    dy = corners[1][1] - corners[0][1]
    angle = -np.degrees(np.arctan2(dy, dx))  # Negative for clockwise rotation

    
    rotation_center = tuple(corners[0])
    rotation_matrix = cv2.getRotationMatrix2D(rotation_center, angle, 1)

   
    img_height, img_width = img.shape[:2]
    rotated_img = cv2.warpAffine(img, rotation_matrix, (img_width, img_height))

    return rotated_img
    
    
corners = corner.detect_corners_and_fit_rectangle(img)
rotated_image = rotate_image_horizontal(img, corners)


    #sudo code
    # if splines[0,1,2,3] two consecutive outer edge is tru
    #categorize as corner piece
    #if one is true, categorize as outer piece
    #outer edge must allign ex splines[outer, 1, 2, 3] [outer,1 2 3 ]until corner piece comes out

