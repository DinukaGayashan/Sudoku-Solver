
import cv2
import numpy as np
from skimage.segmentation import clear_border

image = cv2.imread("test08.png")
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
padding_top = 20
padding_bottom = 20
padding_left = 20
padding_right = 20

# Specify border type (you can choose from cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, etc.)
border_type = cv2.BORDER_CONSTANT

# Specify padding color (default is black)
padding_color = (0, 0, 0)

# Add padding to the image using cv2.copyMakeBorder
padded_image = cv2.copyMakeBorder(image, padding_top, padding_bottom, padding_left, padding_right, border_type, value=padding_color)

cv2.imshow('image', padded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()