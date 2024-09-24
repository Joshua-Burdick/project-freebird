import cv2
import sys

image = cv2.imread(sys.argv[1])
cv2.imshow('Image Window', image)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()  # Close the window