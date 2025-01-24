import cv2
import numpy as np


img1 = cv2.imread('2.jpg')
img2 = cv2.imread('3.jpg')


if img1 is None:
    raise FileNotFoundError("Image1 not found or could not be loaded.")
if img2 is None:
    raise FileNotFoundError("Image2 not found or could not be loaded.")


if img1.shape != img2.shape:
    raise ValueError("Images must have the same dimensions for XNOR operation.")


gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


xnor_result = np.bitwise_not(np.bitwise_xor(gray1, gray2))


false_mask = np.where(xnor_result == 0)


result_img = img1.copy()
result_img[false_mask] = [0, 0, 255]  # Red color in BGR


cv2.imwrite('xnor_result.jpg', result_img)


cv2.imshow('XNOR Result', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

