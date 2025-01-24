import cv2
import numpy as np
from math import log10, copysign

# Read image as a grayscale image
im = cv2.imread('hu.png', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if im is None:
    raise ValueError("Image not found or unable to load.")

# Threshold image to get binary image
_, binary_im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)

# Detect contours (which represent shapes)
contours, _ = cv2.findContours(binary_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dictionary to store Hu Moments for each shape
shape_hu_moments = {}

# Iterate over each contour (each shape)
for i, contour in enumerate(contours):
    # Calculate Moments for the current shape
    moments = cv2.moments(contour)
    
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)
    
    # Log scale Hu moments 
    for j in range(0, 7):
        # Convert to scalar
        huMoment_value = huMoments[j][0]

        # Ensure the value is positive and non-zero to avoid log domain errors
        if huMoment_value != 0:
            huMoments[j] = -1 * copysign(1.0, huMoment_value) * log10(abs(huMoment_value))
        else:
            huMoments[j] = 0  # Handle log(0) case by setting it to 0 or some other appropriate value
    
    # Assign the Hu Moments to the shape number (i+1) in the dictionary
    shape_hu_moments[i + 1] = huMoments

# Print the Hu Moments for each shape
print("Hu Moments for each shape (Log-Scaled):")
for shape_number, huMoments in shape_hu_moments.items():
    print(f"Shape {shape_number}:")
    for j in range(0, 7):
        print(f"  Hu Moment {j+1}: {huMoments[j][0]}")


