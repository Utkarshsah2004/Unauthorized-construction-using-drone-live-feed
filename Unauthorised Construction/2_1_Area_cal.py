import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

image_path = '1.jpg'
img = cv2.imread(image_path, 0)
_, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

def region_growing(img, seed_point, threshold=0):
    h, w = img.shape
    seed_value = img[seed_point]
    visited = np.zeros((h, w), dtype=bool)
    cluster = []
    stack = [seed_point]
    visited[seed_point] = True
    
    while stack:
        x, y = stack.pop()
        cluster.append((x, y))
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny]:
                if abs(int(img[nx, ny]) - int(seed_value)) <= threshold:
                    visited[nx, ny] = True
                    stack.append((nx, ny))
    
    return cluster

seed_points = [(167 , 156), (168 , 74), (257 , 71), (325 , 72),
          (407 , 75), (475 , 77), (473 , 161), (473 , 212),
          (472 , 263), (406 , 153), (468 , 356), (392 , 265),
          (305 , 153),(286 , 207), (178 , 204), (281 , 266),
          (353 , 353), (216 , 356), (165 , 357)]

indian_names = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Krishna", "Ishaan", 
    "Arjun", "Sai", "Anaya", "Diya", "Lakshmi", "Meera", 
    "Aarohi", "Aanya", "Isha", "Aadhya", "Riya", "Neha", 
    "Reyansh", "Kavya", "Rohan", "Saanvi", "Aayush"
]

h, w = binary_img.shape
valid_seed_points = [point for point in seed_points if 0 <= point[0] < h and 0 <= point[1] < w]
random.shuffle(indian_names)

cluster_areas = []
for seed in valid_seed_points:
    cluster = region_growing(binary_img, seed)
    cluster_areas.append(len(cluster))
    for x, y in cluster:
        binary_img[x, y] = 128

plt.imshow(binary_img, cmap='gray')
plt.title("Regions Grown from Seed Points")
plt.show()

for i, area in enumerate(cluster_areas):
    print(f"Area of cluster {i+1} (named {indian_names[i]}): {area} pixels")
