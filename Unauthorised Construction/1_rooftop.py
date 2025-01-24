import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from keras_cv.models.segmentation import DeepLabV3Plus
from keras_cv.models import ResNet50V2Backbone
from tensorflow.keras.models import load_model
from keras.utils import custom_object_scope

# Define the custom objects for loading the model
def custom_objects():
    return {
        'DeepLabV3Plus': DeepLabV3Plus,
        'ResNet50V2Backbone': ResNet50V2Backbone,
    }

# Load the trained model with custom objects
def load_trained_model(model_path):
    with custom_object_scope(custom_objects()):  # Ensure custom layers are recognized
        model = load_model(model_path)
    return model

# Load and preprocess an image
def read_image_mask(image_path, size=(256, 256), mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.io.decode_image(image, channels=1)
        image = tf.image.resize(image, size, method="bicubic")
        image = tf.cast(image, tf.uint8)
    else:
        image = tf.io.decode_image(image, channels=3)
        image = tf.image.resize(image, size, method="bicubic")
        image = tf.cast(tf.clip_by_value(image, 0., 255.), tf.float32)
    return image

# Function to overlay a segmentation map on top of an RGB image
def image_overlay(image, segmented_image, color=(128, 0, 0), alpha=0.7):
    image = image.astype(np.uint8)
    segmented_image = segmented_image.astype(np.uint8)  # Ensure mask is uint8
    color_mask = np.zeros_like(image)
    color_mask[segmented_image == 1] = color
    overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    return overlay

# Display the image, mask, and overlayed image
def display_image_and_mask(data_list, title_list, figsize):
    original_image, mask, overlayed_image = data_list
    rgb_gt_mask = np.zeros(original_image.shape[:2] + (3,))
    rgb_gt_mask[mask == 1] = (128, 0, 0)  # Apply the red color for the mask

    fig, axes = plt.subplots(nrows=1, ncols=len(data_list), figsize=figsize)
    for idx, axis in enumerate(axes.flat):
        axis.set_title(title_list[idx])
        if title_list[idx] == "GT Mask":
            axis.imshow(rgb_gt_mask)
        elif title_list[idx] == "Overlayed Prediction":
            axis.imshow(overlayed_image)
        else:
            axis.imshow(original_image)
        axis.axis('off')
    plt.show()

# Prepare for inference
def inference(model, image_path, num_classes=2):
    image = read_image_mask(image_path)
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    pred = model.predict(image)
    pred = np.argmax(pred, axis=-1)[0]  # Get the class with the highest probability
    overlayed_image = image_overlay(image[0].numpy().astype(np.uint8), pred)
    return image[0].numpy().astype(np.uint8), pred, overlayed_image

# Define the path to your trained model and image
model_path = r"C:\Users\Utkarsh Sah\OneDrive\Desktop\send_moddel\send_moddel\hopeitworks.h5"
test_image_path = r"C:\Users\Utkarsh Sah\OneDrive\Desktop\send_moddel\send_moddel\austin1_142.png"

# Load the model
model = load_trained_model(model_path)

# Run inference
original_image, pred_mask, overlayed_image = inference(model, test_image_path)

# Display results
display_image_and_mask([original_image, pred_mask, overlayed_image], ["Image", "GT Mask", "Overlayed Prediction"], figsize=(16, 6))
