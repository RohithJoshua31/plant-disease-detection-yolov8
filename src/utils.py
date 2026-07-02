"""
utils.py

Utility functions used across the project.
"""

import os
import cv2


def create_directory(path):
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def image_exists(image_path):
    """
    Check whether an image exists.
    """
    return os.path.isfile(image_path)


def load_image(image_path):
    """
    Load an image using OpenCV.
    """
    return cv2.imread(image_path)


def show_image(window_name, image):
    """
    Display an image.
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(output_path, image):
    """
    Save an image.
    """
    cv2.imwrite(output_path, image)
