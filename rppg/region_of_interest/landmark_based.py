import cv2
import numpy as np
import os

# Function to extract eyes and mouth regions
def extract_eyes_mouth(image, landmarks, dilation_factor=5):
    left_eye_points = landmarks[0][0][36:42]  # Extract left eye landmarks
    right_eye_points = landmarks[0][0][42:48]  # Extract right eye landmarks
    mouth_points = landmarks[0][0][48:68]  # Extract mouth landmarks

    # Create masks for eyes and mouth
    eye_mask = np.zeros_like(image, dtype=np.uint8)
    mouth_mask = np.zeros_like(image, dtype=np.uint8)

    # Draw polygons for eyes and mouth regions on masks
    cv2.fillPoly(eye_mask, np.int32([left_eye_points]), (255, 255, 255))
    cv2.fillPoly(eye_mask, np.int32([right_eye_points]), (255, 255, 255))
    cv2.fillPoly(mouth_mask, np.int32([mouth_points]), (255, 255, 255))

    # Apply dilation to the masks
    eye_mask = cv2.dilate(eye_mask, None, iterations=int(dilation_factor))
    mouth_mask = cv2.dilate(mouth_mask, None, iterations=int(dilation_factor))

    # Apply masks to extract eyes and mouth regions
    extracted_eyes = cv2.bitwise_and(image, eye_mask)
    extracted_mouth = cv2.bitwise_and(image, mouth_mask)

    return extracted_eyes, extracted_mouth

