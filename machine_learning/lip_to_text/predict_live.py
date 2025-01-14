import os
import cv2
import dlib
import math
import json
import statistics
from PIL import Image
import imageio.v2 as imageio
import numpy as np
import csv
from collections import deque
import tensorflow as tf
import sys

sys.path.append('../data')
from constants import *

label_dict = {6: 'hello', 5: 'dog', 10: 'my', 12: 'you', 9: 'lips', 3: 'cat', 11: 'read', 0: 'a', 4: 'demo', 7: 'here', 8: 'is', 1: 'bye', 2: 'can'}
count = 0

# Define the input shape
input_shape = (TOTAL_FRAMES, 80, 112, 3)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling3D((2, 2, 2)),
    tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
    tf.keras.layers.MaxPooling3D((2, 2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(label_dict), activation='softmax')
])

model.load_weights('model/model_weights.h5', by_name=True)

# Load the detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/face_weights.dat")

# Setup for video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Failed to open the video stream.")
    sys.exit(1)

# Initialize variables
curr_word_frames = []
not_talking_counter = 0
past_word_frames = deque(maxlen=PAST_BUFFER_SIZE)
predicted_word_label = None
spoken_already = []
draw_prediction = False
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

        landmarks = predictor(image=gray, box=face)

        # Calculate lip region dimensions
        lip_top = landmarks.part(50).y
        lip_bottom = landmarks.part(58).y
        lip_left = landmarks.part(48).x
        lip_right = landmarks.part(54).x
        
        width_diff = LIP_WIDTH - (lip_right - lip_left)
        height_diff = LIP_HEIGHT - (lip_bottom - lip_top)
        
        pad_left = width_diff // 2
        pad_right = width_diff - pad_left
        pad_top = height_diff // 2
        pad_bottom = height_diff - pad_top

        # Ensure no padding exceeds frame bounds
        lip_frame = frame[max(lip_top - pad_top, 0):lip_bottom + pad_bottom, max(lip_left - pad_left, 0):lip_right + pad_right]
        lip_frame = cv2.resize(lip_frame, (LIP_WIDTH, LIP_HEIGHT))

        # Check for invalid lip_frame
        if lip_frame is None or lip_frame.size == 0:
            continue

        lip_frame_lab = cv2.cvtColor(lip_frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lip_frame_lab)

        # Apply CLAHE (contrast limited adaptive histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
        l_channel_eq = clahe.apply(l_channel)
        lip_frame_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
        lip_frame_eq = cv2.cvtColor(lip_frame_eq, cv2.COLOR_LAB2BGR)

        # Further preprocessing
        lip_frame_eq = cv2.GaussianBlur(lip_frame_eq, (7, 7), 0)
        lip_frame_eq = cv2.bilateralFilter(lip_frame_eq, 5, 75, 75)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        lip_frame_eq = cv2.filter2D(lip_frame_eq, -1, kernel)
        lip_frame_eq = cv2.GaussianBlur(lip_frame_eq, (5, 5), 0)

        lip_frame = lip_frame_eq

        # Drawing the mouth landmarks on the frame
        for n in range(48, 61):
            x, y = landmarks.part(n).x, landmarks.part(n).y
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        lip_distance = math.hypot(landmarks.part(57).x - landmarks.part(51).x, landmarks.part(57).y - landmarks.part(51).y)

        # Continuously make predictions when speaking
        if lip_distance > 45:  # Talking
            curr_word_frames.append(lip_frame.tolist())
            not_talking_counter = 0
            draw_prediction = False
            
            # Make a prediction every few frames for dynamic transcription
            if len(curr_word_frames) >= TOTAL_FRAMES:  # Ensure enough frames are collected
                curr_data = np.array([curr_word_frames[:input_shape[0]]])

                prediction = model.predict(curr_data)
                prob_per_class = [(prediction[0][i], label_dict[i]) for i in range(len(prediction[0]))]
                sorted_probs = sorted(prob_per_class, key=lambda x: x[0], reverse=True)
                
                # Display top predictions
                print("Predictions: ")
                for prob, label in sorted_probs:
                    print(f"{label}: {prob:.3f}")

                predicted_class_index = np.argmax(prediction)
                while label_dict[predicted_class_index] in spoken_already:
                    prediction[0][predicted_class_index] = 0
                    predicted_class_index = np.argmax(prediction)
                
                predicted_word_label = label_dict[predicted_class_index]
                spoken_already.append(predicted_word_label)

                print("Predicted word:", predicted_word_label)
                draw_prediction = True

                # Reset for the next prediction cycle
                curr_word_frames = []
        
        # If person stops talking, reset counter and stop displaying predictions
        if lip_distance <= 45:
            not_talking_counter += 1
            if not_talking_counter >= NOT_TALKING_THRESHOLD:
                curr_word_frames = []

    if draw_prediction and count < 20:
        count += 1
        cv2.putText(frame, predicted_word_label, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

    cv2.imshow("Mouth", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        spoken_already = []

    # Exit when escape is pressed
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
