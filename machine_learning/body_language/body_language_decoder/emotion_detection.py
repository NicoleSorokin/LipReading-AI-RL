# Save this as `emotion_detection.py`

import mediapipe as mp
import cv2
import pandas as pd
import pickle
import numpy as np


def detect_emotion(shared_emotion_val):

    with open('body_lang_model.pkl', 'rb') as f:
        model = pickle.load(f)

    cap = cv2.VideoCapture(0)
    mp_holistic = mp.solutions.holistic
    
    while not cap.isOpened():
        print("Error: Unable to access the camera.")
    try:
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                print("TESTING")

                try:
                
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                    face = results.face_landmarks.landmark
                    face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                    row = pose_row + face_row

                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]

                    if body_language_class.split(' ')[0] == 'Threatening':
                        shared_emotion_val.value = 1
                    elif body_language_class.split(' ')[0] == 'Non-threatening':
                        shared_emotion_val.value = -1

                except Exception as e:
                    print("Error:", e)
                    pass

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    
    except Exception as e:
        print("Error:", e)

    cap.release()
    cv2.destroyAllWindows()
