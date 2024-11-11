import cv2
from deepface import DeepFace
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to RGB format (MediaPipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face detection with MediaPipe
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # Get the bounding box from the detection
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, w_bbox, h_bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            # Extract the face ROI (Region of Interest)
            face_roi = rgb_frame[y:y + h_bbox, x:x + w_bbox]

            # Perform emotion analysis on the face ROI
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Determine the dominant emotion
            emotion = result[0]['dominant_emotion']

            negative_emotions = ['angry', 'fear', 'sad', 'disgust']
            positive_emotions = ['neutral', 'happy', 'surprise']

            if emotion in positive_emotions:
                emotion_category = "positive"
            elif emotion in negative_emotions:
                emotion_category = "negative"

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (x, y), (x + w_bbox, y + h_bbox), (0, 0, 255), 2)
            cv2.putText(frame, emotion_category, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()