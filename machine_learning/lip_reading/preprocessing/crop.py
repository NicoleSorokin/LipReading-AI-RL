import shutil
import dlib
import cv2
import os

# Constants
INPUT_PATH = './dataset/'
CROPPED_PATH = './cropped/'
OUTPUT_PATH = './output/'
PREDICTOR_PATH = './predictors/shape_predictor_68_face_landmarks_GTX (1).dat'
CROPPED_SIZE = (200, 100)

# Dlib face detector and landmark predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(PREDICTOR_PATH)

class VideoProcessor:
    def __init__(self, video_name):
        self.video_name = video_name
        self.video_path = os.path.join(INPUT_PATH, video_name)
        self.frames = []
        self.landmarks = []
        self.cropped = []

    def load_frames(self):
        """Load video frames into the frames attribute."""
        video = cv2.VideoCapture(self.video_path)
        frame_buffer = []

        while True:
            success, frame = video.read()
            if not success:
                break
            frame_buffer.append(frame)
        
        video.release()
        self.frames = frame_buffer

    def detect_landmarks(self):
        """Detect landmarks for each frame and store them in the landmarks attribute."""
        if not self.frames:
            raise ValueError("Frames not loaded. Call load_frames() first.")
        
        landmarks_buffer = []
        for frame in self.frames:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray_frame)

            if faces:
                landmark = landmark_predictor(gray_frame, faces[0])
                landmarks_buffer.append(landmark)
            else:
                landmarks_buffer.append(None)  # No face detected in this frame

        self.landmarks = landmarks_buffer

    def crop_mouth(self, size, x_margin, y_margin):
        if not self.landmarks:
            raise ValueError("Landmarks not detected. Call detect_landmarks() first.")

        cropped_frames = []
        
        # Initial reference point (optional: can be dynamically set)
        last_center = None
        
        for i, landmark in enumerate(self.landmarks):
            if landmark is None:
                continue

            lip_landmark = [landmark.part(n) for n in range(48, 68)]

            # Calculate center of the lips
            lip_center_x = int(sum(point.x for point in lip_landmark) / len(lip_landmark))
            lip_center_y = int(sum(point.y for point in lip_landmark) / len(lip_landmark))

            # Set a stable reference point using the last center
            if last_center is None:
                last_center = (lip_center_x, lip_center_y)
            else:
                # Smoothly update the reference point
                last_center = (
                    int(last_center[0] * 0.7 + lip_center_x * 0.3),
                    int(last_center[1] * 0.7 + lip_center_y * 0.3)
                )

            # Define cropping area based on the stable reference point
            left = int(last_center[0] - x_margin)
            right = int(last_center[0] + x_margin)
            top = int(last_center[1] - y_margin)
            bottom = int(last_center[1] + y_margin)


            cropped_frame = self.frames[i][top:bottom, left:right]
            cropped_frame = cv2.resize(cropped_frame, size, interpolation=cv2.INTER_CUBIC)

            cropped_frames.append(cropped_frame)

        self.cropped = cropped_frames
        print(f"Cropped frames count: {len(self.cropped)}")

    #For visualizing landmarks
    def overlay_landmarks(self):
        """Draw landmarks on frames."""
        if not self.frames or not self.landmarks:
            raise ValueError("Frames and landmarks must be loaded and detected first.")
        
        for frame, landmarks in zip(self.frames, self.landmarks):
            if landmarks:
                for n in range(68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    def save_video(self, type='normal'):
        if type == 'cropped':
            if not self.cropped:
                raise ValueError("No cropped frames to save. Call crop_mouth(self, size, x_margin, y_margin) first")
            frames = self.cropped
        else:
            if not self.frames:
                raise ValueError("No frames to save. Call load_frames() first.")
            frames = self.frames
        
        height, width, layers = frames[0].shape
        output_path = os.path.join(OUTPUT_PATH, self.video_name.replace('.mpg', '.mp4'))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
        for frame in frames:
            out.write(frame)

        out.release()
        print(f"Video '{self.video_name}' created successfully at {output_path}")

if __name__ == "__main__":
    video_files = os.listdir(INPUT_PATH)

    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH)
    
    for video_name in video_files:
        video_processor = VideoProcessor(video_name)
        video_processor.load_frames()
        video_processor.detect_landmarks()
        #video_processor.overlay_landmarks()
        video_processor.crop_mouth(CROPPED_SIZE, 28, 18)
        video_processor.save_video('cropped')
