# Save this as `main.py`

import cv2

from multiprocessing import Process, Manager
from emotion_detection import detect_emotion
from q_learning_agent import train_and_classify

if __name__ == "__main__":
    manager = Manager()
    shared_emotion_val = manager.Value('d', -1)  # Shared variable for emotion value

    emotion_process = Process(target=detect_emotion, args=(shared_emotion_val,))
    agent_process = Process(target=train_and_classify, args=(shared_emotion_val,))

    emotion_process.start()
    agent_process.start()

    while True:
        try:
            emotion_process.join()
            agent_process.join()
        except KeyboardInterrupt:
            print("Terminating processes...")
            cv2.destroyAllWindows()
            emotion_process.terminate()
            agent_process.terminate()
            break
