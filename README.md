# LipReading-AI-RL

This project focuses on building an AI system capable of lip-reading to detect potential malicious or threatening behavior when audio is unavailable or corrupted based on lip movements and key words. The system uses supervised learning for lip reading (mapping speech-to-text) and reinforcement learning (RL) to detect behaviors based on decoded speech.

## Use Cases
- Security applications: Detecting threats or violent language when audio is corrupted or unavailable.
- Violence mitigation: This project can potentially be used for public safety, such as in campus surveillance.

## Tech stack: 
- Programming Language: Python 
- Datasets: numpy (data manipulations), GRID corpus, multi-view lip reading sentences or lip reading in the wild (datasets for training) 
- Computer Vision: OpenCV (capturing video), Dlib (detecting lips), FFmpeg (for processing video frames), maybe media pipe also for detecting face 
- Machine Learning Framework: Tensorflow (or PyTorch) 
- Reinforcement Learning: Gymnasium (environment), TF-Agents (reinforcement learning algorithms) (StableBaseline 3 if using PyTorch) 
- Frontend: React next.js 
- Backend: fastapi 
- Algorithms: CNN’s, RNN’s 

## Process FLow Diagram:
![Process Flow Diagram](/process_flow_dgm.jpg)