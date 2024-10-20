
# Design Document

## Idea:  
**Lip Reading AI using Reinforcement Learning**  

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

## The 2 Main Stages:  

### First:  
- Employ conventional computer vision techniques alongside neural networks to identify and classify a person's physical actions.  
- Utilize distinct models to analyze lip movements and classify spoken words, leveraging existing frameworks like Mediapipe or Dlib.  

### Second:  
- Train a reinforcement learning model to recognize sequences of actions and lip movements that indicate malicious behavior.  
- Example: identify when someone makes a threat or exhibits violent movements, categorizing them as malicious.  

## Decisions + Documentation:
### Add your decisions and reasoning here:
- Using [EMOLIPS model]([url](https://github.com/SMIL-SPCRAS/EMOLIPS)) (CNN-LSTM model) to detect emotion from lips using details from the face.
- Negative emotions (e.g. anger, disgust) can be used to assist in threat identification. 

## Rough Milestone Timelines:  

### Week 1:  
- Project kickoff and setup  
- Assign tasks  
- Define goals  
- Start preparing and exploring data, and implement processes to load it in  
- Create basic frontend and backend  
- Set up OpenCV  

### Weeks 2-3:  
- Start implementing CNN  
- Set up lip detection  
- Get the API endpoints working between frontend and backend  

### Weeks 4-5:  
- Data preprocessing should be done by now to be ready for training  
- Continue working on CNN and RNN  
- Start implementing RNN  
- Work on improving lip detection  

### Weeks 6-7:  
- Set up Gym environment and begin integrating TF-Agents for RL  
- Refine CNN, test on datasets  
- Start training using RL  
- Integrate frontend to upload video and send to backend to start testing on it  

### Weeks 8-9:  
- Finish up the CNN and RNN models and integrate with RL  
- Integrate backend logic for handling RL inference and make it work with frontend UI  
- Test the RL on video uploads, adjust parameters  
- Implement end-to-end pipeline: video upload → processing → lip reading → display results  
- Address any performance issues/bugs  

### Week 10:  
- MVP should be done by now  
- Video uploads  
- Keep fine-tuning the RL training (and try to finish integrating with CNN and RNN)  
- Lip reading detection  
- Refine frontend for smooth interaction  

## AFTER WINTER BREAK:  

### Weeks 11-12:  
- Keep training and testing  
- Train the RL agents more, fine-tune the reward functions and policies  

### Weeks 13-14:  
- Conduct tests and identify edge cases  
- Optimize where possible  
- Continue testing and optimizing  

### Week 15:  
- Final touches  

## Process FLow Diagram:
![Process Flow Diagram](/process_flow_dgm.jpg)
