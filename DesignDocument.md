
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
#### Lip Movement to text:
- We will be closely following the methods of [LipNet](https://arxiv.org/pdf/1611.01599) as it has been proven to work and there is lots of existing documentations on this method
- This method uses Dlib for detecting facial landmarks and preprocessing the GRID dataset, then inputs a sequence of frames to 3 layers of 
of a CNN, each followed by a spatial max-pooling layer, then features are processed by two bidrectional GRUs; each time-step of the GRU output is processed by a linear layer and a softmax over the vocabulary. The model is then trained using CTC.
##### Preprocessing:
- Loaded video frames and detected faces and facial landmarks using dlib pretrained models
- Implemented mouth cropping based on mouth landmarks with reference point stabilization
- Uses a weighted average of the current and last frame
- Example (may need to open in a media player, not directly in code editor):
> [Uncropped Video](machine_learning/lip_reading/preprocessing/dataset/example_speaker.mp4)  
> [Cropped Video](machine_learning/lip_reading/preprocessing/output/cropped_example_speaker.mp4)

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
