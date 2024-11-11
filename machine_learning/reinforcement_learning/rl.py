import gym
from gym import spaces
import numpy as np

class MaliciousClassificationEnv(gym.Env):
    """
    Custom Environment for classifying behavior as malicious or non-malicious.
    """
    def __init__(self):
        super(MaliciousClassificationEnv, self).__init__()
        self.action_space = spaces.Discrete(2) # define the two actions spaces, 0-> non-malicious, 1-> malicious
        
        self.observation_space = spaces.MultiDiscrete([2, 2])  # 2 values for each state variable in the observation space
        
        self.emotion_values = [-1, 1]  # -1 for negative, 1 for positive
        self.maliciousness_values = [0, 1]  # 0 for non-malicious, 1 for malicious
        self.current_state = None

    def step(self, action):
        """
        Take a step in the environment based on the action.
        """
        emotion, maliciousness = self.current_state

        # expected action 
        if maliciousness == 1 :
            correct_action = 1
        else :
            correct_action = 0

        # reward computation
        if action == correct_action:
            reward = 10  # positive reward for correct classification and then end it
            done = True  
        else:
            reward = -10  # negative reward for incorrect classification, and also end it
            done = True  

        return self._get_observation(), reward, done, {}
    
    def render(self, mode="human"):
        """
        Render the environment (optional).
        """
        print(f"Current State: {self.current_state}")

    def reset(self):
        """
        Reset the environment to an initial state and return the initial observation.
        """
        emotion = np.random.choice(self.emotion_values)
        maliciousness = np.random.choice(self.maliciousness_values)
        self.current_state = (emotion, maliciousness)
        return self._get_observation()

    def _get_observation(self):
        """
        Return the current state as observation indices.
        """
        emotion_idx = self.emotion_values.index(self.current_state[0])
        malicious_idx = self.maliciousness_values.index(self.current_state[1])
        return np.array([emotion_idx, malicious_idx])