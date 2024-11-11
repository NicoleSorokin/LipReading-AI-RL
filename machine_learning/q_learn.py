import random
import numpy as np

from rl import MaliciousClassificationEnv

# define params
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EXPLORATION_RATE = 1.0
EXPLORATION_DECAY = 0.995
MIN_EXPLORATION_RATE = 0.01
EPISODES = 1000

# init Q-table for all possible state-action pairs
q_table = np.zeros((2, 2, 2))  # shape -> emotion states, maliciousness states, actions

env = MaliciousClassificationEnv() # init environment

for episode in range(EPISODES):
    # reset env and get initial state
    state = env.reset()
    done = False

    while not done:
        # select action based on Q-table
        if random.uniform(0, 1) < EXPLORATION_RATE:
            action = env.action_space.sample()  # select random action
        else:
            action = np.argmax(q_table[state[0], state[1]])  # select action with highest Q-value

        # take action, observe reward and next state
        next_state, reward, done, _ = env.step(action)

        # updare q values
        old_q_value = q_table[state[0], state[1], action]
        next_max_q = np.max(q_table[next_state[0], next_state[1]])
        new_q_value = old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max_q - old_q_value)
        q_table[state[0], state[1], action] = new_q_value

        state = next_state #update curr state

    EXPLORATION_RATE = max(MIN_EXPLORATION_RATE, EXPLORATION_RATE * EXPLORATION_DECAY)

print("Trained Q-table:")
print(q_table)

# TEST
def classify_behavior(emotion, malicious_score):
    """
    Use the trained Q-table to classify behavior as Malicious or Non-Malicious.
    """
    emotion_idx = env.emotion_values.index(emotion)
    malicious_idx = env.maliciousness_values.index(malicious_score)
    action = np.argmax(q_table[emotion_idx, malicious_idx])
    return "Malicious" if action == 1 else "Non-Malicious"

# Test the agent with an example input
print("Classify behavior:", classify_behavior(-1, 1))  # Negative emotion, malicious behavior