# Save this as `q_learning_agent.py`

import random
import numpy as np
from rl import MaliciousClassificationEnv

def train_and_classify(shared_emotion_val):
    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.95
    EXPLORATION_RATE = 1.0
    EXPLORATION_DECAY = 0.995
    MIN_EXPLORATION_RATE = 0.01
    EPISODES = 1000

    q_table = np.zeros((2, 2, 2))
    env = MaliciousClassificationEnv()

    for episode in range(EPISODES):
        state = env.reset()
        done = False

        while not done:
            if random.uniform(0, 1) < EXPLORATION_RATE:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state[0], state[1]])

            next_state, reward, done, _ = env.step(action)

            old_q_value = q_table[state[0], state[1], action]
            next_max_q = np.max(q_table[next_state[0], next_state[1]])
            new_q_value = old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max_q - old_q_value)
            q_table[state[0], state[1], action] = new_q_value

            state = next_state

        EXPLORATION_RATE = max(MIN_EXPLORATION_RATE, EXPLORATION_RATE * EXPLORATION_DECAY)

    print("Trained Q-table:")
    print(q_table)

    # while True:
    #     emotion_val = shared_emotion_val.value
    #     print(emotion_val)
    #     malicious_score = 0  # Example placeholder
    #     emotion_idx = env.emotion_values.index(emotion_val)
    #     malicious_idx = env.maliciousness_values.index(malicious_score)
    #     action = np.argmax(q_table[emotion_idx, malicious_idx])
    #     print(f"Classified as {'Malicious' if action == 1 else 'Non-Malicious'} with emotion_val={emotion_val}")
