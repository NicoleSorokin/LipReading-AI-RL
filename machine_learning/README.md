run the app using the following command
```
streamlit run app.py
```

Currently the app.py contains the body language code *training and details about which can be found in the body_lang_decoder folder*

Additionally, the app.py contains reinforcement learning code, details below:

State Space:
```
def get_state(threatness_level):
    if threatness_level < 0.4:
        return "low"
    elif 0.4 <= threatness_level <= 0.7:
        return "medium"
    else:
        return "high"
```
State space is simplified into three levels (low, medium, high) based on the threat probability from the body language model. This simplifcation allows the learningto be more manageable while still capturing the essential threat levels.

Action Space:
```
actions = ["escalate", "de-escalate"]
```
Action space is simplified into two actions (escalate and de-escalate) based on the current state. This simplifies the learning process as well as the decision making process.

Q-Learning Table:
```
def update_q_table(state, action, reward, next_state):
    if state not in st.session_state.q_table:
        st.session_state.q_table[state] = {a: 0 for a in actions}
    if next_state not in st.session_state.q_table:
        st.session_state.q_table[next_state] = {a: 0 for a in actions}
    
    st.session_state.q_table[state][action] += learning_rate * (
        reward + discount_factor * max(st.session_state.q_table[next_state].values()) - 
        st.session_state.q_table[state][action]
    )
```
The Q-Learning table is a dictionary that stores the Q-values for each state-action pair. The Q-values are updated based on the current state, action, reward, and next state. The Q-values are used to determine the best action to take in the next state.

Action Selection:
```
def choose_action(state):
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    if state in st.session_state.q_table:
        return max(st.session_state.q_table[state], key=st.session_state.q_table[state].get)
    return np.random.choice(actions)
```
The action selection process is based on the current state and the Q-Learning table. The action selection process is random if the exploration rate is high, and based on the Q-values if the exploration rate is low. The Q-values are updated based on the current state, action, reward, and next state.

Reward Calculation:
```
if action == "escalate":
    reward = -1 if threatness_level < 0.5 else 1
else:
    reward = 1 if threatness_level < 0.5 else -1
```
The reward calculation is based on the current action and the threat probability. (Add more info on how the reward is calculated.)

Benefits of using reinforcement learning:
- Learning from trial and error it improves the accuracy of the model.
- It allows for adaptation to new situations.
- The reward system provides immediate feedback about the appropriateness of actions.
- Continuously improve its decision-making based on experience.
- and many more to be added soon...
