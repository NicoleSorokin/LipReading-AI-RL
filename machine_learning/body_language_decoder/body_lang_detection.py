import streamlit as st
import mediapipe as mp
import cv2
import pandas as pd
import pickle
import numpy as np
import warnings
import plotly.graph_objects as go
from collections import deque
warnings.filterwarnings("ignore")

# Initialize session state variables
if 'threat_history' not in st.session_state:
    st.session_state.threat_history = deque(maxlen=100)
    st.session_state.q_table = {}
    st.session_state.action_history = deque(maxlen=100)
    st.session_state.reward_history = deque(maxlen=100)
    st.session_state.running = False

# Page config
st.set_page_config(page_title="Threat Detection System", layout="wide")

# Sidebar
st.sidebar.title("Settings")
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)
discount_factor = st.sidebar.slider("Discount Factor", 0.1, 0.99, 0.9)
epsilon = st.sidebar.slider("Exploration Rate", 0.0, 1.0, 0.2)

# Main title
st.title("Real-time Threat Detection System")

# Control buttons
col1, col2 = st.columns(2)
start_button = col1.button('Start', key='start_button')
stop_button = col2.button('Stop', key='stop_button')

if start_button:
    st.session_state.running = True
if stop_button:
    st.session_state.running = False

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Load pre-trained model
@st.cache_resource
def load_model():
    try:
        with open('body_lang_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file 'body_lang_model.pkl' not found. Please ensure the model file is in the correct location.")
        return None

model = load_model()

# Threat probabilities buffer
threat_probs_window = [[0.5, 0.5] for _ in range(40)]
actions = ["escalate", "de-escalate"]

def rolling_threat_average(threat_probs):
    global threat_probs_window
    threat_probs_window.insert(0, threat_probs)
    threat_probs_window = threat_probs_window[:40]
    return np.mean(threat_probs_window, axis=0)

def normalize_and_scale_landmarks(frame_landmarks, ref_lm_ind, left_lm_ind, right_lm_ind):
    if frame_landmarks is None or frame_landmarks.landmark is None:
        return [0] * (33 * 4)  # Return zeros for missing landmarks
    
    body_frame = frame_landmarks.landmark
    reference_landmark = body_frame[ref_lm_ind]
    ref_x, ref_y, ref_z = reference_landmark.x, reference_landmark.y, reference_landmark.z
    right_lm = body_frame[right_lm_ind]
    left_lm = body_frame[left_lm_ind]
    scale_factor = np.sqrt(
        (right_lm.x - left_lm.x) ** 2 +
        (right_lm.y - left_lm.y) ** 2 +
        (right_lm.z - left_lm.z) ** 2
    )
    if scale_factor == 0:
        scale_factor = 1e-6
    return list(
        np.array([
            [
                (landmark.x - ref_x) / scale_factor,
                (landmark.y - ref_y) / scale_factor,
                (landmark.z - ref_z) / scale_factor,
                landmark.visibility
            ]
            for landmark in body_frame
        ]).flatten()
    )

def get_state(threatness_level):
    if threatness_level < 0.4:
        return "low"
    elif 0.4 <= threatness_level <= 0.7:
        return "medium"
    else:
        return "high"

def choose_action(state):
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    if state in st.session_state.q_table:
        return max(st.session_state.q_table[state], key=st.session_state.q_table[state].get)
    return np.random.choice(actions)

def update_q_table(state, action, reward, next_state):
    if state not in st.session_state.q_table:
        st.session_state.q_table[state] = {a: 0 for a in actions}
    if next_state not in st.session_state.q_table:
        st.session_state.q_table[next_state] = {a: 0 for a in actions}
    
    st.session_state.q_table[state][action] += learning_rate * (
        reward + discount_factor * max(st.session_state.q_table[next_state].values()) - 
        st.session_state.q_table[state][action]
    )

# Create placeholder for video feed
video_placeholder = st.empty()

# Create columns for metrics
col1, col2, col3 = st.columns(3)
threat_metric = col1.empty()
action_metric = col2.empty()
q_values_metric = col3.empty()

# Create placeholder for graphs
graph_placeholder = st.empty()

def update_graphs():
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            y=list(st.session_state.threat_history),
            name="Threat Level",
            line=dict(color="red")
        )
    )
    
    fig.add_trace(
        go.Scatter(
            y=list(st.session_state.reward_history),
            name="Reward",
            line=dict(color="green")
        )
    )
    
    fig.update_layout(
        title="Threat Level and Reward History",
        xaxis_title="Time",
        yaxis_title="Value",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    graph_placeholder.plotly_chart(fig, use_container_width=True)

if model is not None and st.session_state.running:
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open webcam. Please check your camera connection.")
        else:
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while cap.isOpened() and st.session_state.running:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Error: Could not read frame from webcam.")
                        break
                        
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = holistic.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    try:
                        # Process landmarks with error handling
                        normalized_pose_row = normalize_and_scale_landmarks(results.pose_landmarks, 0, 11, 12)
                        normalized_face_row = normalize_and_scale_landmarks(results.face_landmarks, 0, 33, 263)
                        normalized_right_hand_row = normalize_and_scale_landmarks(results.right_hand_landmarks, 0, 4, 20) if results.right_hand_landmarks else [0] * (21 * 4)
                        normalized_left_hand_row = normalize_and_scale_landmarks(results.left_hand_landmarks, 0, 4, 20) if results.left_hand_landmarks else [0] * (21 * 4)
                        row = normalized_pose_row + normalized_face_row + normalized_right_hand_row + normalized_left_hand_row

                        # Make prediction
                        X = pd.DataFrame([row])
                        body_language_prob = model.predict_proba(X)[0]
                        threatness_level = rolling_threat_average(body_language_prob)[1]
                        
                        # Update state and choose action
                        state = get_state(threatness_level)
                        action = choose_action(state)

                        # Calculate reward
                        if action == "escalate":
                            reward = -1 if threatness_level < 0.5 else 1
                        else:
                            reward = 1 if threatness_level < 0.5 else -1

                        # Update histories
                        st.session_state.threat_history.append(threatness_level)
                        st.session_state.action_history.append(action)
                        st.session_state.reward_history.append(reward)

                        # Update Q-table
                        next_state = get_state(rolling_threat_average(body_language_prob)[1])
                        update_q_table(state, action, reward, next_state)

                        # Update metrics
                        threat_metric.metric("Threat Level", f"{threatness_level:.2f}")
                        action_metric.metric("Current Action", action)
                        q_values_metric.metric("Q-Values", str(st.session_state.q_table.get(state, {})))

                        # Update graphs
                        update_graphs()

                        # Add visual feedback to frame
                        if threatness_level > 0.5:
                            status_message = "Warning: Elevated Threat Level"
                            color = (0, 0, 255)
                        else:
                            status_message = "Status: Normal"
                            color = (0, 255, 0)

                        cv2.putText(image, status_message, (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

                        # Display the frame
                        video_placeholder.image(image, channels="BGR", use_container_width=True)

                    except Exception as e:
                        st.error(f"Error processing frame: {str(e)}")

            cap.release()
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Display Q-table
st.header("Q-Learning Table")
st.dataframe(pd.DataFrame.from_dict(st.session_state.q_table, orient='index'))