import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def predict_action(image_path, model, actions):
    # Load image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    # Detect keypoints using MediaPipe
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        image, results = mediapipe_detection(image, holistic)
        keypoints = extract_keypoints(results)

    # Prepare the keypoints for prediction
    keypoints = keypoints.reshape(1, n_timesteps, n_features)  # Reshape for LSTM

    # Make predictions
    prediction = model.predict(keypoints)
    action_index = np.argmax(prediction)
    action_label = actions[action_index]

    return action_label

# Load the trained model
model = load_model('action.h5')

# Define the number of timesteps and features based on your training data
n_timesteps = 277 
n_features = 1662 // n_timesteps

# Example usage
image_path = r'data/train/A/A0_jpg.rf.0caf9445dbc2a944bb713661e9189e26_0.jpg' 
actions = np.array([folder for folder in os.listdir('MP_Data') if os.path.isdir(os.path.join('MP_Data', folder))])  # Load actions

predicted_action = predict_action(image_path, model, actions)
if predicted_action:
    print(f'Predicted Action: {predicted_action}')




