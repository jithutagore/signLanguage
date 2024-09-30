import numpy as np
from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp
import os

# Load the model
model = load_model('action.h5')

# Initialize Mediapipe for pose, face, and hand detection
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Load the label map (the same as used in training)
DATA_PATH = os.path.join('MP_Data')
actions = np.array([folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))])
label_map = {label: num for num, label in enumerate(actions)}

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Function to preprocess the input image
def preprocess_image(image):
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe to get landmarks
    with mp_pose.Pose(static_image_mode=True) as pose, \
         mp_face.FaceMesh(static_image_mode=True) as face, \
         mp_hands.Hands(static_image_mode=True) as hands:

        results_pose = pose.process(image_rgb)
        results_face = face.process(image_rgb)
        results_hands = hands.process(image_rgb)

        # Combine results
        results = type('', (), {})()  # Create a simple empty object
        results.pose_landmarks = results_pose.pose_landmarks
        results.face_landmarks = results_face.multi_face_landmarks[0] if results_face.multi_face_landmarks else None
        results.left_hand_landmarks = results_hands.multi_hand_landmarks[0] if results_hands.multi_hand_landmarks else None
        results.right_hand_landmarks = results_hands.multi_hand_landmarks[1] if results_hands.multi_hand_landmarks and len(results_hands.multi_hand_landmarks) > 1 else None

    keypoints = extract_keypoints(results)
    return keypoints

def predict_action(image):
    n_timesteps = 277  # Same as in your train.py
    n_features = 6     # Update this according to your specific model's expected input

    keypoints = preprocess_image(image)
    
    # Check the shape of keypoints
    print("Keypoints shape before reshaping:", keypoints.shape)

    # Ensure that the number of features matches the expected shape
    if keypoints.shape[0] != n_features * n_timesteps:
        raise ValueError(f"Expected {n_features * n_timesteps} features but got {keypoints.shape[0]}.")

    # Reshape for LSTM (1, timesteps, features)
    keypoints = keypoints.reshape(1, n_timesteps, n_features)  # Adjusted reshaping

    prediction = model.predict(keypoints)
    action_index = np.argmax(prediction)
    return action_index, actions[action_index]  # Return both the predicted action index and label

# Test the model with images from a specified directory
def test_model_on_directory(image_path):

    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at path: {image_path}")
        predicted_action_index, predicted_action_label = predict_action(image)
        print(f'Predicted Action Index: {predicted_action_index}, Action Label: {predicted_action_label}')
    except Exception as e:
        print(f"Error processing: {e}")

# Specify the directory containing test images
if __name__ == "__main__":
    test_image = r'data/train/A/A0_jpg.rf.0caf9445dbc2a944bb713661e9189e26_0.jpg'  # Update this to your test image directory
    test_model_on_directory(test_image)
