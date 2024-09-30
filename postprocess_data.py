import cv2
import numpy as np
import os
import mediapipe as mp
from pathlib import Path

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

# Path for exported data, numpy arrays
DATA_PATH = "data/train"  # Path to the training data
OUTPUT_PATH = "MP_Data"    # Where to save the npy files

# Dynamically load actions from folder names
actions = np.array([folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))])
print(actions)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Initialize MediaPipe Holistic
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        action_folder = os.path.join(DATA_PATH, action)
        if os.path.exists(action_folder):
            image_files = [f for f in os.listdir(action_folder) if f.endswith(".jpg") or f.endswith(".png")]
            
            for idx, image_file in enumerate(image_files):
                image_path = os.path.join(action_folder, image_file)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Error loading image: {image_path}")
                    continue

                # Detect keypoints using MediaPipe
                image, results = mediapipe_detection(image, holistic)

                # Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(OUTPUT_PATH, action, f"{action}_{idx}.npy")
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)  # Create the action directory if it doesn't exist
                np.save(npy_path, keypoints)

                print(f"Saved keypoints for {image_file} to {npy_path}")

print("Keypoints extraction and saving completed.")
