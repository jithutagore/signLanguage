from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect
actions = np.array([folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))])

# Create a label map
label_map = {label: num for num, label in enumerate(actions)}

# Initialize lists to hold sequences and labels
sequences, labels = [], []

# Loop through each action
for action in actions:
    action_folder = os.path.join(DATA_PATH, action)
    sequence_files = [f for f in os.listdir(action_folder) if f.endswith('.npy')]  # Only get .npy files

    for sequence_file in sequence_files:
        sequence_path = os.path.join(action_folder, sequence_file)

        # Load the keypoints sequence from the .npy file
        keypoints = np.load(sequence_path)
        sequences.append(keypoints)

        # Append the corresponding label for this action
        labels.append(label_map[action])

# Convert to numpy arrays
X = np.array(sequences)
# Reshape X for LSTM
n_timesteps = 277  # Specify your desired number of timesteps
print(X.shape[1])
n_features = X.shape[1] // n_timesteps  # Adjust based on your specific data

# Ensure total number of features is divisible by timesteps
if X.shape[1] % n_timesteps != 0:
    print(X.shape[1] % n_timesteps)
    raise ValueError("Number of features must be divisible by the number of timesteps.")

# Reshape to 3D
X = X.reshape(X.shape[0], n_timesteps, n_features)
print("Shape of X after reshaping:", X.shape)

# Convert labels to categorical
y = to_categorical(labels).astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Initialize TensorBoard
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=1, callbacks=[tb_callback])
model.summary()
model.save('action.h5')