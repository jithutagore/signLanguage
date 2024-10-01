import os
import random
import shutil
from pathlib import Path

# Define paths
train_folder = "data/train"
valid_folder = "data/valid"
test_folder = "data/test"

# Create valid and test directories if they don't exist
Path(valid_folder).mkdir(parents=True, exist_ok=True)
Path(test_folder).mkdir(parents=True, exist_ok=True)

# Move 2 images to both valid and test from each class in train
def move_images(train_folder, valid_folder, test_folder):
    # Iterate over each class in the train folder
    for class_folder in os.listdir(train_folder):
        class_train_path = os.path.join(train_folder, class_folder)
        
        # Ensure it's a directory
        if not os.path.isdir(class_train_path):
            continue
        
        # List all image files in the class folder
        image_files = [f for f in os.listdir(class_train_path) if f.endswith('.jpg') or f.endswith('.png')]
        
        # Check if there are at least 4 images in the class folder (2 for valid, 2 for test)
        if len(image_files) < 4:
            print(f"Not enough images in class '{class_folder}' to move.")
            continue
        
        # Randomly select 4 images
        selected_images = random.sample(image_files, 4)
        
        # Create corresponding class directories in valid and test folders
        valid_class_path = os.path.join(valid_folder, class_folder)
        test_class_path = os.path.join(test_folder, class_folder)
        
        Path(valid_class_path).mkdir(parents=True, exist_ok=True)
        Path(test_class_path).mkdir(parents=True, exist_ok=True)
        
        # Move two images to valid and two images to test
        shutil.move(os.path.join(class_train_path, selected_images[0]), os.path.join(valid_class_path, selected_images[0]))
        shutil.move(os.path.join(class_train_path, selected_images[1]), os.path.join(valid_class_path, selected_images[1]))
        shutil.move(os.path.join(class_train_path, selected_images[2]), os.path.join(test_class_path, selected_images[2]))
        shutil.move(os.path.join(class_train_path, selected_images[3]), os.path.join(test_class_path, selected_images[3]))
        
        print(f"Moved {selected_images[0]} and {selected_images[1]} to {valid_class_path}")
        print(f"Moved {selected_images[2]} and {selected_images[3]} to {test_class_path}")

# Run the function to move images
move_images(train_folder, valid_folder, test_folder)
