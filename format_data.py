import os
import cv2
import xml.etree.ElementTree as ET
import random
from pathlib import Path

# Paths
voc_folder = "Sign-Language-1"  # Path to the root folder containing train, valid folders
output_folder = "data"  # Output folder for train, valid, test dataset

# Create output folder if it doesn't exist
Path(output_folder).mkdir(parents=True, exist_ok=True)

# Create output directories for classification
def create_output_dirs(classes):
    for split in ['train', 'valid', 'test']:  # Include 'test' now
        for class_label in classes:
            Path(f"{output_folder}/{split}/{class_label}").mkdir(parents=True, exist_ok=True)

# Function to parse XML and extract bounding box info
def parse_voc_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bboxes = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bboxes.append({'label': label, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

    return bboxes

# Function to process image and save cropped objects
def process_image(image_path, xml_path, split):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return
    
    bboxes = parse_voc_xml(xml_path)
    
    for i, bbox in enumerate(bboxes):
        label = bbox['label']
        xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
        
        # Check bounds before cropping
        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        if xmax > image.shape[1]: xmax = image.shape[1]
        if ymax > image.shape[0]: ymax = image.shape[0]
        
        # Crop the image based on bounding box
        cropped_img = image[ymin:ymax, xmin:xmax]
        
        # Save the cropped image in the corresponding class folder
        output_filename = f"{Path(image_path).stem}_{i}.jpg"
        output_path = f"{output_folder}/{split}/{label}/{output_filename}"
        cv2.imwrite(output_path, cropped_img)

# Function to split data into train, valid, and test
def split_data(image_files, split_ratio):
    random.shuffle(image_files)
    split_index = int(len(image_files) * split_ratio)
    return image_files[:split_index], image_files[split_index:]

# List all folders in the root voc_folder
for split in ['train', 'valid']:  # Only include existing splits
    split_folder = os.path.join(voc_folder, split)
    image_files = [f for f in os.listdir(split_folder) if f.endswith(".jpg")]
    
    # Ensure XML files exist
    xml_files = [f.replace(".jpg", ".xml") for f in image_files]
    
    # Extract classes from the XML files (unique labels)
    classes = set()
    for xml_file in xml_files:
        xml_path = os.path.join(split_folder, xml_file)
        if os.path.exists(xml_path):  # Check if XML file exists
            bboxes = parse_voc_xml(xml_path)
            for bbox in bboxes:
                classes.add(bbox['label'])
    
    # Create folders for each class
    create_output_dirs(classes)
    
    # Process each image in the current split
    for image_file in image_files:
        xml_file = image_file.replace(".jpg", ".xml")
        process_image(os.path.join(split_folder, image_file), os.path.join(split_folder, xml_file), split)

# Create the test split from the train data
train_folder = os.path.join(voc_folder, 'train')
train_image_files = [f for f in os.listdir(train_folder) if f.endswith(".jpg")]

# Split the train images to create a test set
train_images, test_images = split_data(train_image_files, 0.15)  # Pass 0.15 directly


# Move selected images to the test directory
for test_image in test_images:
    test_xml = test_image.replace(".jpg", ".xml")
    os.rename(os.path.join(train_folder, test_image), os.path.join(output_folder, 'test', test_image))
    os.rename(os.path.join(train_folder, test_xml), os.path.join(output_folder, 'test', test_xml))
