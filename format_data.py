import os
import cv2
import xml.etree.ElementTree as ET
import random
from pathlib import Path
import shutil

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

def split_data_with_minimum(images_by_class, min_images_per_class=2, test_ratio=0.15):
    train_images = {}
    valid_images = {}
    test_images = {}

    for class_label, images in images_by_class.items():
        random.shuffle(images)
        
        # Ensure there are enough images to split, handle cases with fewer than the minimum
        if len(images) < 2 * min_images_per_class:
            print(f"Class '{class_label}' has only {len(images)} images, adjusting split.")
            # Split the images proportionally if there are not enough
            test_images[class_label] = images[:len(images)//2]
            valid_images[class_label] = images[len(images)//2:]
            train_images[class_label] = []
        else:
            # Allocate at least 2 images to valid and test
            test_images[class_label] = images[:min_images_per_class]
            valid_images[class_label] = images[min_images_per_class: min_images_per_class * 2]
            remaining_images = images[min_images_per_class * 2:]
        
            # Distribute remaining images based on the test_ratio
            split_index = int(len(remaining_images) * test_ratio)
            test_images[class_label].extend(remaining_images[:split_index])
            valid_images[class_label].extend(remaining_images[split_index:])
        
            # Remaining images go to train
            train_images[class_label] = remaining_images[split_index:]
    
    return train_images, valid_images, test_images


# Move images to corresponding folders
def move_images(images_by_class, split):
    for class_label, images in images_by_class.items():
        for image_file in images:
            xml_file = image_file.replace(".jpg", ".xml")
            src_image_path = os.path.join(train_folder, image_file)
            src_xml_path = os.path.join(train_folder, xml_file)
            dst_image_path = os.path.join(output_folder, split, class_label, image_file)
            dst_xml_path = os.path.join(output_folder, split, class_label, xml_file)
            shutil.move(src_image_path, dst_image_path)
            shutil.move(src_xml_path, dst_xml_path)

# Main processing logic
train_folder = os.path.join(voc_folder, 'train')
train_image_files = [f for f in os.listdir(train_folder) if f.endswith(".jpg")]

# Group images by class
images_by_class = {}
for image_file in train_image_files:
    xml_file = image_file.replace(".jpg", ".xml")
    xml_path = os.path.join(train_folder, xml_file)
    if os.path.exists(xml_path):
        bboxes = parse_voc_xml(xml_path)
        for bbox in bboxes:
            class_label = bbox['label']
            if class_label not in images_by_class:
                images_by_class[class_label] = []
            images_by_class[class_label].append(image_file)

# Create class directories in the output folders
create_output_dirs(images_by_class.keys())

# Split data ensuring minimum images per class
train_images, valid_images, test_images = split_data_with_minimum(images_by_class)

# Move images to corresponding splits
move_images(train_images, 'train')
move_images(valid_images, 'valid')
move_images(test_images, 'test')

print("Data split complete.")
