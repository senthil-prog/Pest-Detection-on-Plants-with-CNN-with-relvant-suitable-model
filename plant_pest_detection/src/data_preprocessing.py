import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
from config import Config
from tqdm import tqdm

class DataPreprocessor:
    def __init__(self):
        self.config = Config()
        
    def load_and_preprocess_data(self):
        """Load images and labels from the raw data directory"""
        print("Loading and preprocessing data...")
        
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.config.CLASS_NAMES):
            class_path = os.path.join(self.config.RAW_DATA_DIR, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Class directory {class_path} not found!")
                continue
                
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"Loading {len(image_files)} images from {class_name}...")
            
            for image_file in tqdm(image_files, desc=f"Processing {class_name}"):
                try:
                    image_path = os.path.join(class_path, image_file)
                    image = self.load_and_resize_image(image_path)
                    if image is not None:
                        images.append(image)
                        labels.append(class_idx)
                except Exception as e:
                    print(f"Error processing {image_file}: {str(e)}")
                    continue
        
        if len(images) == 0:
            raise ValueError("No images were loaded! Check your data directory structure.")
            
        images = np.array(images, dtype=np.float32) / 255.0
        labels = np.array(labels)
        
        print(f"Loaded {len(images)} images with shape {images.shape}")
        print(f"Class distribution: {np.bincount(labels)}")
        
        return images, labels
    
    def load_and_resize_image(self, image_path):
        """Load and resize a single image"""
        try:
            # Use OpenCV for better performance
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, (self.config.IMG_HEIGHT, self.config.IMG_WIDTH))
            
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def create_data_splits(self, images, labels):
        """Split data into train, validation, and test sets"""
        print("Creating data splits...")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=self.config.TEST_SPLIT, 
            stratify=labels, random_state=42
        )
        
        # Second split: separate train and validation
        val_size = self.config.VALIDATION_SPLIT / (1 - self.config.TEST_SPLIT)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, 
            stratify=y_temp, random_state=42
        )
        
        # Convert labels to categorical
        y_train = to_categorical(y_train, self.config.NUM_CLASSES)
        y_val = to_categorical(y_val, self.config.NUM_CLASSES)
        y_test = to_categorical(y_test, self.config.NUM_CLASSES)
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def create_data_generators(self):
        """Create data generators for training with augmentation"""
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=self.config.ROTATION_RANGE,
            width_shift_range=self.config.WIDTH_SHIFT_RANGE,
            height_shift_range=self.config.HEIGHT_SHIFT_RANGE,
            shear_range=self.config.SHEAR_RANGE,
            zoom_range=self.config.ZOOM_RANGE,
            brightness_range=self.config.BRIGHTNESS_RANGE,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            rescale=1.0/255.0
        )
        
        # Validation data generator (no augmentation)
        val_datagen = ImageDataGenerator(rescale=1.0/255.0)
        
        # Create generators from directory
        train_generator = train_datagen.flow_from_directory(
            self.config.RAW_DATA_DIR,
            target_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH),
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            subset='training'
        )
        
        return train_datagen, val_datagen
    
    def prepare_dataset(self):
        """Main method to prepare the complete dataset"""
        # Load and preprocess data
        images, labels = self.load_and_preprocess_data()
        
        # Create data splits
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.create_data_splits(images, labels)
        
        # Save processed data
        self.save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def save_processed_data(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Save processed data to disk"""
        print("Saving processed data...")
        
        os.makedirs(self.config.PROCESSED_DATA_DIR, exist_ok=True)
        
        np.save(os.path.join(self.config.PROCESSED_DATA_DIR, 'X_train.npy'), X_train)
        np.save(os.path.join(self.config.PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
        np.save(os.path.join(self.config.PROCESSED_DATA_DIR, 'X_val.npy'), X_val)
        np.save(os.path.join(self.config.PROCESSED_DATA_DIR, 'y_val.npy'), y_val)
        np.save(os.path.join(self.config.PROCESSED_DATA_DIR, 'X_test.npy'), X_test)
        np.save(os.path.join(self.config.PROCESSED_DATA_DIR, 'y_test.npy'), y_test)
        
        print("Processed data saved successfully!")
    
    def load_processed_data(self):
        """Load previously processed data"""
        print("Loading processed data...")
        
        X_train = np.load(os.path.join(self.config.PROCESSED_DATA_DIR, 'X_train.npy'))
        y_train = np.load(os.path.join(self.config.PROCESSED_DATA_DIR, 'y_train.npy'))
        X_val = np.load(os.path.join(self.config.PROCESSED_DATA_DIR, 'X_val.npy'))
        y_val = np.load(os.path.join(self.config.PROCESSED_DATA_DIR, 'y_val.npy'))
        X_test = np.load(os.path.join(self.config.PROCESSED_DATA_DIR, 'X_test.npy'))
        y_test = np.load(os.path.join(self.config.PROCESSED_DATA_DIR, 'y_test.npy'))
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)