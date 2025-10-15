#!/usr/bin/env python3
"""
Plant Pest Detection CNN Model
Main execution script for training and evaluation
"""

import os
import sys
import argparse
import tensorflow as tf

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from train import ModelTrainer
from predict import PestPredictor
from data_preprocessing import DataPreprocessor

def setup_environment():
    """Setup the environment and check requirements"""
    print("Setting up environment...")
    
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
    else:
        print("No GPU found, using CPU")
    
    # Create necessary directories
    config = Config()
    config.create_directories()
    print("Directory structure created")

def check_data_structure():
    """Check if data directory structure exists"""
    config = Config()
    
    print("Checking data structure...")
    data_exists = os.path.exists(config.RAW_DATA_DIR)
    
    if not data_exists:
        print(f"❌ Raw data directory not found: {config.RAW_DATA_DIR}")
        print("\nPlease create the following directory structure:")
        print(f"{config.RAW_DATA_DIR}/")
        for class_name in config.CLASS_NAMES:
            print(f"  ├── {class_name}/")
            print(f"  │   ├── image1.jpg")
            print(f"  │   ├── image2.jpg")
            print(f"  │   └── ...")
        return False
    
    # Check class directories
    missing_classes = []
    total_images = 0
    
    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            missing_classes.append(class_name)
        else:
            # Count images in directory
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_images += len(image_files)
            print(f"✅ {class_name}: {len(image_files)} images")
    
    if missing_classes:
        print(f"❌ Missing class directories: {missing_classes}")
        return False
    
    if total_images == 0:
        print("❌ No images found in class directories")
        return False
    
    print(f"✅ Data structure OK - Total images: {total_images}")
    return True

def train_mode():
    """Training mode"""
    print("\n" + "="*60)
    print("TRAINING MODE")
    print("="*60)
    
    # Check data structure
    if not check_data_structure():
        print("Please fix data structure before training")
        return
    
    # Start training
    trainer = ModelTrainer()
    model, history = trainer.train_model(use_processed_data=True)
    
    print("\n✅ Training completed successfully!")
    print(f"Model saved in: {Config.SAVED_MODELS_DIR}")

def evaluate_mode():
    """Evaluation mode"""
    print("\n" + "="*60)
    print("EVALUATION MODE")
    print("="*60)
    
    config = Config()
    model_path = os.path.join(config.SAVED_MODELS_DIR, 'pest_detection_model.h5')
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first using: python main.py --mode train")
        return
    
    # Load test data
    data_processor = DataPreprocessor()
    if not data_processor.check_processed_data_exists():
        print("❌ Processed data not found. Please run training first.")
        return
    
    print("Loading test data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_processor.load_processed_data()
    
    # Load model and evaluate
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    print("Evaluating model on test set...")
    trainer = ModelTrainer()
    trainer.evaluate_model(model, X_test, y_test)
    
    print("\n✅ Evaluation completed!")

def predict_mode(image_path=None, directory=None, visualize=False):
    """Prediction mode"""
    print("\n" + "="*60)
    print("PREDICTION MODE")
    print("="*60)
    
    config = Config()
    model_path = os.path.join(config.SAVED_MODELS_DIR, 'pest_detection_model.h5')
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first using: python main.py --mode train")
        return
    
    # Create predictor
    predictor = PestPredictor(model_path)
    
    if image_path:
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        print(f"Making prediction on: {image_path}")
        if visualize:
            predictor.visualize_prediction_confidence(image_path)
        else:
            predictor.predict_single_image(image_path)
    
    elif directory:
        if not os.path.exists(directory):
            print(f"❌ Directory not found: {directory}")
            return
        
        print(f"Making predictions on images in: {directory}")
        output_file = os.path.join("predictions_results.csv")
        results = predictor.predict_from_directory(directory, output_file)
        
        if results:
            print(f"\n✅ Batch prediction completed!")
            print(f"Results saved to: {output_file}")
    
    else:
        print("❌ Please provide either --image or --directory for prediction")

def data_info_mode():
    """Display data information"""
    print("\n" + "="*60)
    print("DATA INFORMATION")
    print("="*60)
    
    config = Config()
    
    # Check raw data
    if check_data_structure():
        print("\n📊 Dataset Statistics:")
        total_images = 0
        class_distribution = {}
        
        for class_name in config.CLASS_NAMES:
            class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
            if os.path.exists(class_dir):
                image_files = [f for f in os.listdir(class_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                count = len(image_files)
                class_distribution[class_name] = count
                total_images += count
        
        print(f"Total images: {total_images}")
        print("Class distribution:")
        for class_name, count in class_distribution.items():
            percentage = (count / total_images) * 100 if total_images > 0 else 0
            print(f"  {class_name}: {count} images ({percentage:.1f}%)")
        
        # Check for class imbalance
        if class_distribution:
            max_count = max(class_distribution.values())
            min_count = min(class_distribution.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 3:
                print(f"\n⚠️  Class imbalance detected (ratio: {imbalance_ratio:.1f})")
                print("Consider data augmentation or class weighting")
            else:
                print(f"\n✅ Dataset is reasonably balanced (ratio: {imbalance_ratio:.1f})")
    
    # Check processed data
    data_processor = DataPreprocessor()
    if data_processor.check_processed_data_exists():
        print("\n📁 Processed Data Found:")
        processed_files = ['X_train.npy', 'y_train.npy', 'X_val.npy', 
                          'y_val.npy', 'X_test.npy', 'y_test.npy']
        for file_name in processed_files:
            file_path = os.path.join(config.PROCESSED_DATA_DIR, file_name)
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  ✅ {file_name} ({size_mb:.1f} MB)")
    else:
        print("\n📁 No processed data found")
        print("Run training to create processed data files")

def model_info_mode():
    """Display model information"""
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    
    config = Config()
    model_path = os.path.join(config.SAVED_MODELS_DIR, 'pest_detection_model.h5')
    
    if os.path.exists(model_path):
        print("📋 Model Details:")
        print(f"  Model file: {model_path}")
        
        # Get file size
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")
        
        # Load and display model info
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"  Input shape: {model.input_shape}")
            print(f"  Output shape: {model.output_shape}")
            print(f"  Total parameters: {model.count_params():,}")
            
            # Count trainable parameters
            trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            print(f"  Trainable parameters: {trainable_params:,}")
            
            print(f"  Classes: {config.CLASS_NAMES}")
            
        except Exception as e:
            print(f"  ❌ Error loading model: {str(e)}")
    else:
        print("❌ No trained model found")
        print("Run training first: python main.py --mode train")

def cleanup_mode():
    """Cleanup generated files"""
    print("\n" + "="*60)
    print("CLEANUP MODE")
    print("="*60)
    
    config = Config()
    
    # Ask for confirmation
    response = input("This will delete all generated files (models, processed data, logs). Continue? (y/N): ")
    if response.lower() != 'y':
        print("Cleanup cancelled")
        return
    
    import shutil
    
    directories_to_clean = [
        config.PROCESSED_DATA_DIR,
        config.SAVED_MODELS_DIR,
        config.CHECKPOINTS_DIR,
        config.LOG_DIR
    ]
    
    cleaned_count = 0
    for directory in directories_to_clean:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                print(f"✅ Cleaned: {directory}")
                cleaned_count += 1
            except Exception as e:
                print(f"❌ Error cleaning {directory}: {str(e)}")
        else:
            print(f"⚪ Not found: {directory}")
    
    # Recreate directories
    config.create_directories()
    
    print(f"\n✅ Cleanup completed! Removed {cleaned_count} directories")

def main():
    parser = argparse.ArgumentParser(
        description='Plant Pest Detection CNN Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train                    # Train the model
  python main.py --mode evaluate                 # Evaluate trained model
  python main.py --mode predict --image test.jpg # Predict single image
  python main.py --mode predict --directory ./test_images # Batch prediction
  python main.py --mode data_info               # Show dataset information
  python main.py --mode model_info              # Show model information
  python main.py --mode cleanup                 # Clean generated files
        """
    )
    
    parser.add_argument('--mode', 
                       choices=['train', 'evaluate', 'predict', 'data_info', 'model_info', 'cleanup'],
                       required=True,
                       help='Operation mode')
    
    parser.add_argument('--image', type=str, 
                       help='Path to image for prediction')
    
    parser.add_argument('--directory', type=str,
                       help='Directory containing images for batch prediction')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Show prediction visualization')
    
    parser.add_argument('--model_path', type=str,
                       help='Custom path to model file')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Execute based on mode
    if args.mode == 'train':
        train_mode()
        
    elif args.mode == 'evaluate':
        evaluate_mode()
        
    elif args.mode == 'predict':
        predict_mode(image_path=args.image, 
                    directory=args.directory,
                    visualize=args.visualize)
        
    elif args.mode == 'data_info':
        data_info_mode()
        
    elif args.mode == 'model_info':
        model_info_mode()
        
    elif args.mode == 'cleanup':
        cleanup_mode()
    
    print("\n🎉 Operation completed!")

if __name__ == "__main__":
    main()