import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import Config
from data_preprocessing import DataPreprocessor
from model_architecture import PestDetectionModel

class ModelTrainer:
    def __init__(self):
        self.config = Config()
        self.config.create_directories()
        
        # Enable mixed precision training if configured
        if self.config.USE_MIXED_PRECISION:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision training enabled")
        
        # Set GPU memory growth
        if self.config.USE_GPU:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"Using GPU: {len(gpus)} device(s) found")
                except RuntimeError as e:
                    print(e)
            else:
                print("No GPU found, using CPU")
    
    def train_model(self, use_processed_data=True):
        """Main training pipeline"""
        print("Starting model training...")
        
        # Load or prepare data
        if use_processed_data and self.check_processed_data_exists():
            print("Loading processed data...")
            data_processor = DataPreprocessor()
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_processor.load_processed_data()
        else:
            print("Preparing fresh data...")
            data_processor = DataPreprocessor()
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_processor.prepare_dataset()
        
        # Create data generators for training with augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=self.config.ROTATION_RANGE,
            width_shift_range=self.config.WIDTH_SHIFT_RANGE,
            height_shift_range=self.config.HEIGHT_SHIFT_RANGE,
            shear_range=self.config.SHEAR_RANGE,
            zoom_range=self.config.ZOOM_RANGE,
            brightness_range=self.config.BRIGHTNESS_RANGE,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
        
        train_generator = train_datagen.flow(X_train, y_train, batch_size=self.config.BATCH_SIZE)
        
        # Build and compile model
        model_builder = PestDetectionModel()
        model = model_builder.build_model(fine_tune=False)  # Start without fine-tuning
        
        # Print model summary
        model_builder.get_model_summary()
        
        # Create callbacks
        callbacks = model_builder.create_callbacks()
        
        # First phase: Train the classifier
        print("\n" + "="*50)
        print("Phase 1: Training the classifier")
        print("="*50)
        
        history_1 = model.fit(
            train_generator,
            epochs=20,  # Initial training epochs
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Second phase: Fine-tuning
        print("\n" + "="*50)
        print("Phase 2: Fine-tuning the model")
        print("="*50)
        
        # Rebuild model with fine-tuning enabled
        model = model_builder.build_model(fine_tune=True)
        model_builder.compile_model(learning_rate=self.config.LEARNING_RATE/10)  # Lower learning rate
        
        # Continue training with fine-tuning
        history_2 = model.fit(
            train_generator,
            epochs=self.config.EPOCHS - 20,  # Remaining epochs
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            initial_epoch=20
        )
        
        # Combine training histories
        history = self.combine_histories(history_1, history_2)
        
        # Evaluate model
        self.evaluate_model(model, X_test, y_test)
        
        # Save final model
        self.save_model(model)
        
        # Plot training history
        self.plot_training_history(history)
        
        return model, history
    
    def check_processed_data_exists(self):
        """Check if processed data files exist"""
        required_files = [
            'X_train.npy', 'y_train.npy', 'X_val.npy', 
            'y_val.npy', 'X_test.npy', 'y_test.npy'
        ]
        
        for file_name in required_files:
            if not os.path.exists(os.path.join(self.config.PROCESSED_DATA_DIR, file_name)):
                return False
        return True
    
    def combine_histories(self, hist1, hist2):
        """Combine two training histories"""
        combined_history = {}
        for key in hist1.history.keys():
            combined_history[key] = hist1.history[key] + hist2.history[key]
        return type('History', (), {'history': combined_history})()
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance on test set"""
        print("\n" + "="*50)
        print("Model Evaluation")
        print("="*50)
        
        # Get predictions
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Calculate accuracy
        test_accuracy = np.mean(y_pred_classes == y_true_classes)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            y_true_classes, y_pred_classes, 
            target_names=self.config.CLASS_NAMES,
            digits=4
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        self.plot_confusion_matrix(cm, self.config.CLASS_NAMES)
        
        # Per-class accuracy
        print("\nPer-class Accuracy:")
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        for i, class_name in enumerate(self.config.CLASS_NAMES):
            print(f"{class_name}: {class_accuracies[i]:.4f}")
    
    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.config.LOG_DIR, 'confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Training Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.config.LOG_DIR, 'training_history.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, model):
        """Save the trained model"""
        model_path = os.path.join(self.config.SAVED_MODELS_DIR, 'pest_detection_model.h5')
        model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        # Also save in TensorFlow SavedModel format
        tf_model_path = os.path.join(self.config.SAVED_MODELS_DIR, 'pest_detection_model_tf')
        model.save(tf_model_path, save_format='tf')
        print(f"TensorFlow model saved to: {tf_model_path}")
    
    def train_alternative_model(self):
        """Train alternative custom CNN model"""
        print("Training alternative CNN model...")
        
        # Load data
        data_processor = DataPreprocessor()
        if self.check_processed_data_exists():
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_processor.load_processed_data()
        else:
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_processor.prepare_dataset()
        
        # Build alternative model
        model_builder = PestDetectionModel()
        model = model_builder.build_alternative_model()
        
        # Create data augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=self.config.ROTATION_RANGE,
            width_shift_range=self.config.WIDTH_SHIFT_RANGE,
            height_shift_range=self.config.HEIGHT_SHIFT_RANGE,
            shear_range=self.config.SHEAR_RANGE,
            zoom_range=self.config.ZOOM_RANGE,
            brightness_range=self.config.BRIGHTNESS_RANGE,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
        
        train_generator = train_datagen.flow(X_train, y_train, batch_size=self.config.BATCH_SIZE)
        
        # Train model
        callbacks = model_builder.create_callbacks()
        
        history = model.fit(
            train_generator,
            epochs=self.config.EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate and save
        self.evaluate_model(model, X_test, y_test)
        
        # Save alternative model
        alt_model_path = os.path.join(self.config.SAVED_MODELS_DIR, 'alternative_cnn_model.h5')
        model.save(alt_model_path)
        print(f"Alternative model saved to: {alt_model_path}")
        
        return model, history

def main():
    parser = argparse.ArgumentParser(description='Train Pest Detection Model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model_type', choices=['efficientnet', 'custom'], 
                       default='efficientnet', help='Model architecture to use')
    parser.add_argument('--use_processed', action='store_true', 
                       help='Use previously processed data if available')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    
    # Create trainer and train model
    trainer = ModelTrainer()
    
    if args.model_type == 'efficientnet':
        model, history = trainer.train_model(use_processed_data=args.use_processed)
    else:
        model, history = trainer.train_alternative_model()
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()