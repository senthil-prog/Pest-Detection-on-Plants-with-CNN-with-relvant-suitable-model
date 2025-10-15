import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from config import Config

class PestPredictor:
    def __init__(self, model_path=None):
        self.config = Config()
        self.model = None
        self.model_path = model_path or os.path.join(
            self.config.SAVED_MODELS_DIR, 'pest_detection_model.h5'
        )
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                print(f"Loading model from: {self.model_path}")
                self.model = tf.keras.models.load_model(self.model_path)
                print("Model loaded successfully!")
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_image(self, image_path):
        """Preprocess a single image for prediction"""
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not load image from {image_path}")
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Assume it's already a numpy array
                image = image_path
            
            # Resize image
            image = cv2.resize(image, (self.config.IMG_HEIGHT, self.config.IMG_WIDTH))
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict_single_image(self, image_path, show_image=True):
        """Make prediction on a single image"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get prediction results
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.config.CLASS_NAMES[predicted_class_idx]
            confidence = predictions[0][predicted_class_idx]
            
            # Get all class probabilities
            class_probabilities = {
                class_name: float(prob) for class_name, prob 
                in zip(self.config.CLASS_NAMES, predictions[0])
            }
            
            # Display results
            print(f"\nPrediction Results:")
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            print(f"\nAll Class Probabilities:")
            for class_name, prob in sorted(class_probabilities.items(), 
                                         key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
            
            # Show image if requested
            if show_image and isinstance(image_path, str):
                self.display_prediction(image_path, predicted_class, confidence)
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_probabilities': class_probabilities
            }
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            raise
    
    def display_prediction(self, image_path, predicted_class, confidence):
        """Display image with prediction results"""
        try:
            # Load and display image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(image)
            plt.title(f'Predicted: {predicted_class}\nConfidence: {confidence:.4f} ({confidence*100:.2f}%)', 
                     fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error displaying image: {str(e)}")
    
    def predict_batch(self, image_paths, output_file=None):
        """Make predictions on multiple images"""
        results = []
        
        print(f"Making predictions on {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                print(f"\nProcessing image {i}/{len(image_paths)}: {os.path.basename(image_path)}")
                result = self.predict_single_image(image_path, show_image=False)
                result['image_path'] = image_path
                result['image_name'] = os.path.basename(image_path)
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
        
        # Save results if output file specified
        if output_file:
            self.save_batch_results(results, output_file)
        
        return results
    
    def save_batch_results(self, results, output_file):
        """Save batch prediction results to CSV"""
        try:
            import pandas as pd
            
            # Prepare data for CSV
            csv_data = []
            for result in results:
                row = {
                    'image_name': result['image_name'],
                    'image_path': result['image_path'],
                    'predicted_class': result['predicted_class'],
                    'confidence': result['confidence']
                }
                
                # Add individual class probabilities
                for class_name, prob in result['class_probabilities'].items():
                    row[f'prob_{class_name}'] = prob
                
                csv_data.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(csv_data)
            df.to_csv(output_file, index=False)
            print(f"Results saved to: {output_file}")
            
        except ImportError:
            print("pandas not installed. Saving results as text file...")
            self.save_batch_results_txt(results, output_file.replace('.csv', '.txt'))
        except Exception as e:
            print(f"Error saving results: {str(e)}")
    
    def save_batch_results_txt(self, results, output_file):
        """Save batch results as text file"""
        try:
            with open(output_file, 'w') as f:
                f.write("Pest Detection Results\n")
                f.write("=" * 50 + "\n\n")
                
                for result in results:
                    f.write(f"Image: {result['image_name']}\n")
                    f.write(f"Path: {result['image_path']}\n")
                    f.write(f"Predicted Class: {result['predicted_class']}\n")
                    f.write(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)\n")
                    f.write("Class Probabilities:\n")
                    for class_name, prob in sorted(result['class_probabilities'].items(), 
                                                 key=lambda x: x[1], reverse=True):
                        f.write(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)\n")
                    f.write("\n" + "-" * 30 + "\n\n")
            
            print(f"Results saved to: {output_file}")
            
        except Exception as e:
            print(f"Error saving text results: {str(e)}")
    
    def predict_from_directory(self, directory_path, output_file=None):
        """Make predictions on all images in a directory"""
        try:
            # Get all image files in directory
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            image_paths = []
            
            for file_name in os.listdir(directory_path):
                if file_name.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(directory_path, file_name))
            
            if not image_paths:
                print(f"No image files found in {directory_path}")
                return []
            
            print(f"Found {len(image_paths)} images in directory")
            
            # Make batch predictions
            return self.predict_batch(image_paths, output_file)
            
        except Exception as e:
            print(f"Error processing directory: {str(e)}")
            raise
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            print("No model loaded!")
            return
        
        print("Model Information:")
        print(f"Input shape: {self.model.input_shape}")
        print(f"Output shape: {self.model.output_shape}")
        print(f"Total parameters: {self.model.count_params():,}")
        print(f"Classes: {self.config.CLASS_NAMES}")
    
    def visualize_prediction_confidence(self, image_path):
        """Visualize prediction confidence with bar chart"""
        try:
            result = self.predict_single_image(image_path, show_image=False)
            
            # Create bar chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Display image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax1.imshow(image)
            ax1.set_title(f'Input Image\n{os.path.basename(image_path)}')
            ax1.axis('off')
            
            # Create confidence bar chart
            classes = list(result['class_probabilities'].keys())
            confidences = list(result['class_probabilities'].values())
            
            bars = ax2.bar(classes, confidences)
            ax2.set_title('Prediction Confidence')
            ax2.set_ylabel('Probability')
            ax2.set_xlabel('Classes')
            ax2.tick_params(axis='x', rotation=45)
            
            # Color the highest bar differently
            max_idx = np.argmax(confidences)
            bars[max_idx].set_color('red')
            
            # Add value labels on bars
            for bar, conf in zip(bars, confidences):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{conf:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error visualizing prediction: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained pest detection model')
    parser.add_argument('--image_path', type=str, help='Path to single image for prediction')
    parser.add_argument('--directory', type=str, help='Directory containing images for batch prediction')
    parser.add_argument('--model_path', type=str, help='Path to trained model file')
    parser.add_argument('--output', type=str, help='Output file for batch results')
    parser.add_argument('--visualize', action='store_true', help='Show visualization of prediction confidence')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = PestPredictor(model_path=args.model_path)
    predictor.get_model_info()
    
    if args.image_path:
        # Single image prediction
        if args.visualize:
            predictor.visualize_prediction_confidence(args.image_path)
        else:
            predictor.predict_single_image(args.image_path)
    
    elif args.directory:
        # Batch prediction on directory
        results = predictor.predict_from_directory(args.directory, args.output)
        
        # Print summary
        if results:
            print(f"\nBatch Prediction Summary:")
            print(f"Total images processed: {len(results)}")
            
            # Class distribution
            class_counts = {}
            for result in results:
                pred_class = result['predicted_class']
                class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
            
            print("Predicted class distribution:")
            for class_name, count in sorted(class_counts.items()):
                print(f"  {class_name}: {count} images")
    
    else:
        print("Please provide either --image_path or --directory argument")
        print("Use --help for more information")

if __name__ == "__main__":
    main()