import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import cv2
import tensorflow as tf
from config import Config

def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def count_images_in_directory(directory_path):
    """Count total images in a directory and its subdirectories"""
    total_count = 0
    class_counts = {}
    
    if not os.path.exists(directory_path):
        return total_count, class_counts
    
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            # Count images in subdirectory
            image_files = [f for f in os.listdir(item_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            count = len(image_files)
            class_counts[item] = count
            total_count += count
    
    return total_count, class_counts

def plot_class_distribution(class_counts, title="Class Distribution"):
    """Plot class distribution as bar chart"""
    if not class_counts:
        print("No data to plot")
        return
    
    plt.figure(figsize=(10, 6))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    bars = plt.bar(classes, counts, color=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def calculate_class_weights(y_train):
    """Calculate class weights for handling imbalanced datasets"""
    from sklearn.utils.class_weight import compute_class_weight
    
    # Get unique classes
    classes = np.unique(np.argmax(y_train, axis=1))
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=np.argmax(y_train, axis=1)
    )
    
    # Convert to dictionary
    class_weight_dict = dict(zip(classes, class_weights))
    
    return class_weight_dict

def plot_training_metrics(history, metrics=['accuracy', 'loss'], save_path=None):
    """Plot training history metrics"""
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        if metric in history.history:
            axes[i].plot(history.history[metric], label=f'Training {metric.title()}')
            if f'val_{metric}' in history.history:
                axes[i].plot(history.history[f'val_{metric}'], label=f'Validation {metric.title()}')
            
            axes[i].set_title(f'Model {metric.title()}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.title())
            axes[i].legend()
            axes[i].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_roc_curves(y_true, y_pred, class_names, save_path=None):
    """Plot ROC curves for multi-class classification"""
    n_classes = len(class_names)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, color in zip(range(n_classes), colors[:n_classes]):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Multi-class Classification')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_gradcam_heatmap(model, img_array, class_idx, last_conv_layer_name):
    """Create Grad-CAM heatmap for model interpretability"""
    try:
        # Create a model that maps the input image to the activations of the last conv layer
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        # Compute the gradient of the top predicted class for the input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]
        
        # Extract the gradients of the top predicted class with regard to
        # the activations of the last convolutional layer
        grads = tape.gradient(loss, conv_outputs)
        
        # Pool the gradients over all the axes leaving out the channel dimension
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel in the feature-map array
        # by "how important this channel is" with regard to the top predicted class
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize the heatmap between 0 & 1 for visualization
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    except Exception as e:
        print(f"Error creating Grad-CAM heatmap: {str(e)}")
        return None

def overlay_heatmap_on_image(img, heatmap, alpha=0.6):
    """Overlay heatmap on original image"""
    try:
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Convert heatmap to RGB
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        
        # Convert BGR to RGB
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on image
        superimposed_img = heatmap_colored * alpha + img * (1 - alpha)
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        return superimposed_img
    
    except Exception as e:
        print(f"Error overlaying heatmap: {str(e)}")
        return img

def save_sample_predictions(model, X_test, y_test, class_names, num_samples=8, save_dir='sample_predictions'):
    """Save sample predictions with confidence scores"""
    try:
        # Create save directory
        create_directory_if_not_exists(save_dir)
        
        # Get random samples
        indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
        
        # Make predictions
        predictions = model.predict(X_test[indices], verbose=0)
        
        # Create subplot
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            if i >= len(axes):
                break
                
            # Get image and predictions
            image = X_test[idx]
            true_class = np.argmax(y_test[idx])
            pred_class = np.argmax(predictions[i])
            confidence = predictions[i][pred_class]
            
            # Plot image
            axes[i].imshow(image)
            
            # Create title with prediction info
            true_label = class_names[true_class]
            pred_label = class_names[pred_class]
            color = 'green' if true_class == pred_class else 'red'
            
            title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}'
            axes[i].set_title(title, color=color, fontsize=10)
            axes[i].axis('off')
        
        # Remove empty subplots
        for i in range(len(indices), len(axes)):
            axes[i].remove()
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(save_dir, 'sample_predictions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Sample predictions saved to: {save_path}")
        
    except Exception as e:
        print(f"Error saving sample predictions: {str(e)}")

def calculate_model_size(model_path):
    """Calculate model size in MB"""
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    return 0

def print_system_info():
    """Print system information"""
    print("System Information:")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python version: {tf.version.VERSION}")
    
    # GPU information
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    else:
        print("GPU: Not available")
    
    # Memory information
    import psutil
    memory = psutil.virtual_memory()
    print(f"System RAM: {memory.total / (1024**3):.1f} GB")
    print(f"Available RAM: {memory.available / (1024**3):.1f} GB")

def validate_image_file(file_path):
    """Validate if file is a valid image"""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    # Check file extension
    if not file_path.lower().endswith(valid_extensions):
        return False, "Invalid file extension"
    
    # Check if file exists
    if not os.path.exists(file_path):
        return False, "File not found"
    
    # Try to load image
    try:
        img = cv2.imread(file_path)
        if img is None:
            return False, "Cannot read image file"
        return True, "Valid image file"
    except Exception as e:
        return False, f"Error reading image: {str(e)}"

def create_model_comparison_table(models_info):
    """Create a comparison table for different models"""
    try:
        import pandas as pd
        
        df = pd.DataFrame(models_info)
        print("\nModel Comparison:")
        print("=" * 80)
        print(df.to_string(index=False))
        
        return df
    
    except ImportError:
        print("pandas not available. Showing simple comparison:")
        print("\nModel Comparison:")
        print("=" * 80)
        
        for i, model_info in enumerate(models_info):
            print(f"\nModel {i+1}:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")

def estimate_training_time(num_images, epochs, batch_size, use_gpu=True):
    """Estimate training time based on dataset size"""
    # Base time per image (in seconds) - these are rough estimates
    base_time_per_image_gpu = 0.01  # 10ms per image on GPU
    base_time_per_image_cpu = 0.1   # 100ms per image on CPU
    
    base_time = base_time_per_image_gpu if use_gpu else base_time_per_image_cpu
    
    # Calculate total time
    batches_per_epoch = num_images // batch_size
    total_batches = batches_per_epoch * epochs
    estimated_seconds = total_batches * batch_size * base_time
    
    # Convert to hours and minutes
    hours = int(estimated_seconds // 3600)
    minutes = int((estimated_seconds % 3600) // 60)
    
    return hours, minutes, estimated_seconds