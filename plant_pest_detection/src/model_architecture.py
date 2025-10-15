import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from config import Config

class PestDetectionModel:
    def __init__(self):
        self.config = Config()
        self.model = None
        
    def build_model(self, fine_tune=True):
        """Build CNN model using EfficientNetB0 as backbone"""
        print("Building model architecture...")
        
        # Load pre-trained EfficientNetB0
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH, self.config.IMG_CHANNELS)
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head
        inputs = tf.keras.Input(shape=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH, self.config.IMG_CHANNELS))
        x = base_model(inputs, training=False)
        
        # Global pooling
        x = GlobalAveragePooling2D()(x)
        
        # Dense layers with regularization
        x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Output layer
        outputs = Dense(self.config.NUM_CLASSES, activation='softmax', name='predictions')(x)
        
        self.model = Model(inputs, outputs)
        
        # Fine-tuning: unfreeze top layers of base model
        if fine_tune:
            base_model.trainable = True
            
            # Fine-tune from this layer onwards
            fine_tune_at = 100
            
            # Freeze all the layers before the `fine_tune_at` layer
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
        
        self.compile_model()
        
        return self.model
    
    def compile_model(self, learning_rate=None):
        """Compile the model with optimizer and loss function"""
        if learning_rate is None:
            learning_rate = self.config.LEARNING_RATE
        
        # Use different learning rates for fine-tuning
        if learning_rate < self.config.LEARNING_RATE:
            # Lower learning rate for fine-tuning
            optimizer = Adam(learning_rate=learning_rate/10)
        else:
            optimizer = Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        print(f"Model compiled with learning rate: {learning_rate}")
    
    def get_model_summary(self):
        """Print model summary"""
        if self.model is None:
            print("Model not built yet!")
            return
        
        print("Model Summary:")
        self.model.summary()
        
        print(f"\nTotal parameters: {self.model.count_params():,}")
        
        # Count trainable vs non-trainable parameters
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])
        
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
    
    def create_callbacks(self):
        """Create training callbacks"""
        callbacks = []
        
        # Model checkpoint
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{self.config.CHECKPOINTS_DIR}/best_model.h5",
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping_callback)
        
        # Reduce learning rate on plateau
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=self.config.REDUCE_LR_PATIENCE,
            min_lr=self.config.MIN_LR,
            verbose=1
        )
        callbacks.append(reduce_lr_callback)
        
        # TensorBoard logging
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.config.TENSORBOARD_LOG_DIR,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)
        
        # Learning rate scheduler
        def cosine_annealing_scheduler(epoch, lr):
            import math
            max_lr = self.config.LEARNING_RATE
            min_lr = self.config.MIN_LR
            cycle_length = 10
            cycle = math.floor(1 + epoch / (2 * cycle_length))
            x = abs(epoch / cycle_length - 2 * cycle + 1)
            new_lr = min_lr + (max_lr - min_lr) * max(0, (1 - x))
            return new_lr
        
        lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(
            cosine_annealing_scheduler, verbose=0
        )
        callbacks.append(lr_scheduler_callback)
        
        return callbacks
    
    def build_alternative_model(self):
        """Alternative model architecture using custom CNN"""
        print("Building alternative CNN architecture...")
        
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH, self.config.IMG_CHANNELS)),
            
            # First convolutional block
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Second convolutional block
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Third convolutional block
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Fourth convolutional block
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Global average pooling
            tf.keras.layers.GlobalAveragePooling2D(),
            
            # Dense layers
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Output layer
            tf.keras.layers.Dense(self.config.NUM_CLASSES, activation='softmax')
        ])
        
        self.model = model
        self.compile_model()
        
        return self.model