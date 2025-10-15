import os

class Config:
    # Data paths
    DATA_DIR = "data"
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    
    # Model paths
    MODEL_DIR = "models"
    SAVED_MODELS_DIR = os.path.join(MODEL_DIR, "saved_models")
    CHECKPOINTS_DIR = os.path.join(MODEL_DIR, "checkpoints")
    
    # Image parameters
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_CHANNELS = 3
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Model parameters
    NUM_CLASSES = 4  # healthy, aphids, caterpillars, spider_mites
    CLASS_NAMES = ['healthy', 'aphids', 'caterpillars', 'spider_mites']
    
    # Data augmentation parameters
    ROTATION_RANGE = 20
    WIDTH_SHIFT_RANGE = 0.2
    HEIGHT_SHIFT_RANGE = 0.2
    SHEAR_RANGE = 0.2
    ZOOM_RANGE = 0.2
    BRIGHTNESS_RANGE = [0.8, 1.2]
    
    # Training settings
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    MIN_LR = 1e-7
    
    # Hardware settings
    USE_MIXED_PRECISION = True
    USE_GPU = True
    
    # Logging
    LOG_DIR = "logs"
    TENSORBOARD_LOG_DIR = os.path.join(LOG_DIR, "tensorboard")
    
    # Create directories if they don't exist
    @classmethod
    def create_directories(cls):
        directories = [
            cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR,
            cls.MODEL_DIR, cls.SAVED_MODELS_DIR, cls.CHECKPOINTS_DIR,
            cls.LOG_DIR, cls.TENSORBOARD_LOG_DIR
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)