import os
import tensorflow as tf

# Define paths to the specific folders
BASE_DIR = os.path.dirname(os.path.dirname(__file__)) # Project Root
TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'train')
VALID_DIR = os.path.join(BASE_DIR, 'data', 'valid')

def load_data(img_size=(224, 224), batch_size=32):
    print(f"Loading Training Data from: {TRAIN_DIR}")
    print(f"Loading Validation Data from: {VALID_DIR}")

    # Check if folders exist
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VALID_DIR):
        raise FileNotFoundError("❌ Error: 'data/train' or 'data/valid' folders are missing!")

    # Load Training Data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )

    # Load Validation Data
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VALID_DIR,
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False # No need to shuffle validation data
    )

    # PRINT CLASS NAMES (Important so we know which is 0 and which is 1)
    class_names = train_ds.class_names
    print(f"✅ Classes found: {class_names}") 
    # Usually: ['crack', 'dent'] -> crack=0, dent=1 (Alphabetical)
    
    return train_ds, val_ds