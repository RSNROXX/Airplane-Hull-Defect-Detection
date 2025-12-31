import os
import random
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

# CONFIG
# We go up one level to find the 'data' folder from root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'saved_models', 'best_model.h5')
TEST_DIR = os.path.join(BASE_DIR, 'data', 'test')
IMG_SIZE = (224, 224)

def predict_hull_damage(image_path):
    """Loads a single image and uses the best model to predict Crack vs Dent."""
    
    # 1. Load Model (only once in a real app, but safe here)
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: 'saved_models/best_model.h5' not found.")
        return

    # Suppress TensorFlow generic loading messages
    tf.get_logger().setLevel('ERROR') 
    model = tf.keras.models.load_model(MODEL_PATH)

    # 2. Preprocess Image
    try:
        img = load_img(image_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Shape: (1, 224, 224, 3)
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    # 3. Predict
    prediction = model.predict(img_array, verbose=0)
    score = prediction[0][0] # Probability

    # 4. Report
    print("\n" + "="*40)
    print(f"📷 IMAGE: {os.path.basename(image_path)}")
    print("="*40)
    
    # Since classes are usually alphabetical: 0=crack, 1=dent
    if score > 0.5:
        confidence = score * 100
        print(f"🛠️  RESULT: DENT DETECTED")
        print(f"   Confidence: {confidence:.2f}%")
    else:
        confidence = (1 - score) * 100
        print(f"⚡ RESULT: CRACK DETECTED")
        print(f"   Confidence: {confidence:.2f}%")
    print("="*40 + "\n")

def get_random_test_image():
    """Finds all images in data/test/crack and data/test/dent and picks one."""
    all_images = []
    
    if not os.path.exists(TEST_DIR):
        print(f"⚠️ Warning: Test folder not found at {TEST_DIR}")
        return None

    # Walk through test/crack and test/dent
    for root, dirs, files in os.walk(TEST_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, file))
    
    if not all_images:
        return None
        
    return random.choice(all_images)

if __name__ == "__main__":
    print(f"\n🎮 ENTERING INTERACTIVE TEST MODE")
    print(f"Scanning folder: {TEST_DIR} ...")
    
    while True:
        # 1. Pick a random image
        random_img = get_random_test_image()
        
        if not random_img:
            print("❌ No images found in data/test/. Please add some images.")
            break
            
        # 2. Run Prediction
        predict_hull_damage(random_img)
        
        # 3. Ask User to Continue
        user_input = input("🔄 Test another random image? (Press Enter for YES, type 'q' to QUIT): ")
        
        if user_input.lower() == 'q':
            print("\n👋 Exiting Interactive Mode.")
            break