import tensorflow as tf
import numpy as np
import os
from src.dataloader import load_data

# CONFIG
MODEL_PATH = os.path.join('saved_models', 'best_model.h5')

def run_auto_test():
    print(f"\n🧪 STARTING AUTO-TEST on {MODEL_PATH}...")
    
    # 1. Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ FAILED: {MODEL_PATH} not found. Run compare.py first.")
        return False # Return failure code
        
    # 2. Load the Best Model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"❌ FAILED: Model file is corrupted. Error: {e}")
        return False

    # 3. Get a small sample of data (Validation set)
    # We use the src.dataloader we built earlier
    print("...Loading verification data from src/dataloader.py")
    _, val_ds = load_data()
    
    # 4. Test on one batch
    print("...Running prediction on a sample batch")
    for images, labels in val_ds.take(1):
        # Predict
        predictions = model.predict(images)
        
        # CHECKS
        # A. Check dimensions
        if len(predictions) != len(labels):
            print("❌ FAILED: Prediction count mismatch.")
            return False
            
        # B. Check probability range (0.0 to 1.0)
        if np.any(predictions < 0) or np.any(predictions > 1):
            print("❌ FAILED: Model output invalid probabilities (must be 0-1).")
            return False
            
        # C. Show a sample
        print(f"✅ PASSED: Processed {len(predictions)} images successfully.")
        print(f"   Sample Output: {predictions[0][0]:.4f} (True Label: {labels[0]})")
        
    print("\n🎉 AUTO-TEST PASSED. Model is safe to use.")
    return True

if __name__ == "__main__":
    success = run_auto_test()
    # Exit with code 0 if success, 1 if failed (so main.py knows)
    exit(0 if success else 1)