import os
import shutil
import tensorflow as tf
from src.dataloader import load_data

# CONFIG
MODEL_DIR = 'saved_models'
CUSTOM_MODEL_PATH = os.path.join(MODEL_DIR, 'custom_model.h5')
TRANSFER_MODEL_PATH = os.path.join(MODEL_DIR, 'transfer_model.h5')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.h5')

def compare_models():
    print("\n⚔️  STARTING MODEL BATTLE: Custom vs. Transfer ⚔️")

    # 1. Check if models exist
    if not os.path.exists(CUSTOM_MODEL_PATH) or not os.path.exists(TRANSFER_MODEL_PATH):
        print("❌ Error: One or both models are missing. Run training scripts first.")
        return False

    # 2. Load Data (We only need Validation/Test data for this)
    # Note: load_data() returns (train_ds, val_ds). We discard train_ds with '_'
    print("...Loading Validation Data")
    _, val_ds = load_data()

    # 3. Evaluate Custom Model
    print(f"\nEvaluating Custom Model ({CUSTOM_MODEL_PATH})...")
    custom_model = tf.keras.models.load_model(CUSTOM_MODEL_PATH)  #  type:ignore
    custom_loss, custom_acc = custom_model.evaluate(val_ds, verbose=0)
    print(f"   -> Accuracy: {custom_acc:.4f}")

    # 4. Evaluate Transfer Model
    print(f"\nEvaluating Transfer Model ({TRANSFER_MODEL_PATH})...")
    transfer_model = tf.keras.models.load_model(TRANSFER_MODEL_PATH)  #  type:ignore
    trans_loss, trans_acc = transfer_model.evaluate(val_ds, verbose=0)
    print(f"   -> Accuracy: {trans_acc:.4f}")

    # 5. Declare Winner
    print("\n" + "="*30)
    if trans_acc > custom_acc:
        print(f"🏆 WINNER: Transfer Learning Model ({trans_acc:.4f})")
        winner_src = TRANSFER_MODEL_PATH
    else:
        print(f"🏆 WINNER: Custom Model ({custom_acc:.4f})")
        winner_src = CUSTOM_MODEL_PATH
    print("="*30)

    # 6. Save the Best Model
    # We copy the winner to a generic name 'best_model.h5'
    # This allows predict.py to always just load 'best_model.h5' without caring who won.
    shutil.copy(winner_src, BEST_MODEL_PATH)
    print(f"\n✅ Saved winner as: {BEST_MODEL_PATH}")
    return True

if __name__ == "__main__":
    compare_models()