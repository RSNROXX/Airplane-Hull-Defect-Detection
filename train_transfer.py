from src.dataloader import load_data
from src.model_defs import build_transfer_model
from src.utils import plot_history
import os

# 1. Load Data
train_data, val_data = load_data()

# 2. Build Model (VGG16 Transfer)
model = build_transfer_model()

# 3. Train
# Note: Transfer learning usually converges faster, so fewer epochs might be needed
print(
    "----Starting training for Transfer Learning Model (VGG16)----"
    )
history = model.fit(train_data, epochs=5, validation_data=val_data)

# 4. Save
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')
    
model.save('saved_models/transfer_model.h5')
print(
    "----Model saved to saved_models/transfer_model.h5----"
    )

# 5. Plot Results
plot_history(history, save_name="transfer_model_results.png")