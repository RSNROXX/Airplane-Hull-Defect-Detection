from src.dataloader import load_data   #  Importing from our new structure
from src.model_defs import build_custom_cnn
from src.utils import plot_history
import os

#  1. Load Data
train_data, val_data = load_data()

#  2. Build Model (using the function we moved to src)
model = build_custom_cnn()

#  3. Train
print(
    "  ---Starting training for Custom Model---  "
    )
history = model.fit(train_data, epochs=10, validation_data=val_data)

#  4. Save
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')
    
model.save('saved_models/custom_model.h5')
print(
    "---Model saved to saved_models/custom_model.h5---"
    )

#  5. Plot Results
plot_history(history, save_name="custom_model_results.png")