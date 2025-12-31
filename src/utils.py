import matplotlib.pyplot as plt
import os

def plot_history(history, save_name="training_plot.png"):
    """
        Plots accuracy/loss and saves it to the 'visualizations' folder.
    """
    
    #  1. Define the professional folder name >> Go up two levels from src/utils.py to get to the Project Root
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    VISUALS_DIR = os.path.join(BASE_DIR, 'visualizations')

    #  2. Create the folder if it doesn't exist
    if not os.path.exists(VISUALS_DIR):
        os.makedirs(VISUALS_DIR)
        print(f"📁 Created folder: {VISUALS_DIR}")

    #  3. Extract data
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # 4. Create the Plot
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'r*-', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'r*-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # 5. Save securely into the new folder
    full_path = os.path.join(VISUALS_DIR, save_name)
    plt.savefig(full_path)
    plt.close() # Close the plot to free up memory
    
    print(f"📊 Graph saved to: {full_path}")