import matplotlib.pyplot as plt
import json

# Get result data from json file
with open("../results/results.json", "r") as openfile:
    json_object = json.load(openfile)

# Get epoch count
epochs_range = range(len(json_object['acc']))

# Create two subplots for accuracy and loss in training and validation
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, json_object['acc'], label='Training Accuracy')
plt.plot(epochs_range, json_object['val_acc'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, json_object['loss'], label='Training Loss')
plt.plot(epochs_range, json_object['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# Save plot to file
plt.savefig("../results/results.png", dpi=400)

# Show plots
plt.show()

