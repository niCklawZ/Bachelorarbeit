import glob
import os.path
import matplotlib.pyplot as plt
import json

# Get most recent file in results directory to create
list_of_files = glob.glob('../results/*.json')
latest_file = max(list_of_files, key=os.path.getctime)

# Uncomment next line if you want to specify another .json file that is not the most recent one
# latest_file = '../results\\results-2024-01-12-15-00-03.json'

file_name = latest_file.split('\\')[1].split('.')[0]

# Get result data from json file
with open(latest_file, "r") as openfile:
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
plt.savefig("../results/" + file_name + ".png", dpi=400)

# Show plots
plt.show()

