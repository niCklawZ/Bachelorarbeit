"""
Author: Nick Kottek
Date: 14.05.2024
"""

import glob
import os.path
import matplotlib.pyplot as plt
import json
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import numpy as np

# Plot a summary of the used dataset

# Read first image of a-i
a_img = Image.open("../dataset_compressed/a/a001.jpg").convert("LA")
b_img = Image.open("../dataset_compressed/b/b001.jpg").convert("LA")
c_img = Image.open("../dataset_compressed/c/c001.jpg").convert("LA")
d_img = Image.open("../dataset_compressed/d/d001.jpg").convert("LA")
e_img = Image.open("../dataset_compressed/e/e001.jpg").convert("LA")
f_img = Image.open("../dataset_compressed/f/f001.jpg").convert("LA")
g_img = Image.open("../dataset_compressed/g/g001.jpg").convert("LA")
h_img = Image.open("../dataset_compressed/h/h001.jpg").convert("LA")
i_img = Image.open("../dataset_compressed/i/i001.jpg").convert("LA")

# Plot images
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

axes[0, 0].imshow(a_img)
axes[0, 0].set_title("a", fontsize=24)
axes[0, 0].axis('off')

axes[0, 1].imshow(b_img)
axes[0, 1].set_title("b", fontsize=24)
axes[0, 1].axis('off')

axes[0, 2].imshow(c_img)
axes[0, 2].set_title("c", fontsize=24)
axes[0, 2].axis('off')

axes[1, 0].imshow(d_img)
axes[1, 0].set_title("d", fontsize=24)
axes[1, 0].axis('off')

axes[1, 1].imshow(e_img)
axes[1, 1].set_title("e", fontsize=24)
axes[1, 1].axis('off')

axes[1, 2].imshow(f_img)
axes[1, 2].set_title("f", fontsize=24)
axes[1, 2].axis('off')

axes[2, 0].imshow(g_img)
axes[2, 0].set_title("g", fontsize=24)
axes[2, 0].axis('off')

axes[2, 1].imshow(h_img)
axes[2, 1].set_title("h", fontsize=24)
axes[2, 1].axis('off')

axes[2, 2].imshow(i_img)
axes[2, 2].set_title("i", fontsize=24)
axes[2, 2].axis('off')

plt.tight_layout()

# Save plot to file
plt.savefig('../results/dataset_visualized.png', dpi=400)
plt.close()

# Plot the history of accuracy and loss in training and validation

# Get most recent file in results directory to create
list_of_files = glob.glob('../results/results-*.json')
latest_file = max(list_of_files, key=os.path.getctime)

# Uncomment next line if you want to specify another .json file that is not the most recent one
# latest_file = '../results\\results-2024-04-29-17-51-20.json'

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

plt.tight_layout()

# Save plot to file
plt.savefig("../results/" + file_name + ".png", dpi=400)
plt.close()

# Plot the confusion matrix of the evaluation

# Get most recent file in results directory to create
list_of_files = glob.glob('../results/evaluationResults-*.json')
latest_file = max(list_of_files, key=os.path.getctime)

# Uncomment next line if you want to specify another .json file that is not the most recent one
latest_file = '../results\\evaluationResults-2024-05-01-19-50-35.json'

file_name = latest_file.split('\\')[1].split('.')[0]

# Get result data from json file
with open(latest_file, "r") as openfile:
    json_object = json.load(openfile)

classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
           'x', 'y', 'sch']
y = json_object['y']
pred = json_object['pred']

# Calculate confusion matrix
conf_mat = confusion_matrix(y, pred)

# Plot confusion matrix
f, ax = plt.subplots()
f.set_figheight(10)
f.set_figwidth(10)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=classes)

# Create custom color map
RdBu = mpl.colormaps['Blues'].resampled(256)
newcolors = RdBu(np.linspace(0, 1, 256))
white = np.array([1, 1, 1, 1])
red = np.array([176 / 255, 36 / 255, 28 / 255, 1])
newcolors[:1, :] = white
newcolors[1:64, :] = red
newcmp = ListedColormap(newcolors)

disp.plot(cmap=newcmp, colorbar=False, ax=ax)

plt.xlabel("Vorhergesage Klasse")
plt.ylabel("Tatsächliche Klasse")

plt.tight_layout()

# Save plot to file
plt.savefig("../results/" + file_name + ".png", dpi=400)
plt.close()

# Plot a bar chart of the classification duration statistics

file = "../results/classificationStatistics.json"

with open(file, "r") as openfile:
    durationStatistics = json.load(openfile)

plt.bar(durationStatistics.keys(), durationStatistics.values(), color='royalblue')

for duration, amount in durationStatistics.items():
    plt.text(duration, amount + 1, str(amount), ha='center', va='bottom')

totalAmount = sum(durationStatistics.values())
plt.text(0.98, 0.95, f"Gesamtmenge: {totalAmount}", ha='right', va='center', transform=plt.gca().transAxes, fontsize=11)

plt.xlabel("Klassifizierungsdauer (ms)")
plt.ylabel("Häufigkeit")
plt.xticks(list(durationStatistics.keys()))
plt.grid(False)

plt.tight_layout()

plt.savefig("../results/classificationStatistics.png")
plt.close()
