"""
Author: Nick Kottek
Date: 14.05.2024
"""

import numpy as np
import tensorflow as tf
import json
import time
from datetime import datetime

# Test tensorflow GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Set some params
batch_size = 64
img_height = 64
img_width = 64

# Generate training, validation and evaluation datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    '../dataset_compressed',
    validation_split=0.3,
    subset="training",
    seed=118,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical')

val_ds = tf.keras.utils.image_dataset_from_directory(
    '../dataset_compressed',
    validation_split=0.3,
    subset="validation",
    seed=118,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical')

val_batches = tf.data.experimental.cardinality(val_ds)
eval_ds = val_ds.take((2 * val_batches) // 3)
val_ds = val_ds.skip((2 * val_batches) // 3)

# Extract class names
class_names = train_ds.class_names
print(class_names)

# Preload data for faster access
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1560).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
eval_ds = eval_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

"""
Define model

Based on:
Rahman, M. M., Islam, M. S., Rahman, M. H., Sassi, R., Rivolta, M. W., & Aktaruzzaman, M. (2019).
A New Benchmark on American Sign Language Recognition using Convolutional Neural
Network. 2019 International Conference on Sustainable Technologies for Industry 4.0 (STI), 1–6.
"""

model = tf.keras.Sequential([
    tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="constant", fill_value=0,
                                      input_shape=(img_height, img_width, 3)),
    tf.keras.layers.RandomRotation(factor=0.1, fill_mode="constant", fill_value=0),
    tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1, fill_mode="constant", fill_value=0),
    tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.Conv2D(32, (5, 5)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(32, (5, 5)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(64, (3, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.Conv2D(256, (3, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.Conv2D(256, (3, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(64),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes),
    tf.keras.layers.Softmax()
])

# Set optimizer function
optimizer = tf.keras.optimizers.RMSprop(0.0005)

# Compile model
model.compile(
    loss=tf.losses.CategoricalCrossentropy(),  # Set loss function
    optimizer=optimizer,
    metrics=['accuracy'],
)

# Print summary of the model
model.summary()

# Set epochs
epochs = 200

# Set reduce learnrate on plateau and early stopping
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=6)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

# Train model and measure duration
start_time = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[reduce_lr, early_stop]
)
end_time = time.time()
duration = end_time - start_time

# Evaluate model with independent dataset
evaluationResult = model.evaluate(eval_ds, batch_size=batch_size)
print("test loss, test acc:", evaluationResult)

# Get the predictions of the independent dataset
evaluationPrediction = model.predict(eval_ds)

# Get the correct labels of eval_ds dataset in numerical form
labels = np.concatenate([batch_labels.numpy() for _, batch_labels in eval_ds])
y = np.argmax(labels, axis=1)

# Get the predicted labels of eval_ds dataset in numerical form
pred = np.argmax(evaluationPrediction, axis=1)

# Export data of evaluation results to generate confusion matrix from it
evaluationResultData = {
    "y": y.tolist(),
    "pred": pred.tolist()
}
evaluationResultDataJSON = json.dumps(evaluationResultData, indent=4)

current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

with open("../results/evaluationResults-" + current_time + ".json", "w") as outfile:
    outfile.write(evaluationResultDataJSON)

# Export history data of accuracy und loss while training and validation to generate a graph from it
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

historyData = {
    'acc': acc,
    'val_acc': val_acc,
    'loss': loss,
    'val_loss': val_loss
}
historyDataJSON = json.dumps(historyData, indent=4)

with open("../results/results-" + current_time + ".json", "w") as outfile:
    outfile.write(historyDataJSON)

# Export model to .keras file with information about date, evaluation results and training time for faster comparison between models
model.save(
    '../models/trainedModel-' + current_time + '-eval_loss ' + str(round(evaluationResult[0], 3)) + '-eval_acc ' + str(
        round(evaluationResult[1], 3)) + '-train_time ' + str(round(duration, 3)) + '.keras')
