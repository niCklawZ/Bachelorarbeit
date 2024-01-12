import tensorflow as tf
import json
import time
from datetime import datetime

# Test tensorflow GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Set some params
batch_size = 128
# batch_size = 20
img_height = 64
img_width = 64

# Generate training, validation and evaluation datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    '../dataset_compressed',
    validation_split=0.3,
    subset="training",
    seed=111,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    '../dataset_compressed',
    validation_split=0.3,
    subset="validation",
    seed=111,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_batches = tf.data.experimental.cardinality(val_ds)
eval_ds = val_ds.take((2 * val_batches) // 3)
val_ds = val_ds.skip((2 * val_batches) // 3)

# Extract class names
class_names = train_ds.class_names
print(class_names)

# Preload data for faster access
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
eval_ds = eval_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

# Define model
'''
model = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])
'''

# Source: Rahman, M. M., Islam, M. S., Rahman, M. H., Sassi, R., Rivolta, M. W., & Aktaruzzaman, M. (2019).
# A New Benchmark on American Sign Language Recognition using Convolutional Neural
# Network. 2019 International Conference on Sustainable Technologies for Industry 4.0 (STI), 1–6.
model = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.Conv2D(32, (5, 5)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(64, (5, 5)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(128, (3, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(256, (3, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(384, (3, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(512, (3, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(84, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-6)),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes)
])

# Compile model
'''
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
'''

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    metrics=['accuracy'],
)

# Print summary of the model
model.summary()

# Set epochs
epochs = 200

# Train model
'''
# Modell trainieren
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
)
'''

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.75, patience=6)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

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
results = model.evaluate(eval_ds, batch_size=batch_size)
print("test loss, test acc:", results)

# Export data of accuracy und loss while training and validation to generate a graph from it
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

json_data = {
    'acc': acc,
    'val_acc': val_acc,
    'loss': loss,
    'val_loss': val_loss
}
json_object = json.dumps(json_data, indent=4)
current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
with open("../results/results-" + current_time + ".json", "w") as outfile:
    outfile.write(json_object)

# Save model to .keras file with information about date, evaluation results and training time for faster comparison between models
model.save('../models/trainedModel-' + current_time + '-eval_loss ' + str(round(results[0], 3)) + '-eval_acc ' + str(round(results[1], 3)) + '-train_time ' + str(round(duration, 3)) + '.keras')
