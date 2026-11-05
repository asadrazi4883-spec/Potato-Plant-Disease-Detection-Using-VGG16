# STEP 1: IMPORT LIBRARIES

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import json 


# STEP 2: LOAD DATASET

data_dir = "dataset"

data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    image_size=(224, 224),
    batch_size=32,
    shuffle=True
)

class_names = data.class_names

print("Classes:", class_names)
print("Dataset loaded successfully!")


# STEP 3: VGG16 PREPROCESSING

from tensorflow.keras.applications.vgg16 import preprocess_input

data = data.map(lambda x, y: (preprocess_input(x), y))

print("VGG16 preprocessing completed!")


# STEP 4: TRAIN-TEST SPLIT

train_size = int(0.8 * len(data))

train = data.take(train_size)

test = data.skip(train_size)

print("Training and testing datasets created!")


# STEP 5: DATA AUGMENTATION

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
])

print("Data augmentation created successfully!")


# STEP 6: LOAD VGG16 MODEL

base_model = tf.keras.applications.VGG16(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = True

for layer in base_model.layers[:-4]:
    layer.trainable = False

print("VGG16 model loaded successfully!")


# STEP 7: BUILD MODEL

model = keras.Sequential([
    data_augmentation,
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

print("Model architecture created successfully!")


# STEP 8: COMPILE MODEL

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled successfully!")


# STEP 9: EARLY STOPPING

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

print("Early stopping configured successfully!")


# STEP 10: TRAIN MODEL

history = model.fit(
    train,
    validation_data=test,
    epochs=15,
    #callbacks=[early_stop]
)

print("Model training completed successfully!")


# STEP 11: CREATE MODEL FOLDER

os.makedirs("model", exist_ok=True)


# STEP 12: SAVE TRAINED MODEL

model.save("model/potato_disease_model.keras")

print("Trained model saved successfully!")


# STEP 13: SAVE TRAINING HISTORY

with open("model/training_history.json", "w") as f:
    json.dump(history.history, f)

print("Training history saved successfully!")


# STEP 14: EVALUATE MODEL

loss, accuracy = model.evaluate(test)

print(f"\nModel Loss: {loss:.4f}")
print(f"Model Accuracy: {accuracy * 100:.2f}%")


# STEP 15: ACCURACY GRAPH

plt.figure(figsize=(8, 5))

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.title('Accuracy vs Epochs')

plt.legend()

plt.show()


# STEP 16: LOSS GRAPH

plt.figure(figsize=(8, 5))

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.title('Loss vs Epochs')

plt.legend()

plt.show()

print("\nAll tasks completed successfully!")
