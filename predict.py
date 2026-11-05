# STEP 1: IMPORT LIBRARIES

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import json

from tensorflow.keras.applications.vgg16 import preprocess_input


# STEP 2: LOAD TRAINED MODEL

model = tf.keras.models.load_model(
    "model/potato_disease_model.keras"
)

print("Trained model loaded successfully!")


# STEP 3: LOAD CLASS NAMES

data_dir = "dataset"

class_names = sorted(os.listdir(data_dir))

print("Classes:", class_names)


# STEP 4: LOAD TRAINING HISTORY

with open("model/training_history.json", "r") as f:
    history = json.load(f)

print("Training history loaded successfully!")


# STEP 5: LOAD TEST DATASET

data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    image_size=(224, 224),
    batch_size=32,
    shuffle=True
)

# Apply VGG16 preprocessing

data = data.map(lambda x, y: (preprocess_input(x), y))

# Split dataset (80% training, 20% testing)

train_size = int(0.8 * len(data))

train = data.take(train_size)
test = data.skip(train_size)

print("Test dataset loaded successfully!")


# STEP 6: MODEL EVALUATION

print("\nEvaluating model...")

loss, accuracy = model.evaluate(test)

print(f"\nModel Loss: {loss:.4f}")
print(f"Model Accuracy: {accuracy * 100:.2f}%")


# STEP 7: LOAD POTATO LEAF IMAGE

image_path = "test_images/late_blight_1.JPG"

img = tf.keras.preprocessing.image.load_img(
    image_path,
    target_size=(224, 224)
)

print("\nImage loaded successfully!")


# STEP 8: CONVERT IMAGE INTO ARRAY

img_array = tf.keras.preprocessing.image.img_to_array(img)


# STEP 9: ADD BATCH DIMENSION

img_array = np.expand_dims(img_array, axis=0)

print("Image converted into array successfully!")


# STEP 10: VGG16 PREPROCESSING

img_array = preprocess_input(img_array)

print("VGG16 preprocessing completed!")


# STEP 11: MAKE PREDICTION

predictions = model.predict(img_array)

print("\nPrediction Probabilities:")

for i, class_name in enumerate(class_names):
    print(f"{class_name}: {predictions[0][i] * 100:.2f}%")


# STEP 12: GET PREDICTED CLASS

predicted_index = np.argmax(predictions[0])

predicted_class = class_names[predicted_index].replace(
    "Potato___", ""
)

confidence = predictions[0][predicted_index] * 100

print(f"\nFinal Prediction: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")


# STEP 13: DISPLAY IMAGE WITH PREDICTION

plt.figure(figsize=(6, 6))

plt.imshow(img)

plt.title(
    f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%"
)

plt.axis("off")

plt.show()


# STEP 14: PREDICTION PROBABILITY GRAPH

plt.figure(figsize=(8, 5))

plt.bar(class_names, predictions[0] * 100)

plt.xlabel("Potato Disease Classes")
plt.ylabel("Prediction Probability (%)")

plt.title("Potato Disease Prediction Probabilities")

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()


# STEP 15: ACCURACY COMPARISON GRAPH

plt.figure(figsize=(8, 5))

plt.plot(history['accuracy'], label='Train Accuracy')

plt.plot(history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.title('Accuracy vs Epochs')

plt.legend()

plt.show()


# STEP 16: LOSS COMPARISON GRAPH

plt.figure(figsize=(8, 5))

plt.plot(history['loss'], label='Train Loss')

plt.plot(history['val_loss'], label='Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.title('Loss vs Epochs')

plt.legend()

plt.show()


print("\nAll tasks completed successfully!")