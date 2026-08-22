import os

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ---------------------------------------
# Paths
# ---------------------------------------

TEST_DIR = "dataset_split/test"

MODEL_PATH = (
    "backend/model/"
    "plant_growth_model.keras"
)


IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32


# ---------------------------------------
# Load Model
# ---------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH
)


print("Model loaded successfully!")


# ---------------------------------------
# Load Test Dataset
# ---------------------------------------

test_dataset = tf.keras.utils.image_dataset_from_directory(

    TEST_DIR,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False
)


class_names = test_dataset.class_names


print("\nTest Classes:")

for index, name in enumerate(class_names):

    print(index, ":", name)


# ---------------------------------------
# Evaluate
# ---------------------------------------

loss, accuracy = model.evaluate(
    test_dataset
)


print("\nTest Loss:", loss)

print(
    "Test Accuracy:",
    accuracy * 100,
    "%"
)


# ---------------------------------------
# Predictions
# ---------------------------------------

y_true = []

y_pred = []


for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )


# ---------------------------------------
# Classification Report
# ---------------------------------------

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)


# ---------------------------------------
# Confusion Matrix
# ---------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)


disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=class_names
)


disp.plot(
    xticks_rotation=45
)

plt.title(
    "Plant Growth Stage Detection - Confusion Matrix"
)

plt.tight_layout()

plt.show()