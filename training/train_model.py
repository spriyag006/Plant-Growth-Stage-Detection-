import os
import json

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# ---------------------------------------
# Configuration
# ---------------------------------------

DATASET_DIR = "dataset_split"

TRAIN_DIR = os.path.join(
    DATASET_DIR,
    "train"
)

VALIDATION_DIR = os.path.join(
    DATASET_DIR,
    "validation"
)

MODEL_DIR = "backend/model"

os.makedirs(MODEL_DIR, exist_ok=True)


IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 15


# ---------------------------------------
# Load Training Dataset
# ---------------------------------------

train_dataset = tf.keras.utils.image_dataset_from_directory(

    TRAIN_DIR,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=True
)


# ---------------------------------------
# Load Validation Dataset
# ---------------------------------------

validation_dataset = tf.keras.utils.image_dataset_from_directory(

    VALIDATION_DIR,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False
)


# ---------------------------------------
# Get Class Names
# ---------------------------------------

class_names = train_dataset.class_names

print("\nClasses:")

for index, name in enumerate(class_names):

    print(index, ":", name)


# Save class names

with open(
    os.path.join(
        MODEL_DIR,
        "class_names.json"
    ),
    "w"
) as file:

    json.dump(
        class_names,
        file
    )


# ---------------------------------------
# Optimize Dataset Pipeline
# ---------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    AUTOTUNE
)


# ---------------------------------------
# Data Augmentation
# ---------------------------------------

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.1
    ),

    layers.RandomZoom(
        0.1
    ),

    layers.RandomContrast(
        0.1
    )

])


# ---------------------------------------
# MobileNetV2 Base Model
# ---------------------------------------

base_model = MobileNetV2(

    input_shape=(
        224,
        224,
        3
    ),

    include_top=False,

    weights="imagenet"
)


# Freeze pretrained layers

base_model.trainable = False


# ---------------------------------------
# Build AI Model
# ---------------------------------------

inputs = layers.Input(
    shape=(224, 224, 3)
)


x = data_augmentation(inputs)


x = preprocess_input(x)


x = base_model(
    x,
    training=False
)


x = layers.GlobalAveragePooling2D()(x)


x = layers.Dense(
    128,
    activation="relu"
)(x)


x = layers.Dropout(
    0.4
)(x)


outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)


model = models.Model(
    inputs,
    outputs
)


# ---------------------------------------
# Compile
# ---------------------------------------

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ---------------------------------------
# Display Model
# ---------------------------------------

model.summary()


# ---------------------------------------
# Callbacks
# ---------------------------------------

model_path = os.path.join(
    MODEL_DIR,
    "plant_growth_model.keras"
)


checkpoint = ModelCheckpoint(

    model_path,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1
)


early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=4,

    restore_best_weights=True
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.2,

    patience=2,

    min_lr=0.000001,

    verbose=1
)


# ---------------------------------------
# Train Model
# ---------------------------------------

print("\nStarting training...\n")


history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)
model.save(model_path)

print("\nTraining completed!")

print(
    "Model saved at:",
    model_path
)