"""
Train a CNN (Convolutional Neural Network) on MNIST dataset.
Run this once: python train_model.py
Creates model.keras in the same directory.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

print("Fetching MNIST dataset...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist.data.astype("float32"), mnist.target.astype(int)

# Normalize to 0–1
X = X / 255.0

# Reshape to (samples, 28, 28, 1) — CNN expects image shape
X = X.reshape(-1, 28, 28, 1)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

print(f"Training samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ── Build CNN ──────────────────────────────────────────────────
model = keras.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Classifier
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),  # 10 digits
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ── Data Augmentation — makes model robust to different styles ──
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,       # slight rotation
    zoom_range=0.10,         # slight zoom
    width_shift_range=0.10,  # slight horizontal shift
    height_shift_range=0.10, # slight vertical shift
)
datagen.fit(X_train)

# ── Train ──────────────────────────────────────────────────────
print("\nTraining CNN (5–10 minutes)...")
callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", patience=2,
                                       factor=0.5, verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                   restore_best_weights=True, verbose=1),
]

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=128),
    epochs=30,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1,
)

# ── Evaluate ───────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

# ── Save ───────────────────────────────────────────────────────
model.save("model.keras")
print("✅ Model saved as model.keras")
print("You can now run: python app.py")