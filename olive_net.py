import os
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "olive_model.h5")
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
IMG_SIZE = 224
CLASSES = {0: "healthy", 1: "diseased"}


def build_model():
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        # block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # classifier
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(epochs=20, batch_size=32):
    print("Loading processed data...")
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(PROCESSED_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))

    print(f"Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples")

    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")

    return model, history


def load_model():
    return keras.models.load_model(MODEL_PATH)


def predict_image(image_path, model=None):
    if model is None:
        model = load_model()

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0][0]
    label = 1 if prediction >= 0.5 else 0
    confidence = prediction if label == 1 else 1 - prediction

    return CLASSES[label], float(confidence)


def predict_from_array(img_array, model=None):
    if model is None:
        model = load_model()

    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0][0]
    label = 1 if prediction >= 0.5 else 0
    confidence = prediction if label == 1 else 1 - prediction

    return CLASSES[label], float(confidence)


if __name__ == "__main__":
    print("=" * 50)
    print("OliNet — Training")
    print("=" * 50)
    train_model()
