import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")
IMG_SIZE = 224
CLASSES = {"healthy": 0, "diseased": 1}


def load_and_preprocess_images(dataset_dir=DATASET_DIR):
    images = []
    labels = []

    for class_name, label in CLASSES.items():
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Directory not found: {class_dir}")
            continue

        files = os.listdir(class_dir)
        print(f"Loading {len(files)} images from '{class_name}'...")

        for filename in files:
            filepath = os.path.join(class_dir, filename)
            img = cv2.imread(filepath)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)


def split_dataset(images, labels, test_size=0.15, val_size=0.15, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, random_state=random_state, stratify=y_train
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def save_processed_data(train, val, test, output_dir=PROCESSED_DIR):
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_train.npy"), train[0])
    np.save(os.path.join(output_dir, "y_train.npy"), train[1])
    np.save(os.path.join(output_dir, "X_val.npy"), val[0])
    np.save(os.path.join(output_dir, "y_val.npy"), val[1])
    np.save(os.path.join(output_dir, "X_test.npy"), test[0])
    np.save(os.path.join(output_dir, "y_test.npy"), test[1])

    print(f"Saved processed data to {output_dir}")
    print(f"  Train: {train[0].shape[0]} samples")
    print(f"  Val:   {val[0].shape[0]} samples")
    print(f"  Test:  {test[0].shape[0]} samples")


def load_processed_data(processed_dir=PROCESSED_DIR):
    train = (
        np.load(os.path.join(processed_dir, "X_train.npy")),
        np.load(os.path.join(processed_dir, "y_train.npy")),
    )
    val = (
        np.load(os.path.join(processed_dir, "X_val.npy")),
        np.load(os.path.join(processed_dir, "y_val.npy")),
    )
    test = (
        np.load(os.path.join(processed_dir, "X_test.npy")),
        np.load(os.path.join(processed_dir, "y_test.npy")),
    )
    return train, val, test


def run_pipeline():
    print("=" * 50)
    print("OliVidi Data Pipeline")
    print("=" * 50)

    images, labels = load_and_preprocess_images()
    if len(images) == 0:
        print("No images found. Place images in dataset/healthy/ and dataset/diseased/")
        return

    print(f"\nTotal images loaded: {len(images)}")
    print(f"  Healthy:  {np.sum(labels == 0)}")
    print(f"  Diseased: {np.sum(labels == 1)}")

    train, val, test = split_dataset(images, labels)
    save_processed_data(train, val, test)
    print("\nPipeline complete!")


if __name__ == "__main__":
    run_pipeline()
