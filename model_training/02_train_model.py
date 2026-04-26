"""
02_train_model.py
-----------------
Trains a MobileNetV2-based classifier on the organized HAM10000 data.

Usage:
    python model_training/02_train_model.py

Prerequisites:
    Run 01_organize_data.py first so organized_data/ is populated.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

# ---------------------------------------------
# Configuration
# ---------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent.parent
ORGANIZED_DIR = BASE_DIR / "organized_data"
MODEL_DIR     = Path(__file__).resolve().parent          # model_training/
MODEL_PATH    = MODEL_DIR / "saved_model.h5"
HISTORY_PATH  = MODEL_DIR / "training_history.png"
CONFMAT_PATH  = MODEL_DIR / "confusion_matrix.png"

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 0.0001
NUM_CLASSES = 7

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# ---------------------------------------------
# 1. Data Generators
# ---------------------------------------------
def build_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        str(ORGANIZED_DIR / "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        str(ORGANIZED_DIR / "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    print(f"[OK] Train samples : {train_gen.samples}  |  Classes: {train_gen.class_indices}")
    print(f"[OK] Val   samples : {val_gen.samples}")
    return train_gen, val_gen

# ---------------------------------------------
# 2. Model Architecture - MobileNetV2 + Custom Head
# ---------------------------------------------
def build_model() -> Model:
    base = MobileNetV2(
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
        weights="imagenet",
    )
    # Freeze all base layers
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs, outputs, name="DermoScope_MobileNetV2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model

# ---------------------------------------------
# 3. Callbacks
# ---------------------------------------------
def build_callbacks():
    return [
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

# ---------------------------------------------
# 4. Training
# ---------------------------------------------
def train_model(model: Model, train_gen, val_gen):
    print("\n[...] Computing class weights ...")
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"[OK] Class weights: {class_weight_dict}\n")

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=build_callbacks(),
        class_weight=class_weight_dict,
        verbose=1,
    )
    return history

# ---------------------------------------------
# 5. Plot Training History
# ---------------------------------------------
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dermo-Scope - Training History", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy", color="#4CAF50", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy",   color="#F44336", linewidth=2)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss", color="#2196F3", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss",   color="#FF9800", linewidth=2)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(HISTORY_PATH), dpi=150)
    plt.close()
    print(f"[OK] Training history saved -> {HISTORY_PATH}")

# ---------------------------------------------
# 6. Evaluation - Confusion Matrix & Report
# ---------------------------------------------
def evaluate_model(model: Model, val_gen):
    print("\n[...] Evaluating on validation set ...")
    val_gen.reset()
    preds      = model.predict(val_gen, verbose=1)
    y_pred     = np.argmax(preds, axis=1)
    y_true     = val_gen.classes
    label_map  = {v: k for k, v in val_gen.class_indices.items()}
    class_list = [label_map[i] for i in range(NUM_CLASSES)]

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_list)
    print("\n-- Classification Report --")
    print(report)

    # Save report to file
    report_path = MODEL_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Dermo-Scope Classification Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(report)
    print(f"[OK] Classification report saved -> {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # The fig variable returned by subplots is intentionally ignored via _
    _, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_list,
        yticklabels=class_list,
        ax=ax,
    )
    ax.set_title("Confusion Matrix - Dermo-Scope", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(str(CONFMAT_PATH), dpi=150)
    plt.close()
    print(f"[OK] Confusion matrix saved -> {CONFMAT_PATH}")

# ---------------------------------------------
# Main
# ---------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print(" Dermo-Scope - Model Training (MobileNetV2)")
    print("=" * 60)

    # Sanity check
    if not (ORGANIZED_DIR / "train").exists():
        raise RuntimeError(
            "organized_data/train not found. "
            "Please run data_tools/01_organize_data.py first."
        )

    train_gen, val_gen = build_generators()
    model = build_model()
    history = train_model(model, train_gen, val_gen)

    # Re-load best checkpoint for evaluation
    print(f"\n[...] Loading best model from {MODEL_PATH}")
    best_model = tf.keras.models.load_model(str(MODEL_PATH))

    plot_history(history)
    evaluate_model(best_model, val_gen)

    val_loss, val_acc = best_model.evaluate(val_gen, verbose=0)
    print(f"\n[OK] Final Val  Loss     : {val_loss:.4f}")
    print(f"[OK] Final Val  Accuracy : {val_acc * 100:.2f}%")
    print(f"\n[OK] Model saved to -> {MODEL_PATH}")
    print("[OK] Training complete!")
