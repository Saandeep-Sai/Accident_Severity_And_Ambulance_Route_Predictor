"""
severity_classifier.py — Phase 3: Accident Severity Classifier (CNN)
=====================================================================
Trains a MobileNetV2-based CNN via transfer learning on REAL accident
severity images. Uses two-stage training: frozen base → fine-tuning
top layers for maximum accuracy.

Image dataset (pre-split — used as-is):
  data/images/training/   — 1,383 images (minor/moderate/severe)
  data/images/validation/ — 248 images (minor/moderate/severe)

Authors : Anupama Boya, Atukula Saipriya, Nalla Vishishta (CSE-A)
Guide   : P. Raj Kumar
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
IMG_SIZE = (224, 224)
NUM_CLASSES = 3
CLASSES = ["minor", "moderate", "severe"]
BATCH_SIZE = 32

# Two-stage training
STAGE1_EPOCHS = 8       # Frozen base
STAGE1_LR = 0.001
STAGE2_EPOCHS = 15      # Fine-tuning top layers
STAGE2_LR = 0.00005     # Much lower LR for fine-tuning

FINE_TUNE_FROM = 120    # Unfreeze layers from index 120 onwards (last ~30 layers)

TRAIN_DIR = os.path.join("data", "images", "training")
VAL_DIR = os.path.join("data", "images", "validation")
MODEL_PATH = "severity_model.h5"
HISTORY_PATH = "training_history.png"

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int = NUM_CLASSES) -> Model:
    """
    Build MobileNetV2 classifier.

    Architecture:
        MobileNetV2 (ImageNet, frozen) →
        GlobalAveragePooling2D →
        Dense(256, relu) → Dropout(0.4) →
        Dense(128, relu) → Dropout(0.3) →
        Dense(num_classes, softmax)
    """
    base = MobileNetV2(weights="imagenet", include_top=False,
                       input_shape=(224, 224, 3))
    for layer in base.layers:
        layer.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=output)
    return model, base


def compile_model(model, lr=STAGE1_LR):
    """Compile with Adam optimizer."""
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def unfreeze_top_layers(model, base, from_layer=FINE_TUNE_FROM):
    """Unfreeze layers from 'from_layer' onwards for fine-tuning."""
    for layer in base.layers[from_layer:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    trainable = sum(p.numpy().size for p in model.trainable_weights)
    print(f"[INFO] Fine-tuning: {trainable:,} trainable params "
          f"(layers {from_layer}+ unfrozen)")


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def create_data_generators():
    """Create training (with augmentation) and validation generators."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.15,
        shear_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=CLASSES,
        seed=SEED, shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=CLASSES,
        seed=SEED, shuffle=False,
    )
    print(f"[INFO] Training: {train_gen.samples} | Validation: {val_gen.samples}")
    print(f"[INFO] Classes : {train_gen.class_indices}")
    return train_gen, val_gen


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_two_stage(model, base, train_gen, val_gen):
    """
    Two-stage training:
      Stage 1: Train only the classifier head (frozen base)
      Stage 2: Unfreeze top layers and fine-tune with low LR
    """
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=2, min_lr=1e-7, verbose=1),
    ]

    # --- Stage 1: Frozen base ---
    print("\n" + "=" * 50)
    print("  STAGE 1: Training classifier head (base frozen)")
    print("=" * 50)
    compile_model(model, lr=STAGE1_LR)
    trainable_s1 = sum(p.numpy().size for p in model.trainable_weights)
    print(f"[INFO] Trainable params: {trainable_s1:,}")

    history1 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=STAGE1_EPOCHS, callbacks=callbacks, verbose=1,
    )

    # --- Stage 2: Fine-tune top layers ---
    print("\n" + "=" * 50)
    print("  STAGE 2: Fine-tuning top MobileNetV2 layers")
    print("=" * 50)
    unfreeze_top_layers(model, base)
    compile_model(model, lr=STAGE2_LR)

    history2 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=STAGE2_EPOCHS, callbacks=callbacks, verbose=1,
    )

    # Merge histories
    history = {}
    for key in history1.history:
        history[key] = history1.history[key] + history2.history[key]

    return history


def plot_training_history(history: dict, save_path: str = HISTORY_PATH):
    """Plot training curves with stage boundary."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["accuracy"]) + 1)
    stage1_end = STAGE1_EPOCHS

    # Accuracy
    ax1.plot(epochs, history["accuracy"], "o-", label="Train Accuracy",
             linewidth=2, markersize=3, color="#2196F3")
    ax1.plot(epochs, history["val_accuracy"], "s-", label="Val Accuracy",
             linewidth=2, markersize=3, color="#FF9800")
    ax1.axvline(x=stage1_end, color="gray", linestyle="--", alpha=0.5,
                label="Fine-tuning starts")
    ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Loss
    ax2.plot(epochs, history["loss"], "o-", label="Train Loss",
             linewidth=2, markersize=3, color="#2196F3")
    ax2.plot(epochs, history["val_loss"], "s-", label="Val Loss",
             linewidth=2, markersize=3, color="#FF9800")
    ax2.axvline(x=stage1_end, color="gray", linestyle="--", alpha=0.5,
                label="Fine-tuning starts")
    ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[✓] Training history saved → {save_path}")


# ---------------------------------------------------------------------------
# Inference API
# ---------------------------------------------------------------------------

_severity_model = None


def _ensure_model_loaded():
    global _severity_model
    if _severity_model is None:
        _severity_model = tf.keras.models.load_model(MODEL_PATH)


def predict_severity(image_path: str) -> dict:
    """
    Predict accident severity from a single image file path.

    Returns
    -------
    dict : {"severity": str, "confidence": float, "probabilities": dict}
    """
    _ensure_model_loaded()
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        return {"severity": "unknown", "confidence": 0.0, "probabilities": {}}

    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = _severity_model.predict(img, verbose=0)[0]
    idx = int(np.argmax(preds))

    return {
        "severity": CLASSES[idx],
        "confidence": float(round(preds[idx], 4)),
        "probabilities": {CLASSES[i]: float(round(preds[i], 4)) for i in range(len(CLASSES))},
    }


def predict_severity_from_bytes(image_bytes: bytes) -> dict:
    """
    Predict accident severity from raw image bytes (e.g. from a
    Streamlit file_uploader). No disk write needed.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of a JPEG/PNG image.

    Returns
    -------
    dict : {"severity": str, "confidence": float, "probabilities": dict}
    """
    _ensure_model_loaded()
    import cv2

    # Decode bytes → numpy array
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"severity": "unknown", "confidence": 0.0, "probabilities": {}}

    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = _severity_model.predict(img, verbose=0)[0]
    idx = int(np.argmax(preds))

    return {
        "severity": CLASSES[idx],
        "confidence": float(round(preds[idx], 4)),
        "probabilities": {CLASSES[i]: float(round(preds[i], 4)) for i in range(len(CLASSES))},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[Phase 3] Severity Classifier — CNN (REAL IMAGES)")
    print("=" * 55)

    for d in [TRAIN_DIR, VAL_DIR]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Not found: {d}")

    # Build
    model, base = build_model()

    # Data
    train_gen, val_gen = create_data_generators()

    # Two-stage training
    history = train_two_stage(model, base, train_gen, val_gen)

    # Save
    model.save(MODEL_PATH)
    print(f"[✓] Model saved → {MODEL_PATH}")
    plot_training_history(history)

    # Final validation accuracy
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"\n  Final Val Accuracy : {val_acc*100:.1f}%")
    print(f"  Final Val Loss     : {val_loss:.4f}")

    # Smoke test
    test_dir = os.path.join(VAL_DIR, "severe")
    if os.path.isdir(test_dir):
        files = [f for f in os.listdir(test_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            result = predict_severity(os.path.join(test_dir, files[0]))
            print(f"\n  Smoke test: {files[0]} → {result}")

    print("\nSeverity classifier ready.")
