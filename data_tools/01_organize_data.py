"""
01_organize_data.py
-------------------
Organizes the HAM10000 dataset into class-specific folders for use with
Keras ImageDataGenerator.flow_from_directory().

Works with the HAM10000.zip structure which contains:
    HAM10000_images_part_1/ISIC_xxxx.jpg
    ham10000_images_part_2/ISIC_xxxx.jpg
    HAM10000_metadata.csv  (in root of zip)

Usage:
    python data_tools/01_organize_data.py
"""

import shutil
import pandas as pd
from pathlib import Path

# ---------------------------------------------
# Configuration
# ---------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent.parent    # major_project/
RAW_DATA_DIR  = BASE_DIR / "raw_data"
ORGANIZED_DIR = BASE_DIR / "organized_data"

# 7 target classes
CLASS_LABELS = ["nv", "mel", "bcc", "akiec", "bkl", "df", "vasc"]

# Train / Validation split ratio
TRAIN_RATIO = 0.8

# ---------------------------------------------
# Step 1 - Validate inputs (recursive search)
# ---------------------------------------------
def validate_inputs():
    # Locate metadata CSV anywhere inside raw_data/
    csv_candidates = list(RAW_DATA_DIR.rglob("HAM10000_metadata.csv"))
    if not csv_candidates:
        raise FileNotFoundError(
            f"HAM10000_metadata.csv not found anywhere in {RAW_DATA_DIR}\n"
            "Please extract HAM10000.zip into raw_data/ first."
        )
    metadata_file = csv_candidates[0]

    # Locate images recursively (handles part_1 / part_2 sub-folders)
    images_found = list(RAW_DATA_DIR.rglob("*.jpg"))
    if not images_found:
        raise FileNotFoundError(
            f"No .jpg images found (recursively) in {RAW_DATA_DIR}"
        )
    print(f"[OK] Metadata : {metadata_file}")
    print(f"[OK] Images   : {len(images_found)} found (searched recursively)")
    return metadata_file, images_found

# ---------------------------------------------
# Step 2 - Create folder structure
# ---------------------------------------------
def create_directory_structure():
    for split in ["train", "val"]:
        for label in CLASS_LABELS:
            folder = ORGANIZED_DIR / split / label
            folder.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Directory structure created under: {ORGANIZED_DIR}")

# ---------------------------------------------
# Step 3 - Read metadata and build image→class map
# ---------------------------------------------
def load_metadata(metadata_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    # Ensure required columns exist
    required_cols = {"image_id", "dx"}
    bad_cols = required_cols - set(df.columns)
    if bad_cols:
        raise ValueError(f"Missing columns in metadata CSV: {bad_cols}")
    df = df[["image_id", "dx"]].dropna()
    df["dx"] = df["dx"].str.strip().str.lower()
    # Filter to known classes
    df = df[df["dx"].isin(CLASS_LABELS)]
    print(f"[OK] Loaded {len(df)} valid records from metadata.")
    return df

# ---------------------------------------------
# Step 4 - Copy images to class folders
# ---------------------------------------------
def organize_images(df: pd.DataFrame, all_image_paths: list):
    from sklearn.model_selection import train_test_split

    # Build a fast lookup: image_id (stem) → full Path
    id_to_path = {p.stem: p for p in all_image_paths}

    train_df, val_df = train_test_split(
        df, test_size=1 - TRAIN_RATIO, random_state=42, stratify=df["dx"]
    )

    copied = 0
    not_found = 0

    def copy_to_split(subset_df: pd.DataFrame, split: str):
        nonlocal copied, not_found
        for _, row in subset_df.iterrows():
            image_id = row["image_id"]
            label = row["dx"]
            src = id_to_path.get(image_id)
            dst = ORGANIZED_DIR / split / label / f"{image_id}.jpg"
            if src and src.exists():
                shutil.copy2(src, dst)
                copied += 1
            else:
                not_found += 1

    print("[...] Copying training images ...")
    copy_to_split(train_df, "train")
    print("[...] Copying validation images ...")
    copy_to_split(val_df, "val")

    print(f"\n[OK] Copied : {copied} images")
    if not_found:
        print(f"[WARN] Not found: {not_found} image IDs had no matching file")

    # Print class distribution
    print("\n-- Training Set Distribution --")
    for cls in CLASS_LABELS:
        count = len(list((ORGANIZED_DIR / "train" / cls).glob("*.jpg")))
        print(f"    {cls}: {count} images")

    print("\n-- Validation Set Distribution --")
    for cls in CLASS_LABELS:
        count = len(list((ORGANIZED_DIR / "val" / cls).glob("*.jpg")))
        print(f"    {cls}: {count} images")

# ---------------------------------------------
# Step 5 - Verify ImageDataGenerator compatibility
# ---------------------------------------------
def verify_with_keras():
    try:
        import tensorflow as tf
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
        )
        train_gen = datagen.flow_from_directory(
            str(ORGANIZED_DIR / "train"),
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical",
        )
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
        val_gen = val_datagen.flow_from_directory(
            str(ORGANIZED_DIR / "val"),
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical",
        )
        print(f"\n[OK] Keras train generator → {train_gen.samples} samples, classes: {train_gen.class_indices}")
        print(f"[OK] Keras val   generator → {val_gen.samples} samples")
    except ImportError:
        print("[WARN] TensorFlow not installed - skipping Keras validation step.")

# ---------------------------------------------
# Main
# ---------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print(" HAM10000 Data Organizer - Dermo-Scope")
    print("=" * 60)
    metadata_file, all_images = validate_inputs()
    create_directory_structure()
    df = load_metadata(metadata_file)
    organize_images(df, all_images)
    verify_with_keras()
    print("\n[OK] Organization complete! You can now run 02_train_model.py")
