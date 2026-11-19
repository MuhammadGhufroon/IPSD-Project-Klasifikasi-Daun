"""Konfigurasi, pemuatan data, dan preprocessing (resize, rescale, augmentasi)."""
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
BASE_DIR = "dataset_daun/plantvillage dataset/color"
KELAS = ["Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
         "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___healthy"]
IMG, BATCH, EPOCHS, N_FOLD = (128, 128), 32, 10, 5
NAMA = {1: "Simple CNN", 2: "Modified CNN", 3: "InceptionV3"}

class_names = sorted(KELAS)


def bersih(n):
    return n.replace("Tomato___", "").replace("_", " ")


short = [bersih(c) for c in class_names]


def load_data_paths(base_dir):
    filepaths, labels = [], []
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                filepaths.append(os.path.join(folder_path, file))
                labels.append(folder)
    return pd.DataFrame({"filepath": filepaths, "label": labels})


def load_dataframe(base_dir=BASE_DIR):
    """DataFrame (filepath, label) khusus 6 kelas tomat, urut tetap agar fold reproducible."""
    df = load_data_paths(base_dir)
    return df[df["label"].isin(KELAS)].sort_values("filepath").reset_index(drop=True)


def get_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255, rotation_range=20, width_shift_range=0.2,
        height_shift_range=0.2, horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1. / 255)  # validasi/uji: hanya rescale
    return train_datagen, val_datagen
