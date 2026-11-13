import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data_paths(base_dir):
  
    filepaths = []
    labels = []
    
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                filepaths.append(os.path.join(folder_path, file))
                labels.append(folder)
                
    df = pd.DataFrame({
        'filepath': filepaths,
        'label': labels
    })
    
    return df

def get_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, val_datagen
