import os
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

from preprocessing import load_data_paths, get_data_generators
from model import build_algo1_simple_cnn, build_algo2_modified_cnn, build_algo3_inceptionv3

BASE_DIR = "dataset_daun/plantvillage dataset/color"
df = load_data_paths(BASE_DIR)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_dg, val_dg = get_data_generators()
common_args = dict(x_col='filepath', y_col='label', target_size=(128, 128), batch_size=16, class_mode='categorical')

def train_algoritma(algo_id):
    print(f"\n=== MEMULAI TRAINING ALGORITMA {algo_id} ===")
    tf.keras.backend.clear_session()
    fold_no = 1
    
    for train_index, val_index in skf.split(df['filepath'], df['label']):
        print(f"--- Melatih Fold Ke-{fold_no} ---")
        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]

        train_gen = train_dg.flow_from_dataframe(train_df, shuffle=True, **common_args)
        val_gen = val_dg.flow_from_dataframe(val_df, shuffle=False, **common_args)

        if algo_id == 1:
            model = build_algo1_simple_cnn()
        elif algo_id == 2:
            model = build_algo2_modified_cnn()
        else:
            model = build_algo3_inceptionv3()

        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5),
            EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
        ]

        model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks, verbose=1)
        
        if fold_no == 5:
            model.save(f'model_algo{algo_id}_final.keras')
            print(f"Model Algoritma {algo_id} berhasil disimpan!")
            
        fold_no += 1

if __name__ == "__main__":
    train_algoritma(1)
    train_algoritma(2)
    train_algoritma(3)