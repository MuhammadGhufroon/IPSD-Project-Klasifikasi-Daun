import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from preprocessing import load_data_paths, get_data_generators
from model import f1_score # Penting untuk load custom metric

BASE_DIR = "dataset_daun/plantvillage dataset/color"
df = load_data_paths(BASE_DIR)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
_, val_dg = get_data_generators()
common_args = dict(x_col='filepath', y_col='label', target_size=(128, 128), batch_size=16, class_mode='categorical')

fold_no = 1
for train_index, val_index in skf.split(df['filepath'], df['label']):
    if fold_no == 5:
        val_df = df.iloc[val_index]
        test_gen = val_dg.flow_from_dataframe(val_df, shuffle=False, **common_args)
        break
    fold_no += 1

def evaluasi_model(algo_id):
    print(f"\n=== EVALUASI ALGORITMA {algo_id} ===")
    model = load_model(f'model_algo{algo_id}_final.keras', custom_objects={'f1_score': f1_score})
    
    test_gen.reset()
    y_true = test_gen.classes
    y_pred_prob = model.predict(test_gen, steps=len(test_gen), verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    class_names = list(test_gen.class_indices.keys())
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Algoritma {algo_id}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_algo{algo_id}.png', dpi=200)
    print(f"Confusion matrix disimpan sebagai confusion_matrix_algo{algo_id}.png")

if __name__ == "__main__":
    evaluasi_model(1)
    evaluasi_model(2)
    evaluasi_model(3)