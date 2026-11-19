"""Training 5-fold stratified cross validation.

Pemakaian:
    python train.py              # latih algoritma 1, 2, 3
    python train.py --algos 3    # hanya InceptionV3
Hasil tiap algoritma disimpan ke hasil_algoN.json (dilewati jika sudah ada).
"""
import os
import json
import random
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")  # harus sebelum model dibuat

from preprocessing import (SEED, IMG, BATCH, EPOCHS, N_FOLD, NAMA, class_names,
                           load_dataframe, get_data_generators)
from model import BUILD


def main(algos):
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    print("GPU:", tf.config.list_physical_devices("GPU"))

    df = load_dataframe()
    print(len(df), "gambar,", df["label"].nunique(), "kelas")

    skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
    folds = list(skf.split(df["filepath"], df["label"]))
    common = dict(x_col="filepath", y_col="label", target_size=IMG, batch_size=BATCH,
                  class_mode="categorical", classes=class_names)

    for algo in algos:
        fn = f"hasil_algo{algo}.json"
        if os.path.exists(fn):
            print(f"{NAMA[algo]}: sudah ada ({fn}), dilewati")
            continue

        res = {"hist": [], "fold": [], "y_true": [], "y_pred": []}
        for k, (tr_idx, te_idx) in enumerate(folds, start=1):
            print(f"\n===== {NAMA[algo]} - Fold {k}/{N_FOLD} =====")
            tf.keras.backend.clear_session()
            tr_df, te_df = df.iloc[tr_idx], df.iloc[te_idx]
            tr_df, va_df = train_test_split(tr_df, test_size=0.1,
                                            stratify=tr_df["label"], random_state=SEED)

            train_dg, val_dg = get_data_generators()
            tr_gen = train_dg.flow_from_dataframe(tr_df, shuffle=True, **common)
            va_gen = val_dg.flow_from_dataframe(va_df, shuffle=False, **common)
            te_gen = val_dg.flow_from_dataframe(te_df, shuffle=False, **common)

            model = BUILD[algo](input_shape=IMG + (3,), num_classes=len(class_names))
            cb = [ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
                  EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
            h = model.fit(tr_gen, validation_data=va_gen, epochs=EPOCHS, callbacks=cb, verbose=2)

            ev = model.evaluate(te_gen, return_dict=True, verbose=0)
            y_pred = model.predict(te_gen, verbose=0).argmax(axis=1)
            y_true = te_gen.classes
            p_, r_, f_, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                            zero_division=0)
            row = {"Fold": k, "Loss": float(ev["loss"]),
                   "Accuracy": float(accuracy_score(y_true, y_pred)),
                   "Precision": float(p_), "Recall": float(r_), "F1": float(f_),
                   "AUC": float(ev["auc"])}
            print(row)
            res["fold"].append(row)
            res["hist"].append({kk: [float(x) for x in v] for kk, v in h.history.items()
                                if kk in ("loss", "val_loss", "accuracy", "val_accuracy")})
            res["y_true"] += [int(x) for x in y_true]
            res["y_pred"] += [int(x) for x in y_pred]

        json.dump(res, open(fn, "w"))
    print("\nSelesai.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    main(ap.parse_args().algos)
