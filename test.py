"""Evaluasi dan pembuatan gambar/tabel dari hasil_algoN.json (jalankan setelah train.py).

Pemakaian:  python test.py
Keluaran : fig1_distribusi.png, fig2_preprocessing.png, fig3_kurva_algoN.png,
           fig4_perbandingan.png, confusion_matrix_algoN.png, fig6_f1_algoN.png,
           hasil_per_fold.csv, hasil_rata_rata.csv
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

from preprocessing import (BASE_DIR, IMG, NAMA, class_names, short,
                           load_dataframe, get_data_generators)

METRIK = ["Loss", "Accuracy", "Precision", "Recall", "F1", "AUC"]


def plot_distribusi(df):  # Soal 1
    counts = df["label"].value_counts().reindex(class_names)
    pct = counts / counts.sum() * 100
    plt.figure(figsize=(9, 4.5))
    ax = sns.barplot(x=short, y=counts.values, hue=short, palette="viridis", legend=False)
    for i, (c, p_) in enumerate(zip(counts.values, pct.values)):
        ax.text(i, c + counts.max() * 0.01, f"{c}\n({p_:.1f}%)", ha="center", fontsize=9)
    ax.set_title(f"Distribusi Jumlah Gambar per Kelas (total {counts.sum()} gambar)")
    ax.set_ylabel("Jumlah gambar"); ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig("fig1_distribusi.png", dpi=150); plt.close()
    print("Kelas terbanyak :", short[int(np.argmax(counts.values))], counts.max())
    print("Kelas tersedikit:", short[int(np.argmin(counts.values))], counts.min())
    print("Rasio terbanyak : tersedikit =", round(counts.max() / counts.min(), 2), ": 1")


def plot_augmentasi(df):  # Soal 2
    train_dg, _ = get_data_generators()
    path = df.sample(1, random_state=3)["filepath"].iloc[0]
    img = tf.keras.utils.img_to_array(tf.keras.utils.load_img(path, target_size=IMG)) / 255.0
    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    axes[0].imshow(img); axes[0].set_title("Asli (128x128, rescale 1/255)", fontsize=8)
    for i in range(1, 5):
        axes[i].imshow(np.clip(train_dg.random_transform(img), 0, 1))
        axes[i].set_title(f"Augmentasi {i}", fontsize=8)
    for a in axes:
        a.axis("off")
    plt.tight_layout(); plt.savefig("fig2_preprocessing.png", dpi=150); plt.close()


def plot_curves(algo, result):  # Soal 5
    H = result[algo]["hist"]
    fig, axes = plt.subplots(len(H), 2, figsize=(11, 3.1 * len(H)))
    for r, h in enumerate(H):
        ep = range(1, len(h["loss"]) + 1)
        axes[r, 0].plot(ep, h["accuracy"], "o-", label="train")
        axes[r, 0].plot(ep, h["val_accuracy"], "s-", label="val")
        axes[r, 0].set_title(f"{NAMA[algo]} - Fold {r+1} - Accuracy")
        axes[r, 1].plot(ep, h["loss"], "o-", label="train")
        axes[r, 1].plot(ep, h["val_loss"], "s-", label="val")
        axes[r, 1].set_title(f"{NAMA[algo]} - Fold {r+1} - Loss")
        for c in (0, 1):
            axes[r, c].legend(); axes[r, c].set_xlabel("Epoch")
    plt.tight_layout(); plt.savefig(f"fig3_kurva_algo{algo}.png", dpi=130); plt.close()


def analisis_kelas(algo, result, top_n=5):  # Soal 3 dan 4
    yt, yp = np.array(result[algo]["y_true"]), np.array(result[algo]["y_pred"])
    cm = confusion_matrix(yt, yp, labels=range(len(short)))
    cmn = cm / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=short, yticklabels=short, ax=axes[0])
    axes[0].set_title(f"Confusion Matrix (jumlah) - {NAMA[algo]}")
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=short, yticklabels=short, ax=axes[1])
    axes[1].set_title("Dinormalisasi per kelas asli")
    for a_ in axes:
        a_.set_xlabel("Prediksi"); a_.set_ylabel("Kelas asli"); a_.tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.savefig(f"confusion_matrix_algo{algo}.png", dpi=150); plt.close()

    rep = pd.DataFrame(classification_report(yt, yp, target_names=short,
                                             output_dict=True, zero_division=0)).T
    print(f"\n--- Classification report: {NAMA[algo]} ---\n{rep.round(4)}")
    f1c = rep.loc[short, "f1-score"]
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(x=short, y=f1c.values, hue=short, palette="mako", legend=False)
    for i, v in enumerate(f1c.values):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    plt.ylim(max(0, f1c.min() - 0.1), 1.02); plt.xticks(rotation=20, ha="right")
    plt.title(f"F1-score per Kelas - {NAMA[algo]}"); plt.tight_layout()
    plt.savefig(f"fig6_f1_algo{algo}.png", dpi=150); plt.close()

    pairs = [(short[i], short[j], int(cm[i, j]))
             for i in range(len(short)) for j in range(len(short)) if i != j and cm[i, j] > 0]
    pairs = sorted(pairs, key=lambda x: -x[2])[:top_n]
    print(f"Kesalahan terbanyak {NAMA[algo]} (asli -> prediksi):")
    for a_, b_, n in pairs:
        print(f"  {a_} -> {b_}: {n} gambar")
    print("Total salah:", int((yt != yp).sum()), "dari", len(yt))
    return f1c, pairs


def main():
    if os.path.isdir(BASE_DIR):  # Soal 1 & 2 butuh dataset
        df = load_dataframe()
        plot_distribusi(df); plot_augmentasi(df)
    else:
        print(f"Dataset '{BASE_DIR}' tidak ditemukan, grafik distribusi & augmentasi dilewati.")

    result = {}
    for a in (1, 2, 3):
        fn = f"hasil_algo{a}.json"
        if os.path.exists(fn):
            result[a] = json.load(open(fn))
    if not result:
        raise SystemExit("Tidak ada hasil_algoN.json. Jalankan train.py dulu.")

    for a in result:
        plot_curves(a, result)

    tab = pd.DataFrame([{"Algoritma": NAMA[a], **d} for a in result for d in result[a]["fold"]])
    tab.to_csv("hasil_per_fold.csv", index=False)
    for a in result:
        print(f"\n=== {NAMA[a]}: hasil per fold (data uji) ===")
        print(tab[tab.Algoritma == NAMA[a]].set_index("Fold")[METRIK].round(4))

    urut = [NAMA[a] for a in result]
    mean_ = tab.groupby("Algoritma")[METRIK].mean().loc[urut]
    std_ = tab.groupby("Algoritma")[METRIK].std().loc[urut]
    print("\n=== Rata-rata ± std 5 fold ===")
    print(mean_.round(4).astype(str) + " ± " + std_.round(4).astype(str))
    mean_.to_csv("hasil_rata_rata.csv")

    m = ["Accuracy", "Precision", "Recall", "F1"]
    mean_[m].plot(kind="bar", yerr=std_[m], capsize=3, figsize=(9, 4.5), rot=0, colormap="viridis")
    plt.ylim(max(0, mean_[m].min().min() - 0.1), 1.0)
    plt.title("Perbandingan Rata-rata 5-Fold (garis error = std antar fold)")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig("fig4_perbandingan.png", dpi=150); plt.close()

    for a in result:
        f1c, pairs = analisis_kelas(a, result)
        H = result[a]["hist"]
        gap = np.mean([h["accuracy"][-1] - h["val_accuracy"][-1] for h in H])
        sub = tab[tab.Algoritma == NAMA[a]]
        print(f"\n=== Ringkasan {NAMA[a]} ===")
        print(f"Accuracy {sub.Accuracy.mean():.4f} ± {sub.Accuracy.std():.4f} | F1 makro {sub.F1.mean():.4f}")
        print(f"Kelas terbaik {f1c.idxmax()} ({f1c.max():.3f}); terburuk {f1c.idxmin()} ({f1c.min():.3f})")
        print(f"Selisih akurasi train-val (epoch akhir): {gap:+.4f}")


if __name__ == "__main__":
    main()
