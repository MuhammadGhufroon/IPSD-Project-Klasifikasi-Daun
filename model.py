"""Arsitektur tiga algoritma: Simple CNN, Modified CNN, dan InceptionV3."""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3
import tensorflow.keras.backend as K


def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    return 2 * (precision * recall) / (precision + recall + K.epsilon())


def compile_and_return(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy",
                           tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall"),
                           tf.keras.metrics.AUC(name="auc"),
                           f1_score])
    return model


# ALGORITMA 1: Simple CNN
def build_algo1_simple_cnn(input_shape=(128, 128, 3), num_classes=6):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax", dtype="float32"),
    ])
    return compile_and_return(model)


# ALGORITMA 2: Modified CNN (BatchNorm + GAP + Dropout)
def build_algo2_modified_cnn(input_shape=(128, 128, 3), num_classes=6):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax", dtype="float32"),
    ])
    return compile_and_return(model)


# ALGORITMA 3: Transfer learning InceptionV3 (bobot ImageNet dibekukan)
def build_algo3_inceptionv3(input_shape=(128, 128, 3), num_classes=6):
    base = InceptionV3(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(scale=2.0, offset=-1.0)(inputs)  # [0,1] -> [-1,1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = models.Model(inputs, outputs)
    return compile_and_return(model)


BUILD = {1: build_algo1_simple_cnn, 2: build_algo2_modified_cnn, 3: build_algo3_inceptionv3}
