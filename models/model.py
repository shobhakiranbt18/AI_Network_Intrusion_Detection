"""
models/model.py
---------------
Defines the Deep Neural Network (DNN) architecture using TensorFlow/Keras.

Binary classifier:
Normal (0) vs Attack (1)
"""

import os
import tensorflow as tf

from keras import Sequential
from keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Input,
)
from keras.regularizers import l2


def build_dnn(input_dim: int,
              learning_rate: float = 0.0005) -> tf.keras.Model:
    """
    Build and compile the Deep Neural Network for Binary Classification.
    
    Architecture optimized for 95%+ accuracy on UNSW-NB15 dataset.
    Binary Classification: Normal (0) vs Attack (1)
    """

    model = Sequential(name="Binary_DNN_Intrusion_Detector")

    # Input layer
    model.add(
        Input(
            shape=(input_dim,),
            name="input"
        )
    )

    # Hidden Layer 1 - Increased capacity
    model.add(
        Dense(
            512,  # Increased from 256
            activation="relu",
            kernel_regularizer=l2(1e-4),
            name="dense_1"
        )
    )
    model.add(BatchNormalization(name="bn_1"))
    model.add(Dropout(0.3, name="drop_1"))  # Reduced from 0.4

    # Hidden Layer 2 - Increased capacity
    model.add(
        Dense(
            256,  # Increased from 128
            activation="relu",
            kernel_regularizer=l2(1e-4),
            name="dense_2"
        )
    )
    model.add(BatchNormalization(name="bn_2"))
    model.add(Dropout(0.25, name="drop_2"))  # Reduced from 0.3

    # Hidden Layer 3 - Increased capacity
    model.add(
        Dense(
            128,  # Increased from 64
            activation="relu",
            kernel_regularizer=l2(1e-4),
            name="dense_3"
        )
    )
    model.add(BatchNormalization(name="bn_3"))
    model.add(Dropout(0.25, name="drop_3"))  # Reduced from 0.3

    # Hidden Layer 4 - Increased capacity
    model.add(
        Dense(
            64,  # Increased from 32
            activation="relu",
            kernel_regularizer=l2(1e-4),
            name="dense_4"
        )
    )
    model.add(BatchNormalization(name="bn_4"))
    model.add(Dropout(0.2, name="drop_4"))

    # Hidden Layer 5 - Additional layer for better learning
    model.add(
        Dense(
            32,
            activation="relu",
            kernel_regularizer=l2(1e-4),
            name="dense_5"
        )
    )

    # Output layer - Binary Classification
    model.add(
        Dense(
            1,
            activation="sigmoid",
            name="binary_output"
        )
    )

    # Compile
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model



def get_callbacks(patience: int = 10) -> list:
    """
    Training callbacks:
    - Early stopping
    - Reduce learning rate
    - Save best model weights
    """

    checkpoint_path = os.path.join(
        os.path.dirname(__file__),
        "best.weights.h5"     # <-- Keras 3 compatible
    )


    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=0,
    )


    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=0,
    )


    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=0,
    )


    return [
        early_stop,
        reduce_lr,
        checkpoint
    ]