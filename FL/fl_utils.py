import tensorflow as tf
import pandas as pd
import numpy as np
import os
import collections

# Configuration
NUM_CLASSES = 7
NUM_FEATURES = 59
BATCH_SIZE = 32
SHUFFLE_BUFFER = 1000
PREFETCH_BUFFER = 10

def load_client_dataset(file_path):
    """
    Loads a single client CSV and returns a tf.data.Dataset
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Client file not found: {file_path}")
    
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Split Features and Label
    features = df.drop('label', axis=1).values.astype(np.float32)
    labels = df['label'].values.astype(np.int32)
    
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    return dataset.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).prefetch(PREFETCH_BUFFER)

def create_keras_model():
    """
    Defines the Keras ANN model with Dropout.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(NUM_FEATURES,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

