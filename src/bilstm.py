import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.callbacks import EarlyStopping



def build_bilstm_model(distance, rounds=25):
    num_sensors = (distance**2 - 1)
    model = models.Sequential([
        layers.Input(shape=(rounds, num_sensors)),
        layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        # Increased Dropout to fight overfitting
        layers.Dropout(0.5), 
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5), 
        layers.Dense(1, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) # Lower LR for stability
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model



# --- 2. THE TRAINING LOOP ---
distances = [3, 5, 7,9,11,13] 

for d in distances:
    csv_file = os.path.join('..', 'data', f'd{d}_matched_dataset.csv')
    print(f"\n{'='*40}")
    print(f"🚀 PROCESSING DISTANCE d={d}")
    print(f"{'='*40}")

    try:
        # A. LOAD DATA
        print(f"📂 Loading {csv_file}...")
        df = pd.read_csv(csv_file, dtype=np.int8)
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.float32)
        del df # Save RAM immediately

        # B. RESHAPE
        num_sensors = (d**2 - 1)
        X_reshaped = X.reshape(-1, 25, num_sensors)
        print(f"✅ Data defined. Shots: {len(y)} | Shape: {X_reshaped.shape}")

        # C. TRAIN AND SAVE
        model = build_bilstm_model(d)
        

        save_path = f'hybrid_model_d{d}.h5'
        checkpoint = ModelCheckpoint(
            save_path, 
            monitor='val_accuracy', 
            save_best_only=True, 
            mode='max', 
            verbose=1
        )

        print(f"🧠 Training Bi-LSTM for d={d}...")
        early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
        model.fit(
            X_reshaped, y,
            epochs=20, 
            batch_size=128, # Adjusted for potentially smaller datasets
            validation_split=0.2,
            callbacks=[checkpoint, early_stop],
            verbose=1
        )
        print(f" BEST MODEL SAVED: {save_path}")

    except FileNotFoundError:
        print(f" Warning: Could not find {csv_file}. Skipping this distance.")
    except Exception as e:
        print(f" Error training d={d}: {e}")

