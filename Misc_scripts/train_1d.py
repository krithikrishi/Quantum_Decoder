import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import os

# --- 1. THE HYBRID ARCHITECTURE (MAM-APPROVED) ---
def build_turbo_model(distance, rounds=25):
    num_sensors = (distance**2 - 1)
    model = models.Sequential([
        layers.Input(shape=(rounds, num_sensors)),
        # No Conv1D, just a fast GRU
        layers.GRU(32, return_sequences=False), 
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 2. THE BATCH TRAINING LOOP ---
DISTANCES = [9] 

for d in DISTANCES:
    csv_file = f"d{d}_s11000_dataset.csv" 
    
    if not os.path.exists(csv_file):
        print(f"Skipping d={d}: {csv_file} not found.")
        continue

    print(f"\n🚀 Starting Hybrid Training for Distance d={d}")
    
    # A. Load and Prep Data
    df = pd.read_csv(csv_file)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int8)
    
    # B. Reshape for Conv1D-LSTM
    num_sensors = (d**2 - 1)
    X_reshaped = X.reshape(-1, 25, num_sensors)
    
    # C. Early Stopping
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=12, 
        restore_best_weights=True,
        verbose=1
    )
    
  
    model = build_turbo_model(d)
    model.fit(
        X_reshaped, y, 
        epochs=50, 
        batch_size=32, 
        validation_split=0.2, 
        callbacks=[early_stop], # Automatically manages stopping
        verbose=1
    )
    
    # E. Save the optimized result
    model_name = f"hybrid_model_d{d}.h5"
    model.save(model_name)
    print(f"✅ Saved trained model as: {model_name}")

print("\n✨ All models trained, optimized, and saved!")