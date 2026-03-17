import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import os

# --- 1. THE TRANSFORMER ARCHITECTURE (With Positional Encoding) ---
def build_transformer_v3(distance, rounds=25):
    num_sensors = (distance**2 - 1)
    inputs = layers.Input(shape=(rounds, num_sensors))
    
    # 1. Project and Add Positional Encoding
    x = layers.Dense(128)(inputs)
    pos_indices = tf.range(start=0, limit=rounds, delta=1)
    pos_embedding = layers.Embedding(input_dim=rounds, output_dim=128)(pos_indices)
    x = x + pos_embedding 

    # 2. Dual Transformer Blocks
    for _ in range(2):
        # Attention Block
        attn_output = layers.MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output) # (Residual 1)
        
        # Feed Forward Block (Fixing the shape mismatch here)
        ffn = layers.Dense(256, activation="relu")(x)
        ffn = layers.Dense(128)(ffn) # Projecting back to 128 so we can add it to 'x'
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn) # (Residual 2)

    # 3. Final Classification
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0005), 
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 2. RAM-FRIENDLY DATA LOADER ---
CSV_NAME = 'd9_s11000_dataset.csv'

print(f"📂 Reading {CSV_NAME} (1.1 GB)...")
# Load as int8 to save massive amounts of RAM
df = pd.read_csv(CSV_NAME, dtype=np.int8)
print("✅ CSV Loaded into Memory.")

# Prepare Features and Labels
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(np.float32)

# Clear the dataframe from RAM immediately
del df 

# Reshape for Transformer: (Samples, 25 Rounds, 80 Sensors)
X_reshaped = X.reshape(-1, 25, 80)
print(f"📊 Data Reshaped: {X_reshaped.shape}")

# --- 3. TRAINING ---
print("\n🚀 Starting d=9 Transformer Training...")

model = build_transformer_v3(9)

# Patience=12: Gives the AI plenty of time to find the signal in d=9
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=12, 
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_reshaped, y,
    epochs=50,
    batch_size=128, # Larger batch size for faster Transformer parallelization
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# --- 4. SAVE AND REPORT ---
best_val_acc = max(history.history['val_accuracy']) * 100
print(f"\n⭐ FINAL RESULTS: Best d=9 Accuracy: {best_val_acc:.2f}%")

model_name = "transformer_d9_optimized.h5"
model.save(model_name)
print(f"✅ Model saved as {model_name}")