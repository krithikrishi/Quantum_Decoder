import stim
import numpy as np
import tensorflow as tf
import os

# --- 1. SET THE TARGET DISTANCE (The only thing you change) ---
D = 7  

# --- 2. CONFIGURATION LOOKUP ---
# This matches your specific dataset shapes and training p-values
CONFIG = {
    3:  {"p": 0.003, "file": "lstm_model_d3_recovered.h5"},
    5:  {"p": 0.001, "file": "lstm_model_d5_recovered.h5"},
    7:  {"p": 0.001, "file": "lstm_model_d7_recovered.h5"},
    9:  {"p": 0.008, "file": "lstm_deep_d9_final.h5"},
    11: {"p": 0.001, "file": "placeholder_d11.h5"},
    13: {"p": 0.001, "file": "placeholder_d13.h5"}
}

# Auto-calculate parameters
p = CONFIG[D]["p"]
m_noise = 5 * p # Your Colab weight
r_noise = 2 * p # Your Colab weight
num_sensors = (D**2 - 1)
total_detectors = 25 * num_sensors # e.g., 25*48 = 1200

# --- 3. LOAD MODEL & CIRCUIT ---
base_path = os.path.dirname(os.path.abspath(__file__)) 
model = tf.keras.models.load_model(os.path.join(base_path, CONFIG[D]["file"]))

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=D, rounds=25,
    after_clifford_depolarization=p,
    after_reset_flip_probability=r_noise,
    before_measure_flip_probability=m_noise
)

# --- 4. DATA GENERATION & SLICING ---
sampler = circuit.compile_detector_sampler()
syndromes, logical_errors = sampler.sample(shots=1000, separate_observables=True)

# THE UNIVERSAL FIX: Slices based on the distance-specific detector count
clean_syndromes = syndromes[:, :total_detectors].astype(np.float32)

# --- 5. RESHAPE & PREDICT ---
X_test = clean_syndromes.reshape(-1, 25, num_sensors)
y_true = logical_errors.astype(np.int8).flatten()

print(f"--- Running Sync Test for d={D} (p={p}) ---")
probs = model.predict(X_test, verbose=0)
preds = (probs > 0.5).astype(int).flatten()

accuracy = np.mean(preds == y_true) * 100
print(f"🚀 RECOVERED ACCURACY: {accuracy:.2f}%")