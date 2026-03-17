import pandas as pd
import numpy as np
import tensorflow as tf
import os

# --- MODEL CONFIG ---
model_configs = {
    3: {'file': 'hybrid_model_d3.h5', 'csv': 'd3_s11000_dataset.csv'},
    5: {'file': 'hybrid_model_d5.h5', 'csv': 'd5_s11000_dataset.csv'},
    7: {'file': 'hybrid_model_d7.h5', 'csv': 'd7_s11000_dataset.csv'},
    9: {'file': 'd9_bilstm_best.h5', 'csv': 'd9_s11000_dataset.csv'}
}

BATCH_SIZE = 512

print("\n" + "="*70)
print(f"{'Distance':<10} | {'Model File':<22} | {'Accuracy':<12} | {'Loss'}")
print("-"*70)

results = []

for d, config in model_configs.items():

    model_path = config['file']
    csv_path = config['csv']

    if not os.path.exists(model_path) or not os.path.exists(csv_path):
        print(f"d = {d:<7} | Missing files")
        continue

    try:
        # Load model
        model = tf.keras.models.load_model(model_path)

        # Automatically detect expected input shape
        _, rounds, sensors = model.input_shape

        # Load dataset
        df = pd.read_csv(csv_path)

        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.float32)

        del df

        # Reshape according to model
        X = X.reshape(-1, rounds, sensors)

        # TensorFlow dataset (faster batching)
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # Evaluate
        loss, accuracy = model.evaluate(dataset, verbose=0)

        print(f"d = {d:<7} | {model_path:<22} | {accuracy*100:>8.2f}% | {loss:.4f}")

        results.append((d, accuracy*100))

    except Exception as e:
        print(f"d = {d:<7} | Error: {str(e)[:40]}")

print("="*70)

# --- SUMMARY ---
if results:
    best_d, best_acc = max(results, key=lambda x: x[1])
    print(f"\n⭐ Best Model: Distance {best_d} with {best_acc:.2f}% accuracy")