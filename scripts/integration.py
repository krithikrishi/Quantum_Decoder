import stim
import tensorflow as tf
import numpy as np
import os

# 1. Setup paths to your desktop models

base_path = r"C:\Users\Krithik Rishi\OneDrive\Desktop\project"

def live_test_decoder(distance, model_file):
    print(f"\n--- Testing Distance d={distance} ---")
    
    # Load the specific model
    model_path = os.path.join(base_path, model_file)
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return

    model = tf.keras.models.load_model(model_path)
    
    # Generate 10 live test cases using local Stim
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", 
        distance=distance, 
        rounds=25, 
        after_clifford_depolarization=0.01
    )
    sampler = circuit.compile_detector_sampler()
    syndromes, truths = sampler.sample(shots=10, separate_observables=True)

    # 2. Reshape and Predict
    num_sensors = distance**2 - 1
    reshaped_syndromes = syndromes.reshape(-1, 25, num_sensors)
    predictions = model.predict(reshaped_syndromes, verbose=0)

    # 3. Analyze Results
    correct = 0
    for i in range(10):
        pred_label = 1 if predictions[i][0] > 0.5 else 0
        actual_label = truths[i][0]
        if pred_label == actual_label:
            correct += 1
            
    print(f"✅ Local Batch Test: {correct}/10 correct.")
    print(f"Success Rate: {correct*10}%")

# Run the test for your hero model
# live_test_decoder(7, "lstm_model_d7_recovered.h5")