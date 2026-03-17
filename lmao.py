import stim
import numpy as np
import pandas as pd
import pymatching

# 1. SET UP THE NOISE MODEL (S11000)


p = 0.002

# S11000 weighting
measurement_noise = 2 * p
reset_noise = 2 * p


# 2. GENERATE THE QUANTUM GRID (Rotated Surface Code)

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=9,
    rounds=25,
    after_clifford_depolarization=p,
    after_reset_flip_probability=reset_noise,
    before_measure_flip_probability=measurement_noise
)


# 3. EXTRACT DETECTION EVENTS (SYNDROMES)

sampler = circuit.compile_detector_sampler()

syndromes, logical_errors = sampler.sample(
    shots=300000,
    separate_observables=True
)

print("Dataset Generated Successfully!")
print(f"Syndrome Bit Array Shape: {syndromes.shape}")
print(f"Total Shots: {len(logical_errors)}")



# 4. MWPM BASELINE VALIDATION

print("\nRunning MWPM Baseline Decoding...")


dem = circuit.detector_error_model(decompose_errors=True)


matching = pymatching.Matching.from_detector_error_model(dem)


predicted_logical = matching.decode_batch(syndromes)


baseline_ler = np.mean(
    predicted_logical.flatten() != logical_errors.flatten()
)

print(f"MWPM Baseline Logical Error Rate: {baseline_ler:.6f}")


#CONVERT DATASET
df = pd.DataFrame(syndromes.astype(int))
df['logical_error'] = logical_errors.astype(int)

df.to_csv('d9_s11000_dataset.csv', index=False)

print("\nDataset saved")
print(df.head())
