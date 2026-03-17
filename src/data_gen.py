import stim
import numpy as np
import pandas as pd
import pymatching
import os

# Create a data directory
if not os.path.exists('data'):
    os.makedirs('data')

config = {
    3:  {"p": 0.003, "shots": 10000},
    5:  {"p": 0.001, "shots": 10000},
    7:  {"p": 0.001, "shots": 50000},
    9:  {"p": 0.002, "shots": 300000}, 
    11: {"p": 0.002, "shots": 150000}, 
    13: {"p": 0.002, "shots": 50000}
}

rounds = 25
results = []

print(f"Starting Matched Batch Generation (Rounds={rounds})")
print("-" * 60)

for d, settings in config.items():
    p = settings["p"]
    shots = settings["shots"]
    
    # S11000 noise weighting 
    # Measurement noise = 5p | Reset noise = 2p 
    m_noise = 5 * p 
    r_noise = 2 * p 
    
    print(f"Processing d={d} | p={p} | Shots: {shots}")
    
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=r_noise,
        before_measure_flip_probability=m_noise
    )

    sampler = circuit.compile_detector_sampler()
    syndromes, logical_errors = sampler.sample(
        shots=shots,
        separate_observables=True
    )

    # MWPM Baseline calculation
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    predicted_logical = matching.decode_batch(syndromes)
    baseline_ler = np.mean(predicted_logical.flatten() != logical_errors.flatten())
    print(f"-> MWPM Baseline LER: {baseline_ler:.6f}")

    # Save to CSV 
    df = pd.DataFrame(syndromes.astype(np.int8))
    df['logical_error'] = logical_errors.astype(np.int8)
    
    file_name = f'data/d{d}_matched_dataset.csv'
    df.to_csv(file_name, index=False)
    print(f"-> Saved to {file_name}")
    
    results.append({
        "Distance": d, 
        "Shots": shots, 
        "MWPM_LER": baseline_ler, 
        "Features": syndromes.shape[1]
    })

# Summary Report
summary_df = pd.DataFrame(results)
summary_df.to_csv('generation_summary.csv', index=False)
print("\n" + "="*60)
print(summary_df)