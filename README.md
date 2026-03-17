# Spatio-Temporal Neural Decoding for Rotated Surface Codes  
### A Hybrid CNN-BiLSTM Approach to Fault-Tolerant Quantum Computing

This repository contains the implementation of a high-fidelity neural decoder designed to handle the **S11000 noise model** ($5p$ measurement, $2p$ reset noise) across syndrome histories of **25 rounds**.

---

## 🚀 Key Results

Our hybrid architecture demonstrates significant superiority over classical baselines and standard Transformers in high-dimensional syndrome spaces.

| Code Distance (d) | Syndrome Features | Bi-LSTM Accuracy |
|-------------------|------------------|------------------|
| d = 3 | 200 | 97.09% |
| d = 5 | 600 | 98.36% |
| d = 7 | 1,200 | 98.43% |
| d = 9 | 2,000 | 92.49% |
| d = 11 | 3,000 | 64.50% |

---

## 🧠 Architecture

The decoder utilizes a **1D-Convolutional Neural Network (CNN)** for spatial feature extraction from the syndrome lattice, integrated with a **Bidirectional LSTM (Bi-LSTM)** backbone to capture temporal correlations across **25 rounds of measurement**.

---

## 🛠️ Setup & Reproducibility

### 1. Installation

```bash
git clone https://github.com/Krithik-Rishi/quantum-decoder.git
cd quantum-decoder
pip install -r requirements.txt
```

---

### 2. Data Generation

Since the datasets exceed GitHub's file size limits, generate them locally using our matched configuration script:

```bash
python src/data_gen.py
```

**Note:** This will recreate the exact shot distributions (up to **300,000 shots**) used in our research.

---

### 3. Training

To train the model:

```bash
python src/bilstm.py 
```

---

### 4. Live Inference Test

Verify the recovered accuracy of the trained weights:

```bash
python scripts/integration.py 
```

---

## 📝 Research Summary

Our study identifies a **"Generalization Wall"** at code distance **$d=11$**, where informational entropy in the **3,000-feature syndrome space** surpasses the inductive bias of current recurrent architectures.

This work advocates for **hierarchical decoding strategies** to achieve **sub-microsecond throughput in fault-tolerant hardware**.
