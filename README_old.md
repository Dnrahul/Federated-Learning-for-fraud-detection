# Federated Learning for Cross-Bank Fraud Detection

**v2.0 - Enhanced with Differential Privacy, Advanced Aggregators, and Privacy Auditing**

A comprehensive PyTorch-based federated learning framework for privacy-preserving fraud detection across multiple financial institutions. Compares **FedAvg**, **FedProx**, and **FedDANE** algorithms with **Differential Privacy (DP-SGD)** and advanced privacy auditing.

---

## 🚀 Key Features

### Core Algorithms
- **FedAvg**: Standard federated averaging of client updates
- **FedProx** (μ): Improved optimization with proximal regularization for heterogeneous data
- **FedDANE** (NEW): Variance-reduced aggregation for faster convergence under non-IID data

### Privacy & Security (NEW)
- **Differential Privacy (DP-SGD)**: Per-sample gradient clipping + Gaussian noise injection
- **Privacy Accounting**: (ε, δ)-DP guarantees using Rényi Differential Privacy
- **Membership Inference Attacks**: Privacy auditing to quantify information leakage
- **Privacy-Utility Trade-off Analysis**: Systematic evaluation of accuracy vs privacy

### Robustness & Realism (NEW)
- **Non-IID Data Distribution**: Simulate realistic heterogeneous client data
- **Client Dropout Simulation**: Robustness evaluation under unreliable clients
- **Convergence Analysis**: Advanced monitoring and metrics tracking
- **Multi-model Support**: Standard and enhanced architectures with batch normalization & attention

### Advanced Visualization (NEW)
- Convergence curves comparing all algorithms
- Privacy-utility trade-off plots
- Non-IID and dropout robustness analysis
- ROC-AUC and Precision-Recall curves per client

📌 **Main Notebooks**:
- `Src/Fedrated_Learning.ipynb` - Original implementation
- `Advanced_FL_Analysis.ipynb` - Full-featured analysis with all new features

---

## Why Federated Learning?
Centralized training is often infeasible in regulated domains like banking due to privacy, security, and compliance constraints. Federated learning enables:
- **Raw data stays local** (banks do not share transaction-level records)
- **Cross-silo collaboration** (clients learn together without pooling data)
- A realistic setup for distributed institutions with heterogeneous data

---

## Project Goal
Train a fraud detection model collaboratively across multiple banks and evaluate how FL strategies behave when client data distributions differ.

---

## What This Repo Contains
- Multi-client FL setup (each CSV file = one client/bank)
- Consistent preprocessing across clients (encoding + scaling)
- PyTorch fraud detection model training per client
- Server aggregation and comparison of **FedAvg vs FedProx**
- **Client-wise evaluation** and comparison table of accuracies

---

## Method Summary
### Data & Preprocessing
- Loads multiple client datasets (CSV files)
- Removes non-learning fields (e.g., identifiers/unused columns)
- **Label-encodes** categorical variables and **standard-scales** numeric features
- Splits into per-client datasets for federated training

### Model
- A lightweight **PyTorch neural network** for binary classification (fraud vs non-fraud)

### Federated Training
Each communication round:
1. Server broadcasts the current global model to clients  
2. Each client trains locally for a few epochs  
3. Clients return model weights/updates  
4. Server aggregates:
   - **FedAvg**: averages client updates  
   - **FedProx**: local training includes proximal regularization with **μ**

Evaluation is performed **per client**, and results are summarized to compare FedAvg and FedProx.

---

## Results
The notebook produces a **client-wise comparison table** of accuracies for:
- **FedAvg**
- **FedProx**

This highlights performance variation across heterogeneous clients and how optimization affects convergence and stability.

---

## How to Run
### Requirements
- Python 3.9+
- Jupyter Notebook / JupyterLab

Suggested packages:
- `numpy`, `pandas`
- `scikit-learn`
- `torch`
- `matplotlib` (optional)

### Steps
1. Clone the repo
2. Install dependencies
3. Run the notebook:

---
### Future Scope (Next Phase: Privacy-Preserving Federated Learning)

Federated learning reduces raw data sharing, but **does not automatically guarantee privacy**—model updates can still leak information under certain threat models. The next phase of this project will introduce explicit privacy guarantees and quantify the privacy–utility trade-off.

### 1) Add Differential Privacy (DP)
- **DP-SGD (example-level DP):** apply per-sample gradient clipping + Gaussian noise during each client’s local training.
- **OR Client-level DP:** clip each client’s model update and add noise at the server during aggregation.
- Track privacy loss with an accountant and report **(ε, δ)**; benchmark **accuracy vs ε** across rounds/noise settings.

### 2) Measure Privacy Risk (Optional, Research-Grade)
- Implement a basic **membership inference** benchmark to estimate whether training participation can be inferred.
- Compare attack success **with vs without DP**, and report privacy gains alongside the utility drop.

### 3) Strengthen Realism Under Heterogeneity (Bonus)
- Simulate stronger **non-IID** client distributions and client dropouts; report robustness using worst-client accuracy and variance across clients.
- Evaluate FedAvg/FedProx/DP variants under the same settings for a consistent comparison.

```bash
jupyter notebook Fedrated_Learning.ipynb


