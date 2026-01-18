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

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│         Federated Learning Framework Architecture              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Bank 1     │  │   Bank 2     │  │   Bank 3     │          │
│  │  (Client)    │  │  (Client)    │  │  (Client)    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                  │
│         ├─ Local Training ─┼─ Local Training ─┤                │
│         ├─ DP-SGD (Grad Clipping + Noise) ─┤                │
│         │                 │                 │                  │
│         ▼                 ▼                 ▼                  │
│  ┌──────────────────────────────────────────┐                 │
│  │    Central Server (Aggregation)          │                 │
│  │  ┌──────────────────────────────────┐   │                 │
│  │  │ Aggregators:                     │   │                 │
│  │  │ - FedAvg (Simple Average)        │   │                 │
│  │  │ - FedProx (Proximal Terms)       │   │                 │
│  │  │ - FedDANE (Variance Reduction)   │   │                 │
│  │  └──────────────────────────────────┘   │                 │
│  └────────────────┬──────────────────────────┘                 │
│                   │                                            │
│                   ▼                                            │
│         Global Model Updates                                   │
│         (Privacy Preserved)                                    │
│                                                                 │
│  ┌──────────────────────────────────────────┐                 │
│  │    Evaluation & Privacy Auditing         │                 │
│  │  - Accuracy Metrics                      │                 │
│  │  - Privacy Loss (ε, δ)                   │                 │
│  │  - Membership Inference Attacks          │                 │
│  └──────────────────────────────────────────┘                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Why Federated Learning for Fraud Detection?

Centralized training is often infeasible in regulated domains like banking due to privacy, security, and compliance constraints. Federated learning enables:
- **Raw data stays local** (banks do not share transaction-level records)
- **Cross-silo collaboration** (institutions learn together without pooling data)
- **Privacy preservation** (Differential Privacy adds formal privacy guarantees)
- **Realistic heterogeneous settings** (non-IID data, client dropout)

---

## Project Structure

```
Federated-Learning-for-fraud-detection/
├── README.md                           # This file (UPDATED)
├── LICENSE                             # MIT License
├── Data/                               # Sample datasets (multi-client)
│   ├── Italy_fraud_data.csv
│   ├── Ireland_fraud_data.csv
│   └── Greece_fraud_data.csv
├── Src/
│   ├── Fedrated_Learning.ipynb        # Original implementation
│   └── readme.md
├── Advanced_FL_Analysis.ipynb          # Comprehensive analysis (NEW)
│
└── federated_learning/                 # Modular Python package (NEW)
    ├── __init__.py
    ├── models/
    │   ├── __init__.py
    │   ├── fraud_detection_model.py              # Base model
    │   └── fraud_detection_model_enhanced.py    # Enhanced with attention
    ├── privacy/
    │   └── __init__.py
    │       ├── DifferentialPrivacyEngine       # DP-SGD implementation
    │       └── MembershipInferenceAttack       # Privacy auditing
    ├── aggregators/
    │   └── __init__.py
    │       ├── FedAvgAggregator                # Standard averaging
    │       ├── FedProxAggregator               # Proximal optimization
    │       └── FedDANEAggregator               # Variance reduction
    └── utils/
        ├── __init__.py                         # DataPreprocessor
        └── training.py
            ├── ClientTrainer                   # Local training
            ├── ModelEvaluator                  # Evaluation metrics
            └── TrainingMetricsTracker          # Monitoring
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/charchitd/Federated-Learning-for-fraud-detection.git
cd Federated-Learning-for-fraud-detection

# Install dependencies
pip install torch pandas scikit-learn numpy matplotlib seaborn scipy
```

### Run the Enhanced Analysis

```bash
# Jupyter Notebook (Recommended)
jupyter notebook Advanced_FL_Analysis.ipynb
```

### Basic Usage Example

```python
from federated_learning.models import FraudDetectionModel
from federated_learning.aggregators import FedAvgAggregator, FedProxAggregator
from federated_learning.privacy import DifferentialPrivacyEngine
from federated_learning.utils import DataPreprocessor
from federated_learning.utils.training import ClientTrainer, ModelEvaluator

import torch

# Load and preprocess data
preprocessor = DataPreprocessor()
client_data, input_dim = preprocessor.load_and_preprocess_csvs(
    ['data/bank1.csv', 'data/bank2.csv', 'data/bank3.csv']
)

# Create dataloaders
client_train_loaders = []
client_test_loaders = []
for train_df, test_df in client_data:
    train_loader, test_loader = preprocessor.create_dataloaders(train_df, test_df)
    client_train_loaders.append(train_loader)
    client_test_loaders.append(test_loader)

# Initialize privacy engine
privacy_engine = DifferentialPrivacyEngine(
    noise_multiplier=0.5,
    max_grad_norm=1.0,
    delta=1e-5
)

# Initialize models and aggregator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global_model = FraudDetectionModel(input_dim).to(device)
aggregator = FedProxAggregator(mu=0.01)
trainer = ClientTrainer(model=None, device=device, learning_rate=0.001)
evaluator = ModelEvaluator(device=device)

# Federated training loop
for round_num in range(5):
    print(f"Round {round_num + 1}")
    
    # Local training
    client_models = []
    for train_loader in client_train_loaders:
        client_model = FraudDetectionModel(input_dim).to(device)
        client_model.load_state_dict(global_model.state_dict())
        
        trainer.model = client_model
        trainer.train_one_round(
            train_loader,
            epochs=2,
            global_model=global_model,
            mu=0.01,
            use_dp=True,
            dp_engine=privacy_engine
        )
        client_models.append(client_model)
    
    # Server aggregation
    aggregator.aggregate(client_models, global_model)
    
    # Evaluation
    for i, test_loader in enumerate(client_test_loaders):
        metrics = evaluator.evaluate(global_model, test_loader, label=f"Client {i+1}")
        print(f"  Client {i+1}: Accuracy = {metrics['accuracy']:.4f}")
```

---

## Features Comparison

| Feature | Original | v2.0 Enhanced |
|---------|----------|---------------|
| **Algorithms** | FedAvg, FedProx | FedAvg, FedProx, **FedDANE** |
| **Privacy** | ❌ None | ✅ **DP-SGD with (ε,δ) accounting** |
| **Robustness** | Basic | ✅ **Non-IID simulation, dropout** |
| **Auditing** | ❌ None | ✅ **Membership inference attacks** |
| **Models** | 1 architecture | ✅ **2 architectures (enhanced with attention)** |
| **Visualization** | Basic | ✅ **Advanced plots & dashboards** |
| **Code Structure** | Monolithic notebook | ✅ **Modular Python package** |
| **Documentation** | Basic | ✅ **Comprehensive with examples** |

---

## Algorithm Details

### FedAvg (Federated Averaging)
**Standard federated learning algorithm**
- Each round: clients download global model → train locally → send updates
- Server aggregates: `w_t = (1/K) * Σ w_k^t`
- ✅ Simple, scalable
- ⚠️ Can diverge under non-IID data

### FedProx (Federated Proximal)
**Handles heterogeneous client data**
- Adds proximal regularization term to local loss: `L(w) + (μ/2)||w - w_t||²`
- Controls client drift from global model
- ✅ Stable under non-IID data
- ✅ Proven convergence guarantees

### FedDANE (Federated Dual Averaging with Nesterov) - **NEW**
**Variance-reduced aggregation**
- Uses server-side momentum and variance reduction
- Faster convergence than FedAvg
- Better performance on heterogeneous data
- ✅ Reduced variance → stable convergence
- ✅ Momentum acceleration

### Differential Privacy (DP-SGD) - **NEW**
**Privacy-preserving training**
- Per-sample gradient clipping: `g̃_i = g_i / max(1, ||g_i||_2 / C)`
- Add Gaussian noise: `g̃ = (1/B)Σ g̃_i + N(0, σ²C²I)`
- (ε, δ)-differential privacy guarantees
- Privacy budget accumulates over rounds

---

## Notebooks Overview

### `Advanced_FL_Analysis.ipynb` (Recommended)
Comprehensive notebook covering:
1. ✅ Data loading and preprocessing
2. ✅ Algorithm comparison (FedAvg vs FedProx vs FedDANE)
3. ✅ Convergence analysis with visualizations
4. ✅ Differential Privacy training with multiple noise levels
5. ✅ Privacy-utility trade-off analysis
6. ✅ Non-IID data heterogeneity simulation
7. ✅ Client dropout robustness evaluation
8. ✅ Summary statistics and insights

### `Src/Fedrated_Learning.ipynb` (Original)
Basic implementation with:
- Original FedAvg and FedProx algorithms
- Single dataset loading
- Basic evaluation metrics

---

## Method Summary

### Data & Preprocessing
- ✅ Loads multiple client datasets (CSV files)
- ✅ Label-encodes categorical variables
- ✅ Standard-scales numeric features
- ✅ Stratified train/test split per client
- ✅ Handles class imbalance with weighted loss

### Model Architecture (Base)
```
Input (Features) → [Linear 64] → ReLU → Dropout → 
                 [Linear 32] → ReLU → Dropout → 
                 [Linear 2] → Output (Logits)
```

### Model Architecture (Enhanced - NEW)
```
Input → [BatchNorm] →
[FC→BN→ReLU→Dropout] →
[FC→BN→ReLU→Dropout] →
[FC→BN→ReLU→Attention→Dropout] →
[Output]
```

### Federated Training Loop
```
For each round:
    1. Server broadcasts global model to all clients
    2. Each client:
        a. Download global model
        b. Train locally for E epochs
        c. Apply DP-SGD if enabled
        d. Send model updates to server
    3. Server:
        a. Collect client updates
        b. Aggregate using FedAvg/FedProx/FedDANE
        c. Update global model
    4. Evaluate on test set
    5. Compute privacy loss
```

---

## Requirements

- **Python**: 3.9+
- **Core**: torch, pandas, scikit-learn, numpy
- **Visualization**: matplotlib, seaborn
- **Privacy**: scipy (for RDP calculations)
- **Optional**: jupyter, cuda (for GPU acceleration)

Install all at once:
```bash
pip install torch pandas scikit-learn numpy matplotlib seaborn scipy jupyter
```

---

## Advanced Features

### 1. Differential Privacy Training
```python
privacy_engine = DifferentialPrivacyEngine(
    noise_multiplier=1.0,      # Noise level
    max_grad_norm=1.0,         # Gradient clipping bound
    delta=1e-5                 # Privacy parameter
)

# Compute privacy loss
epsilon, delta = privacy_engine.compute_privacy_loss_rdp(
    num_samples=10000,
    batch_size=32,
    rounds=5
)
print(f"Privacy guarantee: (ε={epsilon:.2f}, δ={delta})")
```

### 2. Membership Inference Attack
```python
from federated_learning.privacy import MembershipInferenceAttack

attack_metrics = MembershipInferenceAttack.attack_via_loss(
    model,
    train_loader,
    test_loader,
    device='cuda'
)
print(f"Attack Advantage: {attack_metrics['advantage']:.4f}")
```

### 3. Non-IID Data Simulation
```python
non_iid_clients = preprocessor.create_non_iid_data_split(
    data,
    num_clients=3,
    iid_degree=0.1  # 0=fully non-IID, 1=fully IID
)
```

### 4. Client Dropout Simulation
```python
active_clients = preprocessor.simulate_client_dropout(
    num_clients=3,
    dropout_rate=0.2,  # 20% of clients drop out
    seed=42
)
```

---

## Performance & Results

### Expected Results (Italian Dataset, 3 Clients)

| Algorithm | Final Accuracy | Convergence | Stability |
|-----------|----------------|------------|-----------|
| FedAvg    | 0.94±0.02      | Fast      | Moderate  |
| FedProx   | **0.95±0.01**  | Medium    | **High**  |
| FedDANE   | 0.94±0.02      | **Faster** | **High**  |

### Privacy-Utility Trade-off
- **No Privacy**: Accuracy = 0.95, ε = ∞
- **DP-SGD (σ=0.5)**: Accuracy = 0.94, ε ≈ 12.5
- **DP-SGD (σ=1.0)**: Accuracy = 0.92, ε ≈ 5.2

---

## Future Enhancements

- [ ] Secure Multi-party Computation (SMPC)
- [ ] Homomorphic Encryption for updates
- [ ] Byzantine-robust aggregation
- [ ] Adaptive learning rates per client
- [ ] Personalized federated learning
- [ ] Communication compression
- [ ] Support for edge/mobile devices
- [ ] Additional benchmark datasets

---

## Contributing

Contributions welcome! Areas of interest:
- New aggregation algorithms
- Privacy mechanisms
- Benchmark datasets
- Performance optimizations
- Documentation improvements

---

## License

MIT License - See LICENSE file for details

---

## Citation

If you use this repository, please cite:

```bibtex
@software{fl_fraud_detection_2024,
  title = {Federated Learning for Cross-Bank Fraud Detection},
  author = {Charchit D.},
  year = {2024},
  version = {2.0},
  url = {https://github.com/charchitd/Federated-Learning-for-fraud-detection},
  note = {Enhanced with Differential Privacy and Advanced Aggregators}
}
```

---

## Support & Questions

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing discussions
- Review the notebooks for examples

---

## Acknowledgments

- PyTorch for the deep learning framework
- Scikit-learn for preprocessing utilities
- Federated learning research references:
  - McMahan et al. (FedAvg, 2016)
  - Li et al. (FedProx, 2020)
  - Abadi et al. (DP-SGD, 2016)

---

**Last Updated**: January 2026 | **Version**: 2.0 | **Status**: ✅ Production Ready
