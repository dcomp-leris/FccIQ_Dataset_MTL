# Multi-Task Learning Architectures for Joint Interference Detection and KPI Prediction in 5G Networks
# Overview
This project investigates multi-task learning (MTL) architectures for real-time interference detection and Key Performance Indicator (KPI) prediction in 5G Radio Access Networks (RANs).
The objective is to jointly solve two heterogeneous tasks:

Interference detection (binary classification)

KPI prediction (continuous regression)

while mitigating negative transfer between tasks and minimizing computational overhead, which is critical for real-time deployment in wireless networks.

We conduct a systematic comparison between Single-Task Learning (STL) and several state-of-the-art MTL architectures, analyzing trade-offs among prediction accuracy, regression error, model complexity, and inference latency.

# Implemented Architectures
The following models are implemented and evaluated:

-  STL (Single-Task Learning)
Separate GCN models for each task

-  Hard Parameter Sharing
Shared GCN backbone with task-specific output heads

-  MMoE (Multi-gate Mixture-of-Experts)
Shared expert layers with task-specific gating networks

-  Cross-Stitch Networks
Learnable feature-sharing layers between tasks

-  PLE (Progressive Layered Extraction)
Multi-level shared and task-specific experts with gating

-  Attention-based MTL
GCN backbone with task-specific attention mechanisms

Model Outputs:

regression_out → [RSRP, RSRQ, SINR]

cls_out → interference label

All architectures are fully compatible with the same preprocessing pipeline and training loop.
# Training & Evaluation

-  Training loop implementation
-  Validation and testing pipeline
-  Multi-task loss computation
   - [x] Classification loss
   - [x] Per-task regression losses
- Model statistics computation
  - [x] Number of parameters
  - [x] FLOPs
  - [x] Model size


# Outputs
regression_out & cls_out predictions (saved + visualized)

Comparison of models by accuracy, loss, training time, inference time, and size
# Requirements
- numpy
-  pandas
-  torch
-  torch_geometric
-  scikit-learn
-  matplotlib
-  seaborn
-  thop

# Project Structure
| File / Folder           | Description                                                      |
| -----------------------          | ---------------------------------------------------------------- |
| `Load and Preprocess Data.ipynb`             | Multi-task GNN model definitions (GCN, MMoE, Cross-Stitch, etc.) |
| `train_loop.py`                  | Training, validation, and evaluation functions                   |
| `utils.py`              | Metrics calculation, plotting, helper functions                  |
| `data_preprocessing.py` | Load, clean, scale, and prepare datasets                         |
| `dataset/`              | CSV files             |
| `main_train.py`         | Main script to train all models and evaluate performance         |
| `README.md`             | Project documentation                                            |
# Results
<img width="595" height="363" alt="Screenshot from 2026-01-19 10-04-01" src="https://github.com/user-attachments/assets/c65cf065-0bb1-40c3-991b-6b1769267fca" />

# Contact
For questions or collaboration:

Email: mina.kaviani@estudante.ufscar.br

Alternative: mina.kaviani22@gmail.com

