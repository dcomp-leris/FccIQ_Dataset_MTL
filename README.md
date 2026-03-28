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


We also defined the network architecture and training hyperparameters for the models, which are presented in the following: 
<img width="786" height="330" alt="Screenshot from 2026-03-28 14-01-07" src="https://github.com/user-attachments/assets/6a06fe9a-35e4-4a56-943c-c8df5068a3e3" />
<img width="782" height="360" alt="Screenshot from 2026-03-28 14-01-48" src="https://github.com/user-attachments/assets/a7e6eaa7-f5fb-4ef7-ac8e-21ed3805ae39" />
<img width="789" height="337" alt="Screenshot from 2026-03-28 14-02-25" src="https://github.com/user-attachments/assets/8f0edf66-86ab-45ce-a5fa-3abdf145a055" />
<img width="769" height="301" alt="Screenshot from 2026-03-28 14-03-05" src="https://github.com/user-attachments/assets/e0af3ca7-f3e4-41d2-8fd6-6fda146c065f" />
<img width="780" height="333" alt="Screenshot from 2026-03-28 14-03-42" src="https://github.com/user-attachments/assets/4d57a0f4-3148-480f-8138-636195c15516" />
<img width="777" height="340" alt="Screenshot from 2026-03-28 14-04-25" src="https://github.com/user-attachments/assets/4ee124ae-b6df-4d23-b2a5-a46a92c74948" />


# Outputs
regression_out & cls_out predictions (saved + visualized)

Comparison of models by accuracy, loss, training time, inference time, and size
# Requirements
- numpy
-  pandas
-  torch
-  scikit-learn
-  matplotlib
-  seaborn
-  thop

# Project Structure
| File / Folder           | Description                                                      |
| -----------------------          | ---------------------------------------------------------------- |
   | `models.py`             | Models architecture |
| `main_training_loop_(train_all_models).py`                  | Training, validation, evaluation functions, Main script to train all models and evaluate performance                  |
| `Load and Preprocess Data.ipynb`             | Load, clean, scale, prepare datasets, Metrics calculation, plotting, helper functions                  |
| `dataset/`              | CSV files             |
| `README.md`             | Project documentation and instructions.                                            |
# Results
<img width="595" height="363" alt="Screenshot from 2026-01-19 10-04-01" src="https://github.com/user-attachments/assets/c65cf065-0bb1-40c3-991b-6b1769267fca" />

# Contact
For questions or collaboration:

Email: mina.kaviani@estudante.ufscar.br

Alternative: mina.kaviani22@gmail.com

