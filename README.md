# GraphRP: Defending against Model Extraction for GNNs with Model Reprogramming

[![KDD 2026](https://img.shields.io/badge/KDD-2026-blue)](https://kdd2026.kdd.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Paper](https://img.shields.io/badge/Paper-ACM%20DL-red)](https://doi.org/10.1145/3770855.3817983)

Official implementation of **GraphRP**, accepted at **KDD 2026** (Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining, August 9–13, 2026, Jeju Island, Republic of Korea).

> **Defending against Model Extraction for GNNs with Model Reprogramming**  
> Yan Wen, Zhenyi Wang, Heng Huang  
> *KDD 2026* · [ACM Digital Library](https://doi.org/10.1145/3770855.3817983)

---

## Overview

**GraphRP** is a proactive defense framework that protects Graph Neural Networks (GNNs) deployed as black-box APIs from Model Extraction (ME) attacks. Instead of applying static noise, GraphRP leverages *Model Reprogramming* to introduce a **Structure-Aware Gating Mechanism** that:

- **Preserves utility** for benign queries that reside on the training manifold (gating factor α ≈ 0)
- **Poisons labels** for adversarial/OOD queries by maximizing Fisher Information distance (gating factor α ≈ 1)
- **Requires no model retraining** — only a small set of learnable reprogramming parameters Θ_rep are optimized

![GraphRP Overview](assets/overview.png)

---

## Key Results

GraphRP consistently reduces clone model accuracy by up to **17%** compared to state-of-the-art baselines while maintaining high benign utility across diverse graph benchmarks (MUTAG, ENZYMES, NCI1, PROTEINS, OGB-MolHIV, COLLAB).

| Defense        | MUTAG Clone Acc ↓ | ENZYMES Clone Acc ↓ | Benign Acc ↑ |
|----------------|:-----------------:|:-------------------:|:------------:|
| Undefended     | 0.765             | 0.561               | 0.945        |
| MeCo (best baseline) | 0.712       | 0.482               | 0.930        |
| **GraphRP (Ours)** | **0.603**     | **0.364**           | **0.932**    |

---

## Requirements

```bash
Python >= 3.8
PyTorch >= 1.12
PyTorch Geometric >= 2.1
torch-scatter, torch-sparse
scikit-learn
numpy
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Repository Structure

```
GraphRP-KDD2026/
├── README.md
├── requirements.txt
├── data/                   # Dataset loading utilities
├── models/
│   ├── victim/             # Victim GNN architectures (GCN, GraphSAGE, GIN)
│   └── clone/              # Clone model architectures
├── defense/
│   ├── graphrp.py          # Core GraphRP defense module
│   ├── gating.py           # Structure-Aware Prototype Gating
│   └── losses.py           # Utility / Defense / Structural loss functions
├── attack/
│   ├── knockoff.py         # KnockoffNet attack implementation
│   └── adaptive.py         # Gray-box adaptive attacker (GraphGAN-based)
├── train_defense.py        # Main training script for GraphRP
├── evaluate.py             # Evaluation script (clone accuracy + utility)
└── configs/                # Hyperparameter configs per dataset
```

---

## Quick Start

**Step 1: Train the victim model**
```bash
python train_victim.py --dataset MUTAG --model GIN --epochs 200
```

**Step 2: Apply GraphRP defense**
```bash
python train_defense.py --dataset MUTAG --beta1 1.0 --beta2 0.5 --K 5
```

**Step 3: Simulate model extraction attack**
```bash
python attack/knockoff.py --dataset MUTAG --label_type soft --budget 10x
```

**Step 4: Evaluate clone accuracy**
```bash
python evaluate.py --dataset MUTAG --defense graphrp
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{wen2026graphrp,
  title     = {Defending against Model Extraction for GNNs with Model Reprogramming},
  author    = {Wen, Yan and Wang, Zhenyi and Huang, Heng},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  series    = {KDD '26},
  year      = {2026},
  month     = {August},
  address   = {Jeju Island, Republic of Korea},
  publisher = {ACM},
  doi       = {10.1145/3770855.3817983},
  url       = {https://doi.org/10.1145/3770855.3817983}
}
```

---

## License

This project is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## Contact

For questions about the paper or code, please open an issue or contact:  
**Yan Wen** — ywen1@umd.edu  
University of Maryland, College Park
