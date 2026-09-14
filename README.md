# VietPopNet 🇻🇳📊

> **Spatial Population Density Estimation using Deep Learning & Machine Learning in Vietnam**

VietPopNet is a research and deep learning project aimed at modeling and predicting commune-level population density across Vietnam's 34 target provinces using socio-economic indicators, geographic attributes, and land usage microcensus data.

---

## 🌟 Key Features

- **PyTorch Deep Neural Network (VietPopNet MLP)** with BatchNorm & Dropout regularization.
- **Comparative Baseline Models**: Linear Regression, Random Forest, XGBoost.
- **Explainable AI (XAI)**: Feature attribution using **SHAP** (SHapley Additive exPlanations).
- **Synthetic Microcensus Pipeline**: Standardized generation script covering all 6 geographical regions of Vietnam.

---

## 🏗 Repository Structure

```
VietPopNet/
├── data/                                 # Census data & synthetic datasets
│   └── synthetic_commune_census_microcensus.csv
├── notebooks/                            # Step-by-step experiment notebooks
├── src/                                  # Modular PyTorch & ML source code
│   ├── dataset.py                        # Dataset & DataLoader abstractions
│   ├── model.py                          # VietPopNet architecture (v0.1 -> v1.0)
│   ├── train.py                          # Training loop, checkpoints, early stopping
│   └── evaluate.py                       # Evaluation metrics & SHAP interpretability
├── generate_microcensus.py               # Data synthesis generator
├── requirements.txt                      # Project dependencies
└── README.md                             # Project overview
```

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/fucdunko23-uet-vnu/vietpopnet.git
cd vietpopnet

# Create virtual environment & install dependencies
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Data Generation & Pipeline Check

```bash
python generate_microcensus.py
python src/model.py
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
