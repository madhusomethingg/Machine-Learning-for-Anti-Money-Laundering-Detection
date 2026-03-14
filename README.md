# 🔍 Machine Learning for Anti-Money Laundering Detection
### From Transaction Classification to Network-Level Signal Discovery

> Detecting suspicious financial activity across 4.5M+ transactions using classical ML, XGBoost, and dimensionality reduction — with a graph-based research extension exploring network topology as a predictive signal.

---

## 📌 Overview

Money laundering undermines the integrity of financial systems and facilitates corruption, terrorism, and organized crime. Traditional rule-based systems fail at scale — criminals easily adapt by smurfing large sums, routing funds through intermediary chains, and exploiting static thresholds.

This project takes a data-driven approach to the problem:

| Research Question | Approach |
|---|---|
| What transaction patterns indicate laundering? | EDA + feature engineering across behavioral, network & temporal dimensions |
| How effectively can ML classify suspicious transactions? | 6 models benchmarked; XGBoost selected as best performer |
| What features are most significant? | Correlation analysis + PCA visualization + XGBoost feature importance |

---

## 📂 Dataset

**IBM Transactions for Anti-Money Laundering (AML)**  
Source: [Kaggle – IBM AML Dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)

- **File used:** `HI-Small_Trans.csv`
- **Size:** 4,507,864 rows × 11 columns
- **Class balance:** ~99.9% legitimate, ~0.095% suspicious (heavily imbalanced)

| Column | Description |
|--------|-------------|
| `Timestamp` | Date and time of transaction |
| `From Bank` | Sending bank ID |
| `Account` | Sender account ID |
| `To Bank` | Receiving bank ID |
| `Account.1` | Receiver account ID |
| `Amount Received` | Amount received |
| `Receiving Currency` | Currency of received amount |
| `Amount Paid` | Amount paid |
| `Payment Currency` | Currency of payment |
| `Payment Format` | Type of payment (Cash, Cheque, ACH, Credit Card, Wire, Bitcoin, Reinvestment) |
| `Is Laundering` | Target label (0 = legitimate, 1 = suspicious) |

---

## 🔧 Tech Stack

| Category | Libraries |
|----------|-----------|
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| ML & Preprocessing | `scikit-learn`, `category_encoders` |
| Deep Learning | `PyTorch` |
| Boosting | `XGBoost` |
| Dimensionality Reduction | `TSNE`, `PCA`, `TruncatedSVD` |

---

## 🔬 Methodology

### 1. Data Preprocessing & Cleaning
- Removed duplicates; handled 1 NaN row across key columns
- Renamed `Is Laundering` → `Is_Laundering`
- Label encoded categorical columns; encoded `Payment Format` ordinally by complexity:
  ```
  Cash(1) → Cheque(2) → ACH(3) → Credit Card(4) → Wire(5) → Bitcoin(6) → Reinvestment(7)
  ```
- Normalized all features using `MinMaxScaler` (range: −1 to 1)
- Converted `Timestamp` to Unix epoch seconds

### 2. Feature Engineering

Three behavioral dimensions were engineered:

**Transaction Behavior**
- Transaction frequency per account, average transaction amount
- Deviation from historical baseline, transaction type distribution

**Network Behavior**
- Number of unique counterparties, account fan-in / fan-out ratio
- Circular money flow patterns, repeated transaction loops

**Temporal Behavior**
- Burst transaction windows, transactions at unusual hours
- Day-of-week anomalies, time gap between transfers

### 3. Handling Class Imbalance
- Applied **undersampling**: matched non-fraud samples to fraud sample count (4,281 each)
- Used `StratifiedKFold` (5 splits) to preserve label proportions across folds
- Removed 5 timestamp outliers using IQR method

### 4. Dimensionality Reduction & Visualization

| Method | Time | Notes |
|--------|------|-------|
| t-SNE | ~75s | Best visual cluster separation |
| PCA | ~0.006s | Fast; revealed non-linear signal structure |
| Truncated SVD | ~0.21s | Similar to PCA |
| t-SNE + PCA | ~75s | Hybrid |

> **Key Finding:** PCA showed significant overlap between fraud and non-fraud clusters — the data is **not linearly separable**. The laundering signal lives in nonlinear feature interactions, not individual transaction flags.

### 5. Model Benchmarking

| Model | Accuracy | F1 Score | Notes |
|-------|:--------:|:--------:|-------|
| Logistic Regression | 72% | 0.68 | Baseline |
| K-Nearest Neighbors | 74% | 0.70 | Slow on 5M rows |
| Support Vector Machine | 78% | 0.74 | High compute cost |
| Decision Tree | 80% | 0.76 | Overfits |
| Neural Network (PyTorch) | 85% | 0.82 | Slower, no gain over XGBoost |
| **XGBoost ★** | **89%** | **0.85** | **Best performer** |

**Why XGBoost won:** sequential boosting iteratively corrects hard-to-classify samples, handles class imbalance natively, and captures nonlinear feature interactions that linear models miss. Deep learning showed no improvement — tree ensembles remain state-of-the-art for financial tabular data.

---

## 📊 Key Results

- **XGBoost** achieved **89% accuracy** and **F1 = 0.85** on the balanced test set
- PCA visualization confirmed the data is **not linearly separable** — validating the use of ensemble tree methods
- `Timestamp` showed positive correlation with laundering labels; `Payment Format` showed negative correlation
- Undersampling to 4,281 samples per class was necessary to address the ~1000:1 class imbalance

---

## 🧠 Research Proposal: Transaction Network Topology as Predictive Signal

The PCA result raised a deeper question — if fraud and legitimate transactions are visually indistinguishable at the individual level, **are we even looking at the right unit of analysis?**

Money laundering isn't just a transaction problem. It's a network problem. Criminals don't move money in a single hop — they route it through chains of intermediary accounts, creating structures that look normal transaction-by-transaction but are unmistakably suspicious when viewed as a whole. A laundering ring leaves a fingerprint in the *shape* of the transaction graph, even when every individual transfer appears clean.

This is what motivated the research extension: **extract structural signals from the transaction network itself**, rather than from individual rows of data.

> *Can graph-based features extracted from financial transaction networks generate predictive signals for abnormal financial behavior — earlier and more robustly than individual transaction features?*

### Proposed Pipeline

```
Step 1: Build Transaction Graph
        Daily directed graph — accounts as nodes, transactions as weighted edges

Step 2: Extract Network Features
        PageRank · Betweenness Centrality · Eigenvector Centrality · Clustering Coefficient

Step 3: Graph Embeddings
        Node2Vec / DeepWalk — convert topology into dense feature vectors

Step 4: Generate Signals
        Embeddings fed into classifier to flag suspicious account clusters
```

### What Each Network Signal Catches

| Feature | What It Measures | The Signal |
|---------|-----------------|------------|
| **PageRank** | Structural importance in the network | A dormant account that suddenly becomes central is worth investigating — legitimate low-activity accounts don't do that |
| **Betweenness Centrality** | How often an account bridges senders and receivers | High betweenness + low transaction volume is a fingerprint of a *layering account* — one that exists just to pass money through |
| **Clustering Coefficient** | How tightly connected an account's neighbors are | A dense, self-referential cluster of accounts transacting heavily with each other → potential laundering ring |
| **Transaction Cycles** | Funds returning to origin after multiple hops | Money completing a circuit is the hallmark of financial layering |

### Why This Is Novel

Individual transaction classifiers — even good ones like XGBoost — are fundamentally reactive. They wait for a suspicious transaction to occur and then flag it. Graph-based signals can catch suspicious *structures forming* before any individual transaction becomes obviously anomalous. It's an earlier warning system, operating at a higher level of abstraction — and one that's much harder for bad actors to evade, because disguising network topology is far more difficult than disguising transaction amounts.

---

## ⚠️ Limitations

- Dataset is **simulated** — real-world noise and adversarial behavior may reduce performance
- PCA may discard financially meaningful features in exchange for variance
- No temporal graph evolution tracked — the network is treated as static, not rolling

---

## 🔮 Future Work

- **Dynamic Graph Signals** — track how centrality scores evolve over rolling time windows. A *rising* PageRank is more suspicious than a high one
- **GNN-Based Detection** — Graph Neural Networks learn structural embeddings end-to-end directly from the transaction graph, without manual feature engineering
- **Temporal Attention Models** — apply transformer attention to account transaction sequences, treating financial history as language and laundering as anomalous grammar

---

## 🚀 Getting Started

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn category_encoders xgboost torch
```

1. Clone the repo and download `HI-Small_Trans.csv` from [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml), place it in a `data/` folder
2. Open `notebook.ipynb` in Jupyter or Google Colab

---

## 👤 Author

Madhumitha Rajagopal

---

## 📄 License

This project is for educational and research purposes. The IBM AML dataset is provided via Kaggle under their respective terms.
