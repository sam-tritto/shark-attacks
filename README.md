# 🦈 Battle of the Beach Thrillers: Zero-Shot TabFM vs. Saturated CatBoost Text Analytics (Hawai'i Edition)

Welcome to the **Battle of the Beach Thrillers**! This tutorial project demonstrates how to apply Google's state-of-the-art **TabFM** (Tabular Foundation Model) and **CatBoost** to classify case fatalities (`Fatal (Y/N)`) using real historical shark attack reports from the coastlines of **Hawai'i** 🌴 surfing, wading, and diving grounds.

## 🏄‍♂️ The Story & Premise

* **The Vibe:** High-stakes summer beach thriller.
* **The Task:** Binary Classification (`Fatal (Y/N)`) — predicting if a shark encounter will be fatal or survivable based on messy circumstances.
* **The Features:** `Country`, `Activity`, `Time of Day`, `Maturity of Shark` (extracted from species description), `Month`, and `Injury Type`.
* **The Challenge:** The official Global Shark Attack File (GSAF) contains notoriously messy, high-cardinality, and un-standardized categorical text narratives. For example, the `Activity` column doesn't say "Surfing" 500 times. It says:
  * *"Surfing, fell off board, bit on leg"*
  * *"Wading / standing in 3 ft of water"*
  * *"Spearfishing / free diving with catch"*

### ⚔️ The Head-to-Head Showdown

This tutorial pits two modern paradigms against each other:

1. **Google TabFM (Zero-Shot)**: A transformer-based tabular foundation model pre-trained on millions of synthetic causal data matrices. Using an alternating row/column attention mechanism, it performs **In-Context Learning (ICL)**, reading historical records as a "prompt" to predict test targets in a single forward pass—**without a single gradient update or manual text embedding step**.
2. **CatBoost Classifier (Supervised)**: A powerful, gradient-boosted decision tree algorithm trained explicitly on the training data using its **native text processing engines** and dictionary builders to tokenize and build text features on the fly.

---

## 📁 Directory Layout

```text
shark_tabfm_tutorial/
├── pyproject.toml           # Project metadata & uv dependencies configuration
├── README.md                # This documentation
├── download_data.py         # Script to fetch, filter for Hawai'i, and clean GSAF data
├── notebook_tutorial.ipynb  # Core Jupyter tutorial notebook
└── utils/
    ├── __init__.py
    └── data_processors.py   # Subsamples stratified training context for TabFM
```

---

## ⚙️ Environment Setup & Installation

This project uses `uv` for lightning-fast and reproducible package management.

### 1. Install `uv` (if not already installed)
Follow the official instructions or run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Project Dependencies
Run `uv sync` from the project root to create a virtual environment (`.venv`) and install all required packages (including PyTorch, CatBoost, and Google TabFM from Git):
```bash
uv sync
```

---

## 🚀 How to Run the Tutorial

### Step 1: Download & Clean the Data
Run the ingestion script. It will download the official GSAF Excel spreadsheet, filter it for incidents around **Hawai'i** (which naturally limits the dataset to several hundred rows for quick processing), clean/rename the target and features, and export the file as `data/gsaf.csv`.
```bash
uv run download_data.py
```

### Step 2: Open and Run the Jupyter Notebook
Launch the Jupyter Notebook interface:
```bash
uv run jupyter notebook notebook_tutorial.ipynb
```
Select the project's virtual environment `.venv` as your Python kernel and execute all cells.

---

## 🧠 Key Lessons & Revelations

### 1. The In-Context Paradigm Shift
TabFM does not run gradient backpropagation or weight updates on the shark attack data. It simply reads the training set as semantic context (like a prompt) inside its attention window. To update your model's knowledge, you simply append new historical records to the data lake—no scheduled retraining cron jobs required!

### 2. Implicit Feature Engineering
While standard tokenizers process words in isolation (e.g. TF-IDF), TabFM's internal transformers perform cross-attention over rows and columns. It intuitively understands that the combination of `"Spearfishing"` and `"Thigh laceration"` represents a vastly different risk profile than `"Wading"` and `"Thigh laceration"`, extracting these interactions on the fly.

### 3. The Calibration Trade-off
Zero-shot models are excellent at ranking risk (ranking AUC), but their raw predicted probabilities may not match the actual baseline fatality rate of Hawaii shark attacks. We plot a **Calibration Curve (Reliability Diagram)** to show the difference between the supervised CatBoost (which naturally aligns with the baseline rate) and TabFM. You will learn why post-processing calibration (like Platt scaling) is crucial for foundation models in critical domains.
