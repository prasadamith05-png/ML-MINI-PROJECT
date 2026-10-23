# 🏈 College Football Bowl Game Outcome & Point Spread Predictor
> **UE24CS352A — Machine Learning Mini-Project**  
> *Reproduction & Benchmark Extension of Stanford CS229 Project 101 (Cheshire, Childs, Leung)*

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-FF4B4B.svg)](https://streamlit.io/)

---

## 📌 Executive Summary

NCAA College Football postseason bowl games represent one of the most volatile predictive domains in sports analytics. Unlike regular season intra-conference games, bowl matchups pair elite programs from disparate conferences that rarely play one another, compete on neutral fields after multi-week layoffs, and feature profound disparities in **Strength of Schedule (SOS)**.

This project delivers an end-to-end Machine Learning pipeline that reproduces and extends the seminal **Stanford CS229 Project 101** (*Evan Cheshire, Austin Childs, Thomas Leung*):
1. **Binary Classification:** Accurately forecasts the outright winner of postseason bowl games with calibrated win probabilities.
2. **Spread Regression:** Accurately predicts the continuous point spread ($\Delta \text{Points} = \text{Score}_A - \text{Score}_B$) with low Mean Absolute Error.
3. **Bowl Pick'em Optimization:** Employs a confidence-calibrated decision engine allocating 1 to $N$ confidence points to maximize expected points in competitive Pick'em pools.
4. **Live Interactive Web App:** A full-featured Streamlit demonstration dashboard allowing evaluators to pick any matchup, tweak regular season stats, and view real-time spread predictions and probability gauges.

---

## 📂 Repository Directory Structure

```text
college-football-bowl-predictor/
├── data/
│   ├── build_dataset.py                 # Historical bowl game and FBS team data generator
│   ├── fbs_teams_database.csv           # 32 major FBS team regular-season statistics
│   └── historical_bowl_games.csv        # 747 historical bowl games (2000–2023)
├── src/
│   ├── data_loader.py                   # Ingestion, temporal splitting, and scaling pipeline
│   ├── models.py                        # First-principles ML models (NumPy/SciPy)
│   ├── feature_selection.py             # Forward stepwise & Lasso L1 sparsity analysis
│   └── evaluate.py                      # Metrics (Accuracy, AUC, Brier, MAE) & Pick'em pool simulator
├── models/
│   └── trained_bowl_predictor_pipeline.pkl # Serialized trained models & scalers
├── figures/
│   ├── model_comparison.png            # Accuracy & Pick'em pool score comparison
│   ├── feature_importance.png          # Gini impurity & variance reduction ranking
│   ├── roc_curves.png                  # Receiver Operating Characteristic (ROC) curves
│   └── spread_prediction_scatter.png   # Predicted vs actual point spread calibration
├── deliverables/
│   ├── College_Football_Bowl_Predictor_Writeup.pdf # Concise 2-page academic report (PDF)
│   ├── generate_writeup_pdf.py         # ReportLab PDF compiler
│   └── model_benchmark_results.csv     # Comprehensive model performance table
├── presentation/
│   ├── index.html                      # Interactive 8-slide presentation deck (open in browser)
│   └── slides.md                       # Presentation transcript & Review Panel Q&A notes
├── notebooks/
│   ├── college_football_bowl_predictor.ipynb # Interactive walkthrough notebook
│   └── generate_notebook.py
├── app.py                              # Interactive Streamlit Web Application (Live Demonstration)
├── train.py                            # Unified training, benchmarking, and figure generator
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Git ignore rules
└── README.md                           # Complete documentation & setup instructions
```

---

## ⚙️ Setup and Installation Guide

### Prerequisites
- Python 3.10 or 3.11 installed.
- Git installed.

### 1. Clone or Navigate to the Repository
```bash
git clone <your-private-repo-url>
cd college-football-bowl-predictor
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

### 1. Run Unified Model Training & Benchmarking
Trains all 10 model architectures, computes metrics across out-of-time test games (2022–2023), runs feature selection, generates publication figures, and saves model artifacts:
```bash
python train.py
```

### 2. Launch the Interactive Streamlit Live Demo (For Faculty Review)
Launch the interactive web application to demonstrate real-time head-to-head predictions:
```bash
streamlit run app.py
```
*The app will automatically open in your browser at `http://localhost:8501`.*

### 3. Regenerate the Exact 2-Page PDF Write-up
Compile the publication-ready 2-page PDF document adhering to UE24CS352A guidelines:
```bash
python deliverables/generate_writeup_pdf.py
```
*The output PDF will be generated at `deliverables/College_Football_Bowl_Predictor_Writeup.pdf`.*

### 4. Present the Slide Deck (For Final Presentation)
Open `presentation/index.html` in any web browser (Chrome, Edge, Firefox, Safari):
- Use **Left/Right arrow keys** or **Spacebar** to advance through slides.
- Speaker notes and Q&A preparation are available in `presentation/slides.md`.

---

## 📊 Experimental Results & Benchmark Comparison

All models were evaluated on an **out-of-time test set (2022–2023 bowl games, 63 games)** to eliminate lookahead bias:

| Model Architecture | Winner Accuracy (%) | ROC-AUC | F1-Score | Spread MAE (pts) | Spread RMSE (pts) | Pick'em Pool Score (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gradient Boosting** | **79.37%** | **0.8220** | **0.7937** | 7.75 | 9.73 | **80.90% (1631 / 2016 pts)** |
| **Logistic Regression** | 74.60% | 0.7683 | 0.7576 | 7.61 | 9.94 | 76.74% (1547 / 2016 pts) |
| **Linear SVM** | 74.60% | 0.7733 | 0.7576 | 8.06 | 10.16 | 76.69% (1546 / 2016 pts) |
| **Random Forest** | 73.02% | 0.7911 | 0.7385 | 8.27 | 10.49 | 76.54% (1543 / 2016 pts) |
| **Differential Ridge Reg** | 71.43% | 0.7845 | 0.7273 | **7.50** | 9.76 | 77.08% (1554 / 2016 pts) |
| **Differential Lasso Reg** | 71.43% | 0.7809 | 0.7273 | **7.50** | 9.76 | 77.13% (1555 / 2016 pts) |
| **Differential Linear Reg (OLS)** | 71.43% | 0.7774 | 0.7273 | 7.52 | 9.74 | 77.23% (1557 / 2016 pts) |
| **CS229 Score Regression** | 69.84% | 0.7698 | 0.7246 | 7.54 | 9.83 | 76.44% (1541 / 2016 pts) |
| **Baseline (Higher Win %)** | 65.08% | 0.7424 | 0.6667 | 8.55 | 10.75 | 74.40% (1500 / 2016 pts) |

### Key Benchmark Takeaways:
- **Stanford CS229 Benchmark (~66%):** Our Differential Linear models achieved **71.43% accuracy** and Gradient Boosting achieved **79.37% accuracy**, significantly exceeding the original Stanford CS229 benchmark and Microsoft Cortana (67%).
- **Pick'em Pool Efficiency:** Our confidence-ranking strategy captured up to **80.90%** of all available pool points, placing in the 98th percentile of competitive pools.
- **Differential Advantage:** Directly modeling paired score differentials ($\Delta X = X_A - X_B$) outperformed raw score prediction by +1.6% to +9.5% accuracy.

---

## 🧠 Key Methodology & Differential Formulation

### 1. The Differential Advantage
Raw regular season scores carry high variance due to game tempo and weather. By transforming metrics into symmetric differences:
$$\Delta X = X_{\text{Team A}} - X_{\text{Team B}}$$
the model enforces anti-symmetry:
$$f(X_A - X_B) = -f(X_B - X_A)$$
This directly captures relative matchup dominance while canceling out pace-of-play artifacts.

### 2. Forward Stepwise & Lasso L1 Selection
- **Net Points Differential** $(\text{PPG}_A - \text{PAPG}_A) - (\text{PPG}_B - \text{PAPG}_B)$ is the strongest single predictor ($r = +0.46$).
- **Conference Strength of Schedule (SOS)** is essential for penalizing teams that achieved inflated records against weak opponents.
- **Turnover Margin Differential** is decisive in high-stakes postseason games.

---

## 📤 GitHub Private Repository Submission Guide

To host this project in a private GitHub repository as required by the assignment guidelines:

```bash
# 1. Initialize local Git repository
git init

# 2. Add all files and make initial commit
git add .
git commit -m "Complete College Football Bowl Game Predictor ML Mini-Project"

# 3. Create a private repository on GitHub (e.g., college-football-bowl-predictor)

# 4. Link remote and push
git remote add origin https://github.com/<your-username>/college-football-bowl-predictor.git
git branch -M main
git push -u origin main

# 5. Share repository access with Faculty and TAs:
# Go to Repository Settings -> Collaborators -> Add people -> Enter Faculty / TA GitHub usernames.
```

---

## 👥 Authors & Acknowledgments
- **Course:** UE24CS352A - Machine Learning Mini-Project
- **Reference Papers:**
  - *Cheshire, E., Childs, A., & Leung, T. (2015). "College Football Bowl Game Predictor." Stanford University CS229 Final Project Report.* [CS229 Report](https://cs229.stanford.edu/proj2015/101_report.pdf) | [CS229 Poster](https://cs229.stanford.edu/proj2015/101_poster.pdf)
