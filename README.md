# 🎓 Student Performance Prediction System

An end-to-end Machine Learning system that predicts whether a student is **at risk of failing**
before any exam results exist — enabling teachers to intervene early with guidance and support.

> **Input → Student Academic Data → ML Model → Predicted Performance → Early Support**

---

## 📌 Problem Statement

Students often fail courses before teachers or parents realize there is a problem. By the time
final exam results arrive, it is too late to help. This project builds an **early-warning system**
that flags at-risk students using only information available *early in the term* — attendance,
study habits, family background, and lifestyle factors — **without relying on past exam grades**.

## 🎯 Objective

1. Predict student academic performance using machine learning
2. Identify **at-risk students early** so teachers can provide timely guidance
3. Discover which early-knowable factors most strongly indicate failure risk
4. Deliver predictions through a simple, usable web interface

---

## 📊 Dataset

**UCI Student Performance Dataset (Mathematics)** — [UCI ML Repository](https://archive.ics.uci.edu/dataset/320/student+performance)

| Property | Value |
|---|---|
| Students | 395 |
| Features | 33 (demographics, family, school, lifestyle) |
| Raw target | G3 — final grade (0–20) |
| Task A | Multi-class: predict letter grade (A–F) |
| Task B ⭐ | Binary: predict **At-Risk** (G3 < 10, i.e., grade F) |

**Key design decision:** `G1` and `G2` (past period grades) are deliberately **excluded** from
features. They are the strongest predictors of G3 — but an early-warning system is only useful
*before* those grades exist. A bonus experiment (see Results) quantifies exactly what they add.

## 🛠️ Tech Stack

- **Python 3.13** · pandas · NumPy
- **scikit-learn** — Logistic Regression, Decision Tree, Random Forest, GridSearchCV
- **Matplotlib / Seaborn** — EDA visualizations
- **Streamlit** — web application

## 📁 Project Structure

```
student-performance-predictor/
├── data/
│   └── student_data.csv          # UCI Student dataset (Math)
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Loading, target creation, encoding, train/test split
│   ├── 02_model_training.ipynb      # EDA + 3-model comparison + G1/G2 bonus experiment
│   └── 03_improvements.ipynb        # Feature importance, tuning, at-risk binary model
├── src/
│   └── app.py                    # Streamlit web application
└── README.md
```

## 🚀 Getting Started

```bash
# 1. Clone / download this repository, then create a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn streamlit

# 3. Run the notebooks (VS Code: open .ipynb → Select Kernel → run cells top to bottom)
#    notebooks/01 → 02 → 03

# 4. Launch the web app (from the project root folder)
python -m streamlit run src/app.py
# App opens at http://localhost:8501
```

---

## 🔬 Methodology

1. **Data preprocessing** — one-hot encoding of categorical features, stratified 80/20 train/test split
2. **EDA** — correlation heatmap, grade distribution, absences-vs-grade analysis
3. **Multi-class experiments** — predict letter grade (A–F) from early-knowable features only
4. **Binary reframing** — At-Risk detection, the task that matches the real-world use case
5. **Model comparison** — Logistic Regression vs Decision Tree vs Random Forest
6. **Hyperparameter tuning** — GridSearchCV (5-fold CV, scored on **recall** — catching at-risk
   students matters more than overall accuracy for a warning system)
7. **Deployment** — best-recall model packaged as a Streamlit web app

## 📈 Results

### Multi-class grade prediction (A–F, without past grades)

| Model | Accuracy |
|---|---|
| Logistic Regression | 0.253 |
| Decision Tree | 0.241 |
| **Random Forest** | **0.342** |

Exact-grade prediction from lifestyle data alone is genuinely hard — a key finding in itself.

### ⭐ Bonus experiment: the value of past grades

| Random Forest setup | Accuracy |
|---|---|
| Early-warning (no G1/G2) | 0.342 |
| With G1 + G2 | **0.696** |

Past performance roughly doubles accuracy — but arrives too late to serve as an early warning.
This contrast is the central insight of the project.

### At-Risk detection (binary — the deployed task)

| Model | Accuracy | At-Risk Recall (students caught) | False-alarm rate |
|---|---|---|---|
| Logistic Regression | 0.709 | **0.65 (17/26)** ⭐ | — |
| Random Forest | 0.722 | 0.46 (12/26) | — |
| Random Forest (tuned) | 0.709 | 0.50 (13/26) | 0.19 |

**Logistic Regression is deployed in the app**: although Random Forest is marginally more
accurate, a warning system must *catch* at-risk students — missing one defeats the purpose.
LR catches 65% of failing students in advance while flagging only ~19% of safe students.

### Most important early-warning features (Random Forest importance)

1. **Absences** — strongest single signal
2. **Past class failures**
3. **Going out frequency** / age / health / parental education

These are all observable early in the term — and actionable (attendance monitoring, counseling).

---

## 🌐 Web Application

A teacher enters a student profile (age, absences, failures, study time, family education, etc.)
and receives:

- ⚠️ **AT-RISK** flag with recommended interventions (counseling, attendance monitoring, tutoring), or
- ✅ **ON TRACK** confirmation

…along with an at-risk probability bar for transparency.

## ⚠️ Limitations

- Small dataset (395 students, two schools, one country) — generalization is limited
- 65% recall means ~1 in 3 at-risk students may still be missed
- Features are self-reported survey responses (subject to honesty bias)
- Binary at-risk threshold (grade F) — real academic risk is a spectrum

## 🔮 Future Scope

- Larger, multi-school, multi-country datasets
- Adjustable probability threshold (tune sensitivity vs. false alarms per school policy)
- Two-stage system: early-warning model (no grades) + mid-term refinement model (with G1)
- Subject-specific risk models (math vs. languages may show different signals)
- Integration with school ERP / parent notification systems

---

## 👤 Author

Built as an AIML mini project — end-to-end: data preprocessing → model comparison →
hyperparameter tuning → deployment.

*Model: Logistic Regression (class-balanced) · UCI Student Dataset (Math), 395 students ·
Early-warning design: past grades (G1/G2) intentionally excluded.*
