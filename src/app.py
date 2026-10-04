"""
🎓 Student Performance Prediction System — Early Warning App
AIML Mini Project
Run from the PROJECT ROOT folder:  streamlit run src/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Student At-Risk Predictor", page_icon="🎓", layout="centered")

# ----------------- Train model ONCE (cached) -----------------
@st.cache_resource
def train_model():
    df = pd.read_csv("data/student_data.csv")
    X = df.drop(columns=["G1", "G2", "G3"])
    X = pd.get_dummies(X, drop_first=True, dtype=int)
    y = (df["G3"] < 10).astype(int)   # 1 = At-Risk (grade F)
    # Logistic Regression: highest recall for catching at-risk students
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    ])
    model.fit(X, y)
    return model, X.columns.tolist(), X

model, feature_cols, X_train_df = train_model()

# ----------------- UI -----------------
st.title("🎓 Student Performance Prediction System")
st.caption("Early-warning system: predict whether a student is at risk of failing (Grade F) "
           "using only early-knowable information — before any exam results exist.")

st.divider()
st.subheader("Enter Student Profile")

col1, col2 = st.columns(2)
with col1:
    age      = st.slider("Age", 15, 22, 17)
    absences = st.slider("Absences this term", 0, 93, 4)
    failures = st.selectbox("Past class failures", [0, 1, 2, 3], index=0)
    studytime = st.selectbox("Weekly study time (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)", [1, 2, 3, 4], index=1)
    goout    = st.slider("Going out with friends (1=low - 5=high)", 1, 5, 3)
with col2:
    medu     = st.selectbox("Mother's education (0=none - 4=higher)", [0, 1, 2, 3, 4], index=2)
    fedu     = st.selectbox("Father's education (0=none - 4=higher)", [0, 1, 2, 3, 4], index=2)
    health   = st.slider("Current health (1=bad - 5=excellent)", 1, 5, 4)
    freetime = st.slider("Free time after school (1=low - 5=high)", 1, 5, 3)
    sex      = st.selectbox("Sex", ["F", "M"])

higher   = st.selectbox("Wants to pursue higher education?", ["yes", "no"], index=0)
internet = st.selectbox("Has internet access at home?", ["yes", "no"], index=0)

# ----------------- Build input row (start from average student, override chosen features) -----------------
input_row = X_train_df.median(numeric_only=True).to_dict()
for c in feature_cols:
    if c not in input_row:
        input_row[c] = 0   # dummy columns default 0

input_row["age"] = age
input_row["absences"] = absences
input_row["failures"] = failures
input_row["studytime"] = studytime
input_row["goout"] = goout
input_row["Medu"] = medu
input_row["Fedu"] = fedu
input_row["health"] = health
input_row["freetime"] = freetime
input_row["sex_M"] = 1 if sex == "M" else 0
input_row["higher_yes"] = 1 if higher == "yes" else 0
input_row["internet_yes"] = 1 if internet == "yes" else 0

X_new = pd.DataFrame([input_row])[feature_cols]

# ----------------- Predict -----------------
if st.button("🔮 Predict", type="primary", use_container_width=True):
    pred = model.predict(X_new)[0]
    prob_at_risk = model.predict_proba(X_new)[0][1]

    st.divider()
    if pred == 1:
        st.error(f"### ⚠️ AT-RISK of failing (Grade F)\n"
                 f"Model confidence: **{prob_at_risk*100:.0f}%** at-risk\n\n"
                 f"**Recommended actions:** counselor follow-up, attendance monitoring, "
                 f"extra tutoring in weak subjects.")
    else:
        st.success(f"### ✅ ON TRACK\n"
                   f"Model confidence: **{(1-prob_at_risk)*100:.0f}%** safe\n\n"
                   f"No immediate intervention needed. Continue monitoring absences and study habits.")

    st.progress(float(np.clip(prob_at_risk, 0.02, 0.98)),
                text=f"At-risk probability: {prob_at_risk*100:.0f}%")

st.divider()
st.caption("Model: Logistic Regression (class-balanced) · Trained on UCI Student Dataset (Math), "
           "395 students · Early-warning design: past grades (G1/G2) intentionally excluded.")
