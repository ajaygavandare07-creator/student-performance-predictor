# src/preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess(path='data/student_data.csv', test_size=0.2):
    df = pd.read_csv(path)
    X = df.drop(columns=['G1', 'G2', 'G3'])
    X = pd.get_dummies(X, drop_first=True, dtype=int)
    y = (df['G3'] < 10).astype(int)
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)