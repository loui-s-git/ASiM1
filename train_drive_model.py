"""train_drive_model.py - trains the drive classifier of the Franka dashboard and writes drive_model.joblib.

    python3 train_drive_model.py

Features are what the robot can measure: gripper opening (= drive thickness, CSV height_mm) and ln(weight).
The fitted sklearn pipeline is saved with joblib; the dashboard loads it and calls predict_proba. Train with the same
scikit-learn version the dashboard runs with (requirements.txt).
"""
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
CLASSES = ["ssd", "hdd25", "hdd35"]                    # row / column order of the confusion matrix
TYPE_TO_CLASS = {"SSD 2.5": "ssd", "HDD 2.5": "hdd25", "HDD 3.5": "hdd35"}

df = pd.read_csv(HERE / "drive_dimensions.csv", usecols=["type", "height_mm", "weight_g"])
X = np.column_stack([df["height_mm"], np.log(df["weight_g"])])
y = df["type"].str.strip().map(TYPE_TO_CLASS).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegressionCV(
    Cs=[0.1, 1, 10, 100], cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="neg_log_loss",
    max_iter=10000))
model.fit(X_train, y_train)
C = float(model[-1].C_[0])                             # chosen on the training set only
pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, pred)
print(f"test accuracy {test_accuracy:.1%} on {len(y_test)} drives (C = {C:g})")
print(pd.DataFrame(confusion_matrix(y_test, pred, labels=CLASSES), index=CLASSES, columns=CLASSES))

model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=10000)).fit(X, y)   # deployed: all drives
joblib.dump(dict(model=model,
                 points=[[float(w), float(g), c] for w, g, c in zip(df["height_mm"], df["weight_g"], y)],
                 note=f"logistic regression C={C:g}, trained {datetime.now():%Y-%m-%d} on {len(df)} drives, "
                      f"test accuracy {test_accuracy:.0%}"), HERE / "drive_model.joblib")
print(f"written {HERE / 'drive_model.joblib'}")
