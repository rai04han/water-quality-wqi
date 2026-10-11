"""
Run ONCE before the demo:   python make_demo_models.py
(needs China_dataset.csv in the same folder)

Creates:
  holdout_model.pkl / holdout_scaler.pkl
      Random Forest trained ONLY on 2001-2012 data. The app tests it on 2013-2017 records
      it has never seen -> this is the honest accuracy evidence for the demo.
  best_wqi_model.pkl, scaler.pkl, label_encoder.pkl, feature_ranges.pkl
      The deployed model (trained on all rows). Only created if the notebook has not
      already produced them. Same recipe as the notebook (scale -> SMOTE -> RF, seed 42).
"""
import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

FEATURES = [
    'Ammonia (mg/l)', 'Biochemical Oxygen Demand (mg/l)', 'Dissolved Oxygen (mg/l)',
    'Orthophosphate (mg/l)', 'pH (ph units)', 'Temperature (cel)',
    'Nitrogen (mg/l)', 'Nitrate (mg/l)',
]

df = pd.read_csv('China_dataset.csv')
df['year'] = pd.to_datetime(df['Date'], format='%d-%m-%Y').dt.year
le = LabelEncoder().fit(df['CCME_WQI'])
X = df[FEATURES].values
y = le.transform(df['CCME_WQI'])


def fit_rf(X_, y_):
    scaler = StandardScaler().fit(X_)                                  # scaler learns from the training rows only
    X_bal, y_bal = SMOTE(random_state=42).fit_resample(scaler.transform(X_), y_)
    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', criterion='gini',
                                n_jobs=-1, random_state=42).fit(X_bal, y_bal)
    return rf, scaler


# 1) Held-out-years model: train on 2001-2012 only
train_mask = (df['year'] <= 2012).values
rf_h, sc_h = fit_rf(X[train_mask], y[train_mask])
joblib.dump(rf_h, 'holdout_model.pkl')
joblib.dump(sc_h, 'holdout_scaler.pkl')
print(f"holdout model trained on {train_mask.sum()} rows (2001-2012); "
      f"{(~train_mask).sum()} rows from 2013-2017 stay unseen")

# 2) Deployed model: only if the notebook's files are missing
needed = ['best_wqi_model.pkl', 'scaler.pkl', 'label_encoder.pkl', 'feature_ranges.pkl']
if not all(os.path.exists(f) for f in needed):
    rf, sc = fit_rf(X, y)
    joblib.dump(rf, 'best_wqi_model.pkl')
    joblib.dump(sc, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    joblib.dump(df[FEATURES].agg(['min', 'max']), 'feature_ranges.pkl')
    print("deployed model files created")
else:
    print("deployed model files already exist (from the notebook) - left untouched")
