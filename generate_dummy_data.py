from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).parent
OUT = ROOT / "sample_data"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
n = 240
age = rng.integers(21, 61, n)
income = np.clip(rng.normal(60000, 18000, n), 18000, 130000).round(0)
education_num = rng.integers(8, 17, n)
experience = np.clip(age - 21 + rng.normal(0, 3, n), 0, 40).round(1)
gender = rng.choice([0, 1], n, p=[0.48, 0.52])
# Intentionally introduce a modest gender-linked effect so the audit has something to find.
logit = -7.0 + 0.000045 * income + 0.32 * education_num + 0.045 * experience - 0.85 * (gender == 0)
prob = 1 / (1 + np.exp(-logit))
approved = (rng.random(n) < prob).astype(int)
df = pd.DataFrame({"age": age, "income": income, "education_num": education_num, "experience": experience, "gender": gender, "approved": approved})
df.loc[[7, 53], "income"] = np.nan
df.loc[[19], "education_num"] = np.nan
X = df.drop(columns="approved").copy()
X = X.fillna(X.median())
model = LogisticRegression(max_iter=1000, random_state=42).fit(X, df["approved"])
model_path = OUT / "ethicscan_demo_model.joblib"
data_path = OUT / "ethicscan_demo.csv"
joblib.dump(model, model_path)
df.to_csv(data_path, index=False)
print(f"Created {data_path}")
print(f"Created {model_path}")
print(f"Rows: {len(df)} | positive label rate: {df.approved.mean():.1%}")
