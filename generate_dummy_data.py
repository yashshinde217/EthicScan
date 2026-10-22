"""
Generate two loan-approval demo datasets and models:

  loan_good   — fair model trained on 80%, evaluated on held-out 20%
  loan_biased — biased model heavily penalising female applicants
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).parent
OUT  = ROOT / "sample_data"
OUT.mkdir(exist_ok=True)

for f in OUT.iterdir():
    f.unlink()

rng = np.random.default_rng(42)
N   = 1000  # larger pool so 20% test set = 200 rows

# ── shared applicant features ─────────────────────────────────────────────────
age              = rng.integers(22, 65, N)
annual_income    = np.clip(rng.normal(55000, 20000, N), 15000, 150000).round(0)
credit_score     = np.clip(rng.normal(680, 80, N), 300, 850).round(0)
loan_amount      = np.clip(rng.normal(18000, 8000, N), 2000, 60000).round(0)
employment_years = np.clip(rng.normal(7, 4, N), 0, 40).round(1)
debt_to_income   = np.clip(rng.normal(0.32, 0.12, N), 0.05, 0.75).round(3)
gender           = rng.choice(["male", "female"], N, p=[0.50, 0.50])

def encode(df):
    """Encode gender with pd.factorize — matches the audit pipeline exactly."""
    d = df.copy()
    d["gender"] = pd.factorize(d["gender"])[0]  # male->0, female->1 (first-seen)
    return d

# =============================================================================
# 1. GOOD MODEL — purely merit-based, no gender signal in labels
# =============================================================================
logit_good = (
    -6.0
    + 0.000030 * annual_income
    + 0.012    * credit_score
    - 0.000010 * loan_amount
    + 0.060    * employment_years
    - 3.5      * debt_to_income
    # no gender term
)
prob_good     = 1 / (1 + np.exp(-logit_good))
approved_good = (rng.random(N) < prob_good).astype(int)

df_good_full = pd.DataFrame({
    "age": age, "annual_income": annual_income, "credit_score": credit_score,
    "loan_amount": loan_amount, "employment_years": employment_years,
    "debt_to_income": debt_to_income, "gender": gender,
    "approved": approved_good,
})

# 80/20 stratified split — model trains on train, CSV saved is test only
train_good, test_good = train_test_split(
    df_good_full, test_size=0.2, random_state=42, stratify=df_good_full["approved"]
)

# add a couple of missing values to the test CSV for validation demo
test_good = test_good.copy()
test_good.iloc[2,  test_good.columns.get_loc("credit_score")]   = np.nan
test_good.iloc[10, test_good.columns.get_loc("annual_income")]  = np.nan

X_tr = encode(train_good.drop(columns="approved"))
X_tr = X_tr.fillna(X_tr.median())

model_good = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42
    )),
])
model_good.fit(X_tr, train_good["approved"])

joblib.dump(model_good, OUT / "loan_good_model.joblib")
test_good.to_csv(OUT / "loan_good.csv", index=False)

X_te = encode(test_good.drop(columns="approved"))
X_te = X_te.fillna(X_te.median())
preds_good  = model_good.predict(X_te)
acc_good    = accuracy_score(test_good["approved"], preds_good)
male_rate_g = preds_good[test_good["gender"].values == "male"].mean()
fem_rate_g  = preds_good[test_good["gender"].values == "female"].mean()
di_good     = fem_rate_g / male_rate_g if male_rate_g > 0 else 0
print(f"[loan_good]   test_rows={len(test_good)}  accuracy={acc_good:.1%}  "
      f"male={male_rate_g:.1%}  female={fem_rate_g:.1%}  DI={di_good:.3f}")

# =============================================================================
# 2. BIASED MODEL — strong penalty against female applicants baked into labels
# =============================================================================
gender_penalty = np.where(gender == "female", -3.2, 0.0)
logit_biased = (
    -5.5
    + 0.000025 * annual_income
    + 0.010    * credit_score
    - 0.000008 * loan_amount
    + 0.050    * employment_years
    - 3.0      * debt_to_income
    + gender_penalty
)
prob_biased     = 1 / (1 + np.exp(-logit_biased))
approved_biased = (rng.random(N) < prob_biased).astype(int)

df_biased_full = pd.DataFrame({
    "age": age, "annual_income": annual_income, "credit_score": credit_score,
    "loan_amount": loan_amount, "employment_years": employment_years,
    "debt_to_income": debt_to_income, "gender": gender,
    "approved": approved_biased,
})

train_biased, test_biased = train_test_split(
    df_biased_full, test_size=0.2, random_state=42, stratify=df_biased_full["approved"]
)

test_biased = test_biased.copy()
test_biased.iloc[2,  test_biased.columns.get_loc("credit_score")]  = np.nan
test_biased.iloc[10, test_biased.columns.get_loc("annual_income")] = np.nan

X_tr_b = encode(train_biased.drop(columns="approved"))
X_tr_b = X_tr_b.fillna(X_tr_b.median())

model_biased = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])
model_biased.fit(X_tr_b, train_biased["approved"])

joblib.dump(model_biased, OUT / "loan_biased_model.joblib")
test_biased.to_csv(OUT / "loan_biased.csv", index=False)

X_te_b      = encode(test_biased.drop(columns="approved"))
X_te_b      = X_te_b.fillna(X_te_b.median())
preds_biased = model_biased.predict(X_te_b)
acc_biased   = accuracy_score(test_biased["approved"], preds_biased)
male_rate_b  = preds_biased[test_biased["gender"].values == "male"].mean()
fem_rate_b   = preds_biased[test_biased["gender"].values == "female"].mean()
di_biased    = fem_rate_b / male_rate_b if male_rate_b > 0 else 0
print(f"[loan_biased] test_rows={len(test_biased)}  accuracy={acc_biased:.1%}  "
      f"male={male_rate_b:.1%}  female={fem_rate_b:.1%}  DI={di_biased:.3f}")
print("\nDone. Files written to sample_data/")
