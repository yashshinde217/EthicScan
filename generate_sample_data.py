from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).parent
OUT = ROOT / "sample_data"
OUT.mkdir(exist_ok=True)

def generate(n: int, seed: int, gender_penalty: float, missing_frac: float, label: str):
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 61, n)
    income = np.clip(rng.normal(60000, 18000, n), 18000, 130000).round(0)
    education_num = rng.integers(8, 17, n)
    experience = np.clip(age - 21 + rng.normal(0, 3, n), 0, 40).round(1)
    gender = rng.choice([0, 1], n, p=[0.48, 0.52])  # 0=female, 1=male
    race = rng.choice([0, 1, 2], n, p=[0.55, 0.30, 0.15])  # 0=group A, 1=B, 2=C

    logit = (
        -7.0
        + 0.000045 * income
        + 0.32 * education_num
        + 0.045 * experience
        + gender_penalty * (gender == 0)   # penalty applied to female group
        + 0.2 * (race == 0)                # slight race effect scales with scenario
    )
    prob = 1 / (1 + np.exp(-logit))
    approved = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "education_num": education_num,
        "experience": experience,
        "gender": gender,
        "race": race,
        "approved": approved,
    })

    # Inject missing values proportional to missing_frac
    n_missing = max(1, int(n * missing_frac))
    missing_idx = rng.choice(n, n_missing, replace=False)
    df.loc[missing_idx[:len(missing_idx)//2], "income"] = np.nan
    df.loc[missing_idx[len(missing_idx)//2:], "education_num"] = np.nan

    X = df.drop(columns="approved").copy()
    X = X.fillna(X.median())
    model = LogisticRegression(max_iter=1000, random_state=seed).fit(X, df["approved"])

    model_path = OUT / f"ethicscan_{label}_model.joblib"
    data_path  = OUT / f"ethicscan_{label}.csv"
    joblib.dump(model, model_path)
    df.to_csv(data_path, index=False)

    pos_rate = df["approved"].mean()
    female_rate = df.loc[df["gender"] == 0, "approved"].mean()
    male_rate   = df.loc[df["gender"] == 1, "approved"].mean()
    di = female_rate / male_rate if male_rate > 0 else 0.0
    dpd = female_rate - male_rate

    print(f"\n[{label.upper()}]")
    print(f"  Rows            : {n}")
    print(f"  Overall approval: {pos_rate:.1%}")
    print(f"  Female approval : {female_rate:.1%}  |  Male approval: {male_rate:.1%}")
    print(f"  Disparate Impact: {di:.3f}  (threshold < 0.80 = HIGH risk)")
    print(f"  Dem. Parity Diff: {dpd:+.3f}  (threshold |x| > 0.10 = MEDIUM risk)")
    print(f"  Missing values  : {df.isna().sum().sum()}")
    print(f"  Saved -> {data_path.name}  +  {model_path.name}")


# ── 1. NORMAL — minimal bias, disparate impact well above 0.80 ────────────────
generate(
    n=500,
    seed=42,
    gender_penalty=-0.10,   # tiny penalty
    missing_frac=0.01,
    label="normal",
)

# ── 2. MODERATE — noticeable bias, disparate impact around 0.70–0.79 ─────────
generate(
    n=500,
    seed=7,
    gender_penalty=-0.85,   # same as original demo
    missing_frac=0.03,
    label="moderate",
)

# ── 3. TOO RISKY — severe bias, disparate impact well below 0.60 ─────────────
generate(
    n=500,
    seed=99,
    gender_penalty=-2.50,   # strong penalty on female group
    missing_frac=0.06,
    label="risky",
)

print("\nDone. Upload any pair to EthicScan:")
print("  Normal   -> ethicscan_normal_model.joblib   + ethicscan_normal.csv")
print("  Moderate -> ethicscan_moderate_model.joblib + ethicscan_moderate.csv")
print("  Risky    -> ethicscan_risky_model.joblib    + ethicscan_risky.csv")
print("Use 'approved' as target and 'gender' (or 'race') as sensitive attribute.")
