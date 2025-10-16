# train_model_ebm.py
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from interpret.glassbox import ExplainableBoostingClassifier
import joblib

# ============== CONFIG ==============
DATA_CSV = r"C:\Users\Admin\Desktop\Sync\1. PhD\3. Dermatology Projects\2. SEER Melanoma\1. Metastasis\1. Project Folder\Data\Filtered_Data_Metastasis.csv"

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_BUNDLE_PATH = ARTIFACTS_DIR / "metastasis_pipeline.joblib"
SCHEMA_PATH       = ARTIFACTS_DIR / "input_schema.json"
APP_THRESH_PATH   = ARTIFACTS_DIR / "app_thresholds.json"

MODEL_NAME = "EBM (Explainable Boosting)"

# ============== LOAD DATA ==============
df = pd.read_csv(DATA_CSV)

NUMERIC = ["age", "breslow_thickness", "mitotic_rate", "t_stage"]
CORE_BIN = [
    "sex_Male",
    "race_White",
    "marital_status_Married (including common law)",
    "ulceration_Ulceration present",
]

# Dynamic one-hots
LATERALITY_DUMMIES = [c for c in df.columns if c.startswith("laterality_")]
HISTOLOGY_DUMMIES  = [c for c in df.columns if c.startswith("icdo3_histbehav_")]

print(f"Found laterality columns: {len(LATERALITY_DUMMIES)}")
print(f"  e.g. {LATERALITY_DUMMIES[:5]}")
print(f"Found histology columns: {len(HISTOLOGY_DUMMIES)}")
print(f"  e.g. {HISTOLOGY_DUMMIES[:5]}")

FEATURES = NUMERIC + CORE_BIN + LATERALITY_DUMMIES + HISTOLOGY_DUMMIES
TARGET = "target_distant"

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    raise RuntimeError(f"Required columns missing in CSV: {missing}")

X = df[FEATURES].copy()
y = df[TARGET].astype(int).copy()

mask = X.notna().all(axis=1) & y.notna()
X = X.loc[mask]
y = y.loc[mask]

# ============== SPLIT ==============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ============== PREPROCESSOR ==============
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
preprocess = ColumnTransformer(
    transformers=[("num", num_pipe, NUMERIC)],
    remainder="passthrough",
)

# ============== MODEL (EBM) ==============
ebm = ExplainableBoostingClassifier(
    interactions=10,
    learning_rate=0.05,
    max_leaves=3,
    outer_bags=2,
    random_state=42,
)

neg = int((y_train == 0).sum())
pos = int((y_train == 1).sum())
scale_pos_weight = (neg / pos) if pos > 0 else 1.0
w_train = np.where(y_train.values == 1, scale_pos_weight, 1.0)

pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", ebm),
])

pipe.fit(X_train, y_train, model__sample_weight=w_train)

# ============== EVALUATION ==============
y_prob_test = pipe.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, y_prob_test)
auprc = average_precision_score(y_test, y_prob_test)
brier = brier_score_loss(y_test, np.clip(y_prob_test, 0, 1))
print(f"Test AUROC={auroc:.3f}  AUPRC={auprc:.3f}  Brier={brier:.4f}  N={len(y_test):,}")

fpr, tpr, thr = roc_curve(y_test, y_prob_test)
idx = np.where(tpr >= 0.90)[0]
thr_90sens = float(thr[idx[0]]) if idx.size else 0.02
print(f"[App thresholds] 90% sensitivity threshold = {thr_90sens:.5f}")

# ============== SAVE ARTIFACTS ==============
bundle = {
    "pipeline": pipe,
    "feature_order": FEATURES,
    "model_name": MODEL_NAME,
    # NEW: store the exact one-hot lists so the app can build dropdowns reliably
    "laterality_cols": LATERALITY_DUMMIES,
    "histology_cols": HISTOLOGY_DUMMIES,
}
joblib.dump(bundle, MODEL_BUNDLE_PATH)
print(f"Saved model -> {MODEL_BUNDLE_PATH}")

def num_field(name, series, step):
    a = float(series.min())
    b = float(series.max())
    if name == "age":
        a = max(19.0, a)
        step = 1.0
    return {"min": float(a), "max": float(b), "step": float(step)}

schema = {
    "numeric": {
        "age":               num_field("age", df["age"], 1.0),
        "breslow_thickness": num_field("breslow_thickness", df["breslow_thickness"], 0.1),
        "mitotic_rate":      num_field("mitotic_rate", df["mitotic_rate"], 0.5),
        "t_stage":           num_field("t_stage", df["t_stage"], 0.1),
    }
}
ARTIFACTS_DIR.joinpath("input_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved schema -> {SCHEMA_PATH}")

ARTIFACTS_DIR.joinpath("app_thresholds.json").write_text(json.dumps({"thr_90sens": thr_90sens}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved app thresholds -> {APP_THRESH_PATH}")
