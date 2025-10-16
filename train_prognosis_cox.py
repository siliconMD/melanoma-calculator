# train_prognosis_cox.py
# -----------------------------------------------------------
# Trains a Cox PH overall-survival model in Python (lifelines)
# and saves prognosis artifacts for your Streamlit app:
#   - artifacts/prognosis/cox_os_lifelines.joblib
#   - artifacts/prognosis/cox_meta.json
#
# Requires: lifelines, pandas, numpy, joblib, openpyxl (if reading .xlsx)
# -----------------------------------------------------------

import json
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import joblib

# ========= PATHS (EDIT IF NEEDED) =========
ART_ROOT = Path("artifacts")
PROG_DIR = ART_ROOT / "prognosis"
PROG_DIR.mkdir(parents=True, exist_ok=True)

DATA = r"C:\Users\Admin\Desktop\Sync\1. PhD\3. Dermatology Projects\2. SEER Melanoma\1. Metastasis\1. Project Folder\Data\Filtered_Data_Prognosis.xlsx"

MODEL_OUT = PROG_DIR / "cox_os_lifelines.joblib"
META_OUT  = PROG_DIR / "cox_meta.json"

# ========= LOAD DATA =========
if not Path(DATA).exists():
    raise FileNotFoundError(f"Could not find data file at:\n{DATA}")

print(f"[INFO] Loading data: {DATA}")
if DATA.lower().endswith(".xlsx"):
    df0 = pd.read_excel(DATA)
else:
    df0 = pd.read_csv(DATA)

KEEP = [
    "survival_months","vital_status","cause_death",
    "age","sex","marital_status","income",
    "breslow_thickness","ulceration","mitotic_rate",
    "combined_t_stage","combined_n_stage",
    "surgery_site","radiation_recode","chemotherapy_recode_yes_nounk"
]
have = [c for c in KEEP if c in df0.columns]
missing_req = set(["survival_months","vital_status","age"]) - set(have)
if missing_req:
    raise RuntimeError(f"Missing required columns: {sorted(missing_req)}")

df = df0[have].copy()
print(f"[INFO] Rows loaded: {len(df):,}")

# ========= OUTCOME / TIME =========
df["event"] = df["vital_status"].astype(str).str.startswith("Dead").astype(int)
df["time_years"] = pd.to_numeric(df["survival_months"], errors="coerce") / 12.0
df = df[np.isfinite(df["time_years"])].copy()
print(f"[INFO] After valid time filtering: {len(df):,} rows")

# ========= RECODES =========
# 1) Income -> 4 bins (0..3)
if "income" in df.columns:
    inc = pd.to_numeric(df["income"], errors="coerce")
    keep_inc = np.isfinite(inc)
    df = df.loc[keep_inc].copy()
    inc = inc.loc[df.index]
    df["income4"] = pd.cut(inc, [-np.inf, 3, 7, 11, np.inf], labels=["0","1","2","3"])
else:
    df["income4"] = "0"

# 2) Ulceration -> drop “Not documented/No mention/Pathologist …”; standardize Present/Absent
if "ulceration" in df.columns:
    ulc_raw = df["ulceration"].astype(str)
    rm_pat = r"Not documented|No mention|Pathologist"
    keep_ulc = ~ulc_raw.str.contains(rm_pat, case=False, na=False)
    # filter BOTH df and ulc_raw before building the standardized vector
    df = df.loc[keep_ulc].copy()
    ulc_raw = ulc_raw.loc[df.index]
    df["ulceration_std"] = np.where(
        ulc_raw.str.contains("present", case=False, na=False), "Present", "Absent"
    )
else:
    df["ulceration_std"] = "Absent"

# 3) Breslow -> 4 categories (≤1, >1–2, >2–4, >4) -> 0..3
if "breslow_thickness" in df.columns:
    br = pd.to_numeric(df["breslow_thickness"], errors="coerce")
    keep_br = np.isfinite(br)
    df = df.loc[keep_br].copy()
    br = br.loc[df.index]
    df["breslow_cat"] = pd.cut(br, [-np.inf, 1, 2, 4, np.inf], labels=["0","1","2","3"])
else:
    df["breslow_cat"] = "0"

# 4) Mitotic rate: drop -1,-2,-3; bin → 0..3 (0; >0–<1; 1–5; >5)
if "mitotic_rate" in df.columns:
    mr = pd.to_numeric(df["mitotic_rate"], errors="coerce")
    keep_mr = np.isfinite(mr)
    df = df.loc[keep_mr].copy()
    mr = mr.loc[df.index]
    keep_codes = ~df["mitotic_rate"].isin([-1, -2, -3])
    df = df.loc[keep_codes].copy()
    mr = df["mitotic_rate"].astype(float)
    df["mitotic_cat"] = np.where(mr <= 0, "0",
                          np.where(mr < 1, "1",
                          np.where(mr <= 5, "2", "3")))
else:
    df["mitotic_cat"] = "0"

# 5) T/N: numeric -> floor, clamp (T 0..4, N 0..3)
Traw = pd.to_numeric(df["combined_t_stage"], errors="coerce") if "combined_t_stage" in df.columns else pd.Series(np.nan, index=df.index)
Nraw = pd.to_numeric(df["combined_n_stage"], errors="coerce") if "combined_n_stage" in df.columns else pd.Series(np.nan, index=df.index)
mask_TN = np.isfinite(Traw) & np.isfinite(Nraw)
df = df.loc[mask_TN].copy()
T_floor = np.floor(pd.to_numeric(df["combined_t_stage"], errors="coerce")).clip(0,4).astype(int).astype(str)
N_floor = np.floor(pd.to_numeric(df["combined_n_stage"], errors="coerce")).clip(0,3).astype(int).astype(str)
df["T_cat"] = T_floor
df["N_cat"] = N_floor

# 6) Marital: Married vs Not Married; drop Unknown
if "marital_status" in df.columns:
    m = df["marital_status"].astype(str)
    keep_m = m.str.lower() != "unknown"
    df = df.loc[keep_m].copy()
    m = m.loc[df.index]
    df["married_bin"] = np.where(m.str.contains("Married", case=False, na=False), 1, 0).astype(int)
else:
    df["married_bin"] = 0

# 7) Sex: Male=1, Female=0
if "sex" in df.columns:
    sx = df["sex"].astype(str).str.lower()
    keep_sx = sx.isin(["male","female"])
    df = df.loc[keep_sx].copy()
    sx = sx.loc[df.index]
    df["sex_bin"] = (sx == "male").astype(int)
else:
    df["sex_bin"] = 0

# 8) Surgery site: 0(No), 1(10..90), 2(99)
if "surgery_site" in df.columns:
    sraw = pd.to_numeric(df["surgery_site"], errors="coerce")
    keep_s = np.isfinite(sraw)
    df = df.loc[keep_s].copy()
    sraw = sraw.loc[df.index]
    snew = np.where(sraw == 0, "0",
            np.where(sraw == 99, "2",
            np.where((sraw >= 10) & (sraw <= 90), "1", np.nan)))
    df["surg3"] = snew
    df = df.loc[df["surg3"].notna()].copy()
else:
    df["surg3"] = "0"

# 9) Radiation recode: Yes vs No/Unknown
if "radiation_recode" in df.columns:
    val = df["radiation_recode"].astype(str)
    yes_rad = ["Radiation, NOS  method or source not specified", "Beam radiation"]
    df["radiation_bin"] = np.where(val.isin(yes_rad), "1", "0")
else:
    df["radiation_bin"] = "0"

# 10) Chemo recode: Yes vs No/Unknown
if "chemotherapy_recode_yes_nounk" in df.columns:
    valc = df["chemotherapy_recode_yes_nounk"].astype(str)
    df["chemo_bin"] = np.where(valc.str.fullmatch("Yes", case=False, na=False), "1", "0")
else:
    df["chemo_bin"] = "0"

print(f"[INFO] After recodes: {len(df):,} rows remain")

# ========= DESIGN MATRIX =========
NUMS = []
if "age" in df.columns:
    age_num = pd.to_numeric(df["age"], errors="coerce")
    df = df.loc[np.isfinite(age_num)].copy()
    NUMS.append("age")
else:
    raise RuntimeError("Column 'age' is required for the model.")

BIN_NUM = []
if "sex_bin" in df.columns:       BIN_NUM.append("sex_bin")
if "married_bin" in df.columns:   BIN_NUM.append("married_bin")

CATS = [c for c in ["income4","breslow_cat","ulceration_std","mitotic_cat","T_cat","N_cat","surg3","radiation_bin","chemo_bin"] if c in df.columns]

use_cols = ["time_years","event"] + NUMS + BIN_NUM + CATS
df = df[use_cols].dropna().copy()
print(f"[INFO] After dropping NAs in model cols: {len(df):,} rows")

X_cat = pd.get_dummies(df[CATS], drop_first=True) if CATS else pd.DataFrame(index=df.index)
X_num = df[NUMS + BIN_NUM].astype(float)
X = pd.concat([X_num, X_cat], axis=1)

df_fit = pd.concat([df[["time_years","event"]], X], axis=1)

print(f"[INFO] Final design matrix shape: {X.shape}")
print("[INFO] Columns:")
for c in X.columns:
    print("   -", c)

# ========= FIT COX MODEL =========
cph = CoxPHFitter(penalizer=0.1)
cph.fit(df_fit, duration_col="time_years", event_col="event", show_progress=True)
cindex = float(cph.concordance_index_)
print(f"[OK] Cox model fit. Harrell's C-index = {cindex:.3f}")

# ========= SAVE ARTIFACTS =========
bundle = {
    "model": cph,
    "feature_names": list(X.columns),
    "columns_info": {
        "numeric": NUMS + BIN_NUM,
        "categorical_onehot": list(X_cat.columns) if len(X_cat.columns) else []
    }
}
joblib.dump(bundle, MODEL_OUT)
print(f"[SAVED] {MODEL_OUT}")

meta = {
    "framework": "lifelines",
    "concordance_index": cindex,
    "n_rows": int(len(df)),
    "predictors": {"numeric": NUMS + BIN_NUM, "categorical": CATS}
}
META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"[SAVED] {META_OUT}")

print("\nAll done.\n"
      "Open the Streamlit app and go to the Prognosis tab.\n")
