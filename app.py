# app.py
import json, joblib, numpy as np, pandas as pd, streamlit as st
from pathlib import Path
import math
from typing import List, Dict
from lifelines import CoxPHFitter  # needed to load the Cox model

# --------- Paths ----------
ROOT      = Path(".")
ASSETS    = ROOT / "assets"
FIGS      = ASSETS / "figures"
TITLE_ICON= ASSETS / "melanoma_logo.svg"
ART_DIR   = ROOT / "artifacts"          # <- metastasis + prognosis artifacts both here now

# --------- Page config ----------
st.set_page_config(page_title="Melanoma Calculator", page_icon=str(TITLE_ICON), layout="centered")

with st.sidebar.expander("⚙️ Environment versions"):
    import sklearn, numpy, pandas, joblib, scipy, importlib
    def ver(m): 
        try: return importlib.import_module(m).__version__
        except: return "not installed"
    st.write({
        "scikit-learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "joblib": joblib.__version__,
        "scipy": scipy.__version__,
        "interpret": ver("interpret"),
        "lifelines": ver("lifelines"),
        "streamlit": ver("streamlit"),
    })

# --------- Load metastasis model + schema + thresholds ----------
MODEL_BUNDLE = joblib.load(ART_DIR / "metastasis_pipeline.joblib")
PIPE       = MODEL_BUNDLE["pipeline"]
FEATURES   = MODEL_BUNDLE["feature_order"]
MODEL_NAME = MODEL_BUNDLE.get("model_name", "Model")

SCHEMA = json.loads((ART_DIR / "input_schema.json").read_text(encoding="utf-8"))
APP_CFG = {}
cfg_path = ART_DIR / "app_thresholds.json"
if cfg_path.exists():
    APP_CFG = json.loads(cfg_path.read_text(encoding="utf-8"))
THR_90 = float(APP_CFG.get("thr_90sens", 0.02))  # fallback

# --------- Load prognosis (Cox) artifacts (if available) ----------
COX = None
COX_FEATURES: List[str] = []
PROG_META = {}
try:
    prog_bundle_path = ART_DIR / "cox_os_lifelines.joblib"  # <- moved up one level
    prog_meta_path   = ART_DIR / "cox_meta.json"            # <- moved up one level
    if prog_bundle_path.exists():
        prog_bundle = joblib.load(prog_bundle_path)
        COX = prog_bundle["model"]                   # lifelines CoxPHFitter
        COX_FEATURES = prog_bundle["feature_names"]  # design-matrix cols
    if prog_meta_path.exists():
        PROG_META = json.loads(prog_meta_path.read_text(encoding="utf-8"))
except Exception as e:
    st.warning(f"Prognosis artifacts present but failed to load ({e}). The OS calculator will be hidden.")

# ================= Allowed values (observed) =================
BRESLOW_ALLOWED = sorted([
    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
    1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,
    2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,
    3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,
    4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,
    5.0,5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,
    6.0,6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9,
    7.0,7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,
    8.0,8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8,8.9,
    9.0,9.1,9.2,9.3,9.4,9.5,9.6,9.7,9.8
])

MITOTIC_OPTIONS = [
    ("Unknown (not documented)", -1.0),
    ("Ordered, results not in chart", -2.0),
    ("Described with non-mm² denominator", -3.0),
    ("0", 0.0), ("0.5", 0.5), ("1", 1.0), ("1.5", 1.5),
    ("2", 2.0), ("3", 3.0), ("4", 4.0), ("5", 5.0),
    ("6", 6.0), ("7", 7.0), ("8", 8.0), ("9", 9.0),
    ("10", 10.0), ("≥11", 12.0),
]
MITOTIC_LABELS = [lbl for (lbl, _) in MITOTIC_OPTIONS]
MITOTIC_MAP    = dict(MITOTIC_OPTIONS)

LATERALITY_UI_ORDER = [
    "Left", "Right", "Midline", "Bilateral", "Side unspecified",
    "Not a paired site", "One side (unspecified)"
]

T_STAGE_DOC = """
**How T-staging maps to thickness & ulceration (melanoma)**

| T | Thickness (mm) | Ulceration |
|---|----------------|------------|
| **Tis** | In situ (epidermis only) | — |
| **T1a** | **< 0.8** | No |
| **T1b** | **< 0.8** with ulceration **OR** **0.8–1.0** (± ulceration) | See rule |
| **T2a** | **1.01–2.0** | No |
| **T2b** | **1.01–2.0** | Yes |
| **T3a** | **2.01–4.0** | No |
| **T3b** | **2.01–4.0** | Yes |
| **T4a** | **> 4.0** | No |
| **T4b** | **> 4.0** | Yes |

**Notes**
- “a” vs “b” subcategories are defined by **ulceration** (that’s why ulceration appears in the table).
"""

# ================= Helpers =================
def tstage_selector() -> float:
    mapping = {
        "Tis": 0.5,
        "T1": 1.0, "T1a": 1.1, "T1b": 1.2,
        "T2": 2.0, "T2a": 2.1, "T2b": 2.2,
        "T3": 3.0, "T3a": 3.1, "T3b": 3.2,
        "T4": 4.0, "T4a": 4.1, "T4b": 4.2
    }
    label = st.selectbox("T stage", list(mapping.keys()), index=1)
    with st.expander("What do the T stages mean?"):
        st.markdown(T_STAGE_DOC)
    return mapping[label]

def age_args_int() -> dict:
    d = SCHEMA["numeric"]["age"]
    min_v = int(max(19, math.ceil(d["min"])))
    max_v = int(math.floor(d["max"]))
    return dict(min_value=min_v, max_value=max_v, step=1)

def snap_to_allowed(x: float, allowed: List[float]) -> float:
    arr = np.asarray(allowed, dtype=float)
    idx = int(np.abs(arr - float(x)).argmin())
    return float(arr[idx])

def label_if_adjusted(original: float, snapped: float) -> str:
    return "" if abs(original - snapped) < 1e-9 else f"Adjusted to nearest observed value: **{snapped:g}**"

def extract_dummy_options(prefix: str) -> Dict[str, str]:
    found = [c for c in FEATURES if c.startswith(prefix)]
    mapping: Dict[str, str] = {}
    for col in found:
        label = col[len(prefix):].replace("_", " ")
        mapping[label] = col
    return mapping

def laterality_ui_mapping() -> Dict[str, str]:
    raw = extract_dummy_options("laterality_")
    alias_pairs = [
        ("Left", "Left - origin of primary"),
        ("Right", "Right - origin of primary"),
        ("Midline", "Paired site: midline tumor"),
        ("Bilateral", "Bilateral, single primary"),
        ("Side unspecified", "Only one side - side unspecified"),
        ("Not a paired site", "Not a paired  site"),
        ("One side (unspecified)", "Only one side - side unspecified"),
    ]
    ui_map: Dict[str, str] = {}
    for pretty, raw_key in alias_pairs:
        match = [col for label, col in raw.items() if raw_key in label]
        if match:
            ui_map[pretty] = match[0]
    for label, col in raw.items():
        if col not in ui_map.values():
            ui_map[label] = col
    return ui_map

def histology_ui_mapping() -> Dict[str, str]:
    raw = extract_dummy_options("icdo3_histbehav_")
    ui_map: Dict[str, str] = {}
    for label, col in raw.items():
        pretty = label.split(": ", 1)[1] if ": " in label else label
        ui_map[pretty] = col
    return dict(sorted(ui_map.items(), key=lambda kv: kv[0].lower()))

# ================= Header =================
left, right = st.columns([1, 6], vertical_alignment="center")
with left:
    st.image(str(TITLE_ICON), width=96)
with right:
    st.title("Melanoma Distant Metastasis and Prognostic Calculator")
st.caption("Research/education purposes only")

with st.expander("About this tool"):
    st.markdown("""
Estimates probability of **distant metastasis at diagnosis** using an
**Explainable Boosting Machine (EBM)** trained on SEER-derived variables.
Also provides an **overall-survival (OS) prognosis** calculator (Cox model) if trained artifacts are present.
""")

# ================= Tabs (requested order) =================
tabs = st.tabs(["Metastasis (Distant)", "Prognosis (OS)", "Figures", "Methods"])

# ================= Metastasis (Distant) =================
with tabs[0]:
    with st.form("inputs"):
        st.subheader("Tumor characteristics")
        c1, c2 = st.columns(2)

        breslow_in = c1.number_input(
            "Breslow thickness (mm)",
            min_value=float(BRESLOW_ALLOWED[0]),
            max_value=float(BRESLOW_ALLOWED[-1]),
            step=0.1,
            value=0.6
        )

        mito_label = c2.selectbox(
            "Mitotic rate (/mm²)*",
            MITOTIC_LABELS,
            index=MITOTIC_LABELS.index("0"),
            help="Special codes: Unknown (−1), Ordered/no result (−2), non-mm² denominator (−3)."
        )
        mito_val = float(MITOTIC_MAP[mito_label])

        tnum = tstage_selector()
        ulcer = st.selectbox("Ulceration", ["Present", "Absent"], index=0)

        # Laterality
        lat_map = laterality_ui_mapping()
        lat_labels = [lab for lab in LATERALITY_UI_ORDER if lab in lat_map] + \
                     [lab for lab in lat_map.keys() if lab not in LATERALITY_UI_ORDER]
        laterality_ui = st.selectbox("Laterality", lat_labels, index=0)

        # Histology
        hist_map = histology_ui_mapping()
        hist_labels = list(hist_map.keys())
        hist_idx = hist_labels.index("Superficial spreading melanoma") if "Superficial spreading melanoma" in hist_labels else 0
        hist_ui = st.selectbox("Histology (ICD-O-3)", hist_labels, index=hist_idx)

        st.markdown("---")

        st.subheader("Demographics")
        d1, d2 = st.columns(2)
        age = d1.number_input("Age (years)", **age_args_int(), value=60)
        sex = d2.selectbox("Sex", ["Male", "Female"], index=0)
        race = d1.selectbox("Race", ["Caucasian", "Non-Caucasian"], index=0)
        marital = d2.selectbox("Marital status", ["Married", "Not married"], index=0)

        submitted = st.form_submit_button("Estimate Risk")

    if submitted:
        breslow_snapped = snap_to_allowed(float(breslow_in), BRESLOW_ALLOWED)

        adj_msgs = []
        msg = label_if_adjusted(float(breslow_in), breslow_snapped)
        if msg: adj_msgs.append(f"• Breslow: {msg}")
        if adj_msgs:
            st.info("Input normalization applied:\n\n" + "\n".join(adj_msgs))

        row = {f: 0.0 for f in FEATURES}
        if "age" in row: row["age"] = int(age)
        if "breslow_thickness" in row: row["breslow_thickness"] = breslow_snapped
        if "mitotic_rate" in row: row["mitotic_rate"] = mito_val
        if "t_stage" in row: row["t_stage"] = float(tnum)

        if "sex_Male" in row: row["sex_Male"] = 1.0 if sex == "Male" else 0.0
        if "race_White" in row: row["race_White"] = 1.0 if race == "Caucasian" else 0.0
        mkey = "marital_status_Married (including common law)"
        if mkey in row: row[mkey] = 1.0 if marital == "Married" else 0.0
        ukey = "ulceration_Ulceration present"
        if ukey in row: row[ukey] = 1.0 if ulcer == "Present" else 0.0

        # laterality one-hots
        for c in FEATURES:
            if c.startswith("laterality_"): row[c] = 0.0
        sel_lat_col = lat_map.get(laterality_ui)
        if sel_lat_col in row: row[sel_lat_col] = 1.0

        # histology one-hots
        for c in FEATURES:
            if c.startswith("icdo3_histbehav_"): row[c] = 0.0
        sel_hist_col = hist_map.get(hist_ui)
        if sel_hist_col in row: row[sel_hist_col] = 1.0

        X_one = pd.DataFrame([row], columns=FEATURES)

        prob = float(PIPE.predict_proba(X_one)[0, 1])
        pct  = prob * 100.0

        bucket = ("Very low" if prob < THR_90 else
                  "Low"       if pct < 15  else
                  "Moderate"  if pct < 40  else
                  "Elevated"  if pct < 60  else
                  "High")

        st.success(f"Estimated probability of **distant metastasis at diagnosis**: **{pct:.1f}%** ({bucket})")
        with st.expander("Details (model inputs)"):
            st.dataframe(X_one)

# ================= Prognosis (OS) — Calculator only =================
with tabs[1]:
    st.markdown("### Overall Survival (Cox model)")
    if COX is None or not COX_FEATURES:
        st.info("Prognosis calculator unavailable. Train it with: `python train_prognosis_cox.py`.")
    else:
        with st.form("prognosis_form"):
            # Tumor Characteristics
            st.subheader("Tumor Characteristics")
            tc1, tc2 = st.columns(2)
            breslow_cat_p = tc1.selectbox("Breslow category", ["0 (≤1)","1 (>1–2)","2 (>2–4)","3 (>4)"], index=0)
            ulcer_p = tc2.selectbox("Ulceration", ["Absent","Present"], index=0)
            mitotic_cat_p = tc1.selectbox("Mitotic rate", ["0 (0)","1 (>0–<1)","2 (1–5)","3 (>5)"], index=0)
            T_cat_p = tc2.selectbox("T stage (collapsed)", ["0","1","2","3","4"], index=1)
            N_cat_p = tc1.selectbox("N stage (collapsed)", ["0","1","2","3"], index=0)

            st.markdown("---")

            # Demographics
            st.subheader("Demographics")
            d1, d2 = st.columns(2)
            age_p = d1.number_input("Age (years)", min_value=19, max_value=110, step=1, value=60)
            sex_p = d2.selectbox("Sex", ["Male","Female"], index=0)
            marital_p = d1.selectbox("Marital status", ["Married","Not married"], index=0)
            income4_p = d2.selectbox("Income quartile (0=lowest → 3=highest)", ["0","1","2","3"], index=1)

            st.markdown("---")

            # Treatment
            st.subheader("Treatment")
            t1, t2 = st.columns(2)
            surg3_p = t1.selectbox("Surgery", ["0 (No)","1 (Procedure)","2 (Unknown)"], index=1)
            radiation_p = t2.selectbox("Radiation", ["0 (No/Unknown)","1 (Yes)"], index=0)
            chemo_p = t1.selectbox("Chemotherapy", ["0 (No/Unknown)","1 (Yes)"], index=0)

            submitted_prog = st.form_submit_button("Estimate OS (1/3/5 years)")

        if submitted_prog:
            vec = {f: 0.0 for f in COX_FEATURES}
            if "age" in vec: vec["age"] = float(age_p)
            if "sex_bin" in vec: vec["sex_bin"] = 1.0 if sex_p == "Male" else 0.0
            if "married_bin" in vec: vec["married_bin"] = 1.0 if marital_p == "Married" else 0.0

            def set_onehot(base: str, level: str):
                col = f"{base}_{level}"
                if col in vec:
                    vec[col] = 1.0

            if income4_p in {"1","2","3"}: set_onehot("income4", income4_p)
            if breslow_cat_p[0] in {"1","2","3"}: set_onehot("breslow_cat", breslow_cat_p[0])
            if "ulceration_std_Present" in vec:
                vec["ulceration_std_Present"] = 1.0 if ulcer_p == "Present" else 0.0
            if mitotic_cat_p[0] in {"1","2","3"}: set_onehot("mitotic_cat", mitotic_cat_p[0])
            if T_cat_p in {"1","2","3","4"}: set_onehot("T_cat", T_cat_p)
            if N_cat_p in {"1","2","3"}: set_onehot("N_cat", N_cat_p)
            if surg3_p[0] in {"1","2"}: set_onehot("surg3", surg3_p[0])
            if "radiation_bin_1" in vec:
                vec["radiation_bin_1"] = 1.0 if radiation_p.startswith("1") else 0.0
            if "chemo_bin_1" in vec:
                vec["chemo_bin_1"] = 1.0 if chemo_p.startswith("1") else 0.0

            X_one_prog = pd.DataFrame([vec], columns=COX_FEATURES)

            try:
                sf_df = COX.predict_survival_function(X_one_prog, times=[1.0, 3.0, 5.0])
                col0 = sf_df.columns[0]
                s1 = float(sf_df.loc[1.0, col0])
                s3 = float(sf_df.loc[3.0, col0])
                s5 = float(sf_df.loc[5.0, col0])
                st.success(f"Estimated Overall Survival: **1y {s1*100:.1f}%**, **3y {s3*100:.1f}%**, **5y {s5*100:.1f}%**")
                if "c_index" in PROG_META:
                    st.caption(f"C-index (training): {PROG_META['c_index']}")
            except Exception as e:
                st.error(f"Failed to compute survival estimates: {e}")

# ================= Figures (EBM + Nomogram image) =================
with tabs[2]:
    st.markdown("### Model performance & example explanation")
    st.image(str(FIGS / "Fig1_ROC_PR_EBM.png"), use_column_width=True,
             caption="AUROC, Precision–Recall, and Confusion Matrices (EBM highlighted)")
    st.caption(
        "AUROC reflects ranking across thresholds; PRC emphasizes the rare positive class; "
        "confusion matrices show performance at selected sensitivity operating points."
    )
    st.image(str(FIGS / "Fig2_SHAP_Force.png"), use_column_width=True,
             caption="Example SHAP/force plot (idx 22)")
    st.markdown("### Nomogram (Overall Survival)")
    st.image(str(FIGS / "Fig3_Nomogram.png"), use_column_width=True,
             caption="Nomogram: 1-, 3-, 5-year Overall Survival")

# ================= Methods =================
with tabs[3]:
    st.markdown("### Methods (summary)")
    st.markdown(f"""
- **Outcome:** Distant metastasis at diagnosis (binary).
- **Model:** Explainable Boosting Machine (EBM), sample-weighted for class imbalance.
- **Inputs in this app:** Age, Sex, Race, Marital status, Laterality, Histology (ICD-O-3 one-hots), Breslow thickness, Mitotic rate, T stage.
- **Risk buckets:** “Very low” uses the saved 90%-sensitivity probability threshold from the test set; above that, categories are based on absolute percent: Low (<15%), Moderate (15–40%), Elevated (40–60%), High (≥60%).
- **Intended use:** Research/education only; not for clinical decision-making.
    """)
    st.markdown("### Glossary: T-staging")
    st.markdown(T_STAGE_DOC)

# ================= Footer =================
st.divider()
st.caption("© Jeff J.H. Kim — Research use only. Consider calibration & external validation before clinical use.")
