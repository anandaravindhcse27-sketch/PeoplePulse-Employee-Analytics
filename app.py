import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)

# ----------------------------- PAGE CONFIG ------------------------------------
st.set_page_config(
    page_title="PeoplePulse | Employee Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------- PROFESSIONAL UI -------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#14213a; --muted:#718096; --blue:#2563eb; --cyan:#06b6d4; --violet:#7c3aed; }
    html, body, [class*="css"] { font-family: Manrope, Inter, sans-serif; }
    .stApp { background: radial-gradient(circle at 12% 5%, #e8f1ff 0, transparent 28%), radial-gradient(circle at 88% 18%, #f1eaff 0, transparent 24%), #f5f7fc; }
    [data-testid="stHeader"] { background: rgba(245,247,252,.78); backdrop-filter: blur(14px); }
    [data-testid="stSidebar"] { background: linear-gradient(165deg,#0b1222 0%,#111d35 58%,#172b4d 100%); border-right:1px solid rgba(148,163,184,.14); }
    [data-testid="stSidebar"] * { color:#e8eef8 !important; }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] { background:rgba(255,255,255,.06); border:1px solid rgba(148,163,184,.22); border-radius:16px; padding:10px; }
    .block-container { padding-top:1.35rem; padding-bottom:2.6rem; max-width:1540px; animation: pageIn .7s ease both; }
    @keyframes pageIn { from {opacity:0; transform:translateY(12px)} to {opacity:1; transform:translateY(0)} }
    @keyframes gradientFlow { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
    @keyframes heroReveal { from {opacity:0; transform:translateY(-14px) scale(.99)} to {opacity:1; transform:translateY(0) scale(1)} }
    @keyframes cardRise { from {opacity:0; transform:translateY(18px)} to {opacity:1; transform:translateY(0)} }
    @keyframes softPulse { 0%,100%{box-shadow:0 10px 28px rgba(37,99,235,.12)} 50%{box-shadow:0 16px 38px rgba(124,58,237,.2)} }
    .hero { position:relative; overflow:hidden; background:linear-gradient(115deg,#0b1222,#173d72,#2563eb,#6d28d9); background-size:240% 240%; animation:heroReveal .8s ease both, gradientFlow 16s ease infinite; border-radius:24px; padding:29px 32px; color:white; margin-bottom:23px; box-shadow:0 18px 42px rgba(30,64,175,.18); }
    .hero:before { content:""; position:absolute; width:310px;height:310px;right:-75px;top:-155px;border-radius:50%;border:1px solid rgba(255,255,255,.16);box-shadow:0 0 0 28px rgba(255,255,255,.035),0 0 0 58px rgba(255,255,255,.025); }
    .hero:after { content:""; position:absolute; width:160px;height:160px;right:18%;bottom:-125px;border-radius:50%;background:rgba(255,255,255,.1);filter:blur(2px); }
    .hero h1 { color:white; font-size:clamp(26px,3vw,36px); font-weight:800; letter-spacing:-1.1px; margin:0 0 9px; position:relative; z-index:1; }
    .hero p { color:#dbeafe; margin:0; font-size:14px; max-width:850px; position:relative; z-index:1; }
    .hero-tag { display:inline-block; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.22); border-radius:999px; padding:6px 12px; font-size:11px; letter-spacing:.8px; margin-bottom:13px; backdrop-filter:blur(8px); position:relative; z-index:1; }
    .section-title { font-size:22px; font-weight:800; color:var(--ink); margin:7px 0 3px; letter-spacing:-.4px; }
    .section-subtitle { color:var(--muted); font-size:13px; margin-bottom:16px; }
    div[data-testid="stMetric"] { background:rgba(255,255,255,.88); border:1px solid rgba(226,232,240,.95); padding:17px 19px; border-radius:18px; box-shadow:0 8px 24px rgba(15,23,42,.045); backdrop-filter:blur(10px); animation:cardRise .65s ease both; transition:transform .22s ease, box-shadow .22s ease, border-color .22s ease; }
    div[data-testid="stMetric"]:hover { transform:translateY(-4px); border-color:#bfdbfe; box-shadow:0 15px 32px rgba(37,99,235,.11); }
    div[data-testid="stMetricLabel"] { color:#64748b; font-weight:700; }
    div[data-testid="stMetricValue"] { color:#14213a; font-weight:800; letter-spacing:-.8px; }
    div[data-testid="stTabs"] button { border-radius:11px 11px 0 0; font-weight:700; padding:11px 16px; transition:background .2s ease, color .2s ease; }
    div[data-testid="stTabs"] button:hover { background:#eaf1ff; color:#1d4ed8; }
    div[data-testid="stVerticalBlockBorderWrapper"] { background:rgba(255,255,255,.9); border:1px solid #e6ebf2; border-radius:17px; padding:13px 16px; box-shadow:0 7px 22px rgba(16,26,45,.035); }
    [data-testid="stPlotlyChart"] { animation:cardRise .7s ease both; border-radius:16px; overflow:hidden; transition:transform .22s ease, box-shadow .22s ease; }
    [data-testid="stPlotlyChart"]:hover { transform:translateY(-2px); box-shadow:0 12px 28px rgba(15,23,42,.07); }
    [data-testid="stDataFrame"] { border:1px solid #e2e8f0; border-radius:13px; overflow:hidden; }
    .small-note { color:#7b8798; font-size:12px; }
    .stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button { border-radius:10px; font-weight:700; transition:transform .18s ease, box-shadow .18s ease; }
    .stButton button:hover, .stDownloadButton button:hover, [data-testid="stFormSubmitButton"] button:hover { transform:translateY(-2px); box-shadow:0 8px 18px rgba(37,99,235,.15); }
    input, textarea { border-radius:10px !important; }
    footer { visibility:hidden; }
    /* Clearly visible continuous motion */
    @keyframes auroraMove { 0% { background-position:0% 50%; } 50% { background-position:100% 50%; } 100% { background-position:0% 50%; } }
    @keyframes orbitGlow { 0% { transform:translate(0,0) scale(1); opacity:.45; } 50% { transform:translate(-34px,22px) scale(1.12); opacity:.85; } 100% { transform:translate(0,0) scale(1); opacity:.45; } }
    @keyframes livePulse { 0% { box-shadow:0 0 0 0 rgba(52,211,153,.65); } 70% { box-shadow:0 0 0 10px rgba(52,211,153,0); } 100% { box-shadow:0 0 0 0 rgba(52,211,153,0); } }
    @keyframes shimmer { 0% { transform:translateX(-130%); } 100% { transform:translateX(130%); } }
    @keyframes floatUp { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-7px); } }
    .stApp { background-size:220% 220%; animation:auroraMove 22s ease infinite; }
    .hero { background-size:300% 300%; animation:heroReveal .8s ease both, auroraMove 9s ease infinite; }
    .hero:before { animation:orbitGlow 7s ease-in-out infinite; }
    .hero:after { animation:floatUp 5s ease-in-out infinite; }
    .hero-live { display:inline-flex; align-items:center; gap:8px; margin-top:17px; padding:7px 12px; border-radius:999px; background:rgba(4,18,40,.28); border:1px solid rgba(255,255,255,.2); color:#e0f2fe; font-size:12px; letter-spacing:.2px; position:relative; z-index:2; }
    .live-dot { width:8px; height:8px; border-radius:50%; background:#34d399; animation:livePulse 1.8s infinite; }
    div[data-testid="stMetric"] { position:relative; overflow:hidden; }
    div[data-testid="stMetric"]:after { content:""; position:absolute; top:0; bottom:0; left:0; width:38%; background:linear-gradient(100deg,transparent,rgba(255,255,255,.62),transparent); transform:translateX(-130%); pointer-events:none; }
    div[data-testid="stMetric"]:hover:after { animation:shimmer 1.05s ease; }
    [data-testid="stPlotlyChart"] { animation:cardRise .7s ease both; }
    [data-testid="stPlotlyChart"]:hover { transform:translateY(-3px) scale(1.003); box-shadow:0 16px 34px rgba(37,99,235,.13); }
    .motion-strip { height:5px; border-radius:999px; margin:0 0 22px; overflow:hidden; background:#dbeafe; }
    .motion-strip span { display:block; width:42%; height:100%; border-radius:999px; background:linear-gradient(90deg,#06b6d4,#2563eb,#8b5cf6,#ec4899); background-size:220% 100%; animation:stripFlow 3.2s linear infinite; }
    @keyframes stripFlow { 0% { transform:translateX(-110%); background-position:0% 50%; } 100% { transform:translateX(260%); background-position:100% 50%; } }
    /* HIGH-MOTION PREMIUM MODE: always-on visual motion */
    @keyframes auroraSweep { 0%{background-position:0% 0%} 50%{background-position:100% 100%} 100%{background-position:0% 0%} }
    @keyframes orbDriftA { 0%,100%{transform:translate3d(0,0,0) scale(1)} 35%{transform:translate3d(9vw,7vh,0) scale(1.16)} 70%{transform:translate3d(-5vw,12vh,0) scale(.9)} }
    @keyframes orbDriftB { 0%,100%{transform:translate3d(0,0,0) scale(1)} 40%{transform:translate3d(-10vw,-5vh,0) scale(1.2)} 75%{transform:translate3d(4vw,8vh,0) scale(.88)} }
    @keyframes borderRun { 0%{background-position:0% 50%} 100%{background-position:200% 50%} }
    @keyframes titleGlow { 0%,100%{text-shadow:0 0 0 rgba(255,255,255,0)} 50%{text-shadow:0 0 24px rgba(147,197,253,.55)} }
    @keyframes scanLine { 0%{transform:translateY(-120%);opacity:0} 15%{opacity:.5} 85%{opacity:.35} 100%{transform:translateY(360%);opacity:0} }
    @keyframes chartFloat { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-5px)} }
    @keyframes pulseRing { 0%{transform:scale(.75);opacity:.7} 100%{transform:scale(1.8);opacity:0} }
    @keyframes dotBlink { 0%,100%{opacity:.35} 50%{opacity:1} }
    .stApp { background:linear-gradient(125deg,#f4f8ff,#f7f3ff,#eefaff,#f5f7fc); background-size:350% 350%; animation:auroraSweep 24s ease-in-out infinite; }
    .stApp:before,.stApp:after { content:""; position:fixed; z-index:0; pointer-events:none; width:34vw;height:34vw;max-width:520px;max-height:520px;border-radius:50%;filter:blur(55px);opacity:.22; }
    .stApp:before { left:-12vw;top:18vh;background:radial-gradient(circle,#60a5fa 0%,#8b5cf6 42%,transparent 70%);animation:orbDriftA 18s ease-in-out infinite; }
    .stApp:after { right:-12vw;bottom:5vh;background:radial-gradient(circle,#22d3ee 0%,#a78bfa 42%,transparent 72%);animation:orbDriftB 22s ease-in-out infinite; }
    [data-testid="stAppViewContainer"] > .main { position:relative;z-index:1; }
    .hero { border:1px solid rgba(255,255,255,.22); box-shadow:0 22px 55px rgba(37,99,235,.25), inset 0 1px 0 rgba(255,255,255,.22); }
    .hero h1 { animation:titleGlow 4.5s ease-in-out infinite; }
    .hero-tag { animation:floatUp 4s ease-in-out infinite; }
    .hero-live { position:relative; }
    .live-dot { position:relative; }
    .live-dot:after { content:"";position:absolute;inset:-5px;border:1px solid #34d399;border-radius:50%;animation:pulseRing 1.8s ease-out infinite; }
    .motion-strip { height:6px;box-shadow:0 0 18px rgba(59,130,246,.22); }
    div[data-testid="stMetric"] { animation:cardRise .8s cubic-bezier(.2,.75,.25,1) both, softPulse 6s ease-in-out infinite; }
    div[data-testid="stMetric"]:nth-child(2n) { animation-delay:.18s,1.1s; }
    div[data-testid="stMetric"]:nth-child(3n) { animation-delay:.32s,2.1s; }
    div[data-testid="stMetric"]:hover { transform:translateY(-8px) scale(1.025); box-shadow:0 20px 42px rgba(37,99,235,.2); }
    div[data-testid="stMetric"]:before { content:"";position:absolute;inset:0;padding:1px;border-radius:18px;background:linear-gradient(120deg,transparent,#93c5fd,#c4b5fd,#67e8f9,transparent);background-size:250% 100%;animation:borderRun 7s linear infinite; -webkit-mask:linear-gradient(#fff 0 0) content-box,linear-gradient(#fff 0 0);-webkit-mask-composite:xor;mask-composite:exclude;pointer-events:none; }
    [data-testid="stPlotlyChart"] { position:relative;animation:cardRise .8s ease both,chartFloat 7s ease-in-out infinite; }
    [data-testid="stPlotlyChart"]:after { content:"";position:absolute;top:0;left:0;right:0;height:35%;background:linear-gradient(180deg,rgba(96,165,250,.08),transparent);pointer-events:none;animation:scanLine 8s ease-in-out infinite; }
    div[data-testid="stVerticalBlockBorderWrapper"] { transition:transform .25s ease,box-shadow .25s ease; }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover { transform:translateY(-3px);box-shadow:0 16px 34px rgba(37,99,235,.10); }
    div[data-testid="stTabs"] button[aria-selected="true"] { position:relative;color:#2563eb!important; }
    div[data-testid="stTabs"] button[aria-selected="true"]:after { content:"";position:absolute;bottom:0;left:12%;width:76%;height:3px;border-radius:3px;background:linear-gradient(90deg,#06b6d4,#2563eb,#8b5cf6);background-size:200% 100%;animation:borderRun 2.5s linear infinite; }
    .section-title { background:linear-gradient(90deg,#14213a,#2563eb,#7c3aed,#14213a);background-size:250% auto;background-clip:text;-webkit-background-clip:text;color:transparent;animation:borderRun 9s linear infinite; }
    @media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation:none !important; transition:none !important; scroll-behavior:auto !important; } }

    /* Full-screen animated canvas layer (kept behind Streamlit content) */
    html, body, [data-testid="stAppViewContainer"], .stApp { background: transparent !important; }
    .pp-animated-backdrop { position:fixed; inset:0; z-index:0; overflow:hidden; pointer-events:none; background:linear-gradient(120deg,#071326,#101b3d,#172554,#0b2540,#17143b); background-size:400% 400%; animation:ppBgShift 22s ease infinite; }
    .pp-animated-backdrop:before { content:""; position:absolute; inset:-20%; opacity:.24; background-image:linear-gradient(rgba(125,211,252,.13) 1px,transparent 1px),linear-gradient(90deg,rgba(125,211,252,.13) 1px,transparent 1px); background-size:54px 54px; mask-image:linear-gradient(to bottom,black,transparent 90%); animation:ppGridDrift 28s linear infinite; }
    .pp-blob { position:absolute; width:42vw; height:42vw; min-width:320px; min-height:320px; border-radius:50%; filter:blur(75px); opacity:.42; mix-blend-mode:screen; }
    .pp-blob.one { left:-12vw; top:-10vh; background:radial-gradient(circle,#06b6d4 0%,#2563eb 34%,transparent 70%); animation:ppBlobOne 19s ease-in-out infinite; }
    .pp-blob.two { right:-13vw; top:12vh; background:radial-gradient(circle,#8b5cf6 0%,#ec4899 30%,transparent 70%); animation:ppBlobTwo 24s ease-in-out infinite; }
    .pp-blob.three { left:28vw; bottom:-28vh; background:radial-gradient(circle,#0ea5e9 0%,#6366f1 38%,transparent 72%); animation:ppBlobThree 21s ease-in-out infinite; }
    .pp-blob.four { right:20vw; bottom:8vh; width:22vw;height:22vw; min-width:220px;min-height:220px; background:radial-gradient(circle,#22d3ee 0%,#3b82f6 42%,transparent 72%); animation:ppBlobFour 16s ease-in-out infinite; }
    @keyframes ppBgShift { 0%,100%{background-position:0% 50%} 50%{background-position:100% 50%} }
    @keyframes ppGridDrift { from{transform:translate3d(0,0,0)} to{transform:translate3d(54px,54px,0)} }
    @keyframes ppBlobOne { 0%,100%{transform:translate(0,0) scale(1)} 35%{transform:translate(22vw,12vh) scale(1.16)} 70%{transform:translate(8vw,30vh) scale(.88)} }
    @keyframes ppBlobTwo { 0%,100%{transform:translate(0,0) scale(1)} 40%{transform:translate(-24vw,18vh) scale(1.12)} 75%{transform:translate(-8vw,-12vh) scale(.9)} }
    @keyframes ppBlobThree { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(-16vw,-22vh) scale(1.2)} }
    @keyframes ppBlobFour { 0%,100%{transform:translate(0,0)} 50%{transform:translate(10vw,-14vh)} }
    [data-testid="stAppViewContainer"] > .main { position:relative; z-index:1; background:transparent !important; }
    [data-testid="stHeader"] { background:rgba(7,19,38,.24) !important; backdrop-filter:blur(16px); }
    .block-container { position:relative; z-index:2; }
    /* premium dark-glass content panels */
    div[data-testid="stMetric"], div[data-testid="stVerticalBlockBorderWrapper"] { background:rgba(255,255,255,.91) !important; backdrop-filter:blur(18px); }
    .hero { box-shadow:0 24px 70px rgba(37,99,235,.32),0 0 36px rgba(34,211,238,.12); }
    @media (prefers-reduced-motion: reduce) { .pp-animated-backdrop,.pp-blob,.pp-animated-backdrop:before { animation:none !important; } }

</style>
""", unsafe_allow_html=True)

# Full-viewport animated background layer
st.markdown('''<div class="pp-animated-backdrop" aria-hidden="true"><div class="pp-blob one"></div><div class="pp-blob two"></div><div class="pp-blob three"></div><div class="pp-blob four"></div></div>''', unsafe_allow_html=True)

# ----------------------------- CONSTANTS --------------------------------------
REQUIRED = [
    "Department", "Years_Experience", "Training_Hours", "Attendance_Rate",
    "Projects_Completed", "Manager_Rating", "Monthly_Salary"
]
NUMERIC = REQUIRED[1:]
TARGET = "Performance_Label"
CLASS_COLORS = {"High": "#16a34a", "Medium": "#f59e0b", "Low": "#ef4444"}
PLOT_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=20, r=20, t=55, b=25),
    font=dict(family="Inter, sans-serif", color="#334155"),
    paper_bgcolor="white",
    plot_bgcolor="white"
)

@st.cache_data
def load_sample():
    file_path = Path(__file__).parent / "data" / "employee_data.csv"
    if not file_path.exists():
        raise FileNotFoundError("Dataset not found. Place employee_data.csv in the data folder.")
    return pd.read_csv(file_path)

# ----------------------------- SIDEBAR ----------------------------------------
with st.sidebar:
    st.markdown("## 📈 PeoplePulse")
    st.caption("Employee intelligence workspace")
    st.divider()
    uploaded = st.file_uploader("Upload employee CSV", type=["csv"])
    st.markdown("### Dashboard filters")

try:
    raw = pd.read_csv(uploaded) if uploaded is not None else load_sample()
except Exception as exc:
    st.error(f"Unable to load dataset: {exc}")
    st.stop()

if raw.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

missing = [col for col in REQUIRED if col not in raw.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.info("Required columns: " + ", ".join(REQUIRED))
    st.stop()

raw = raw.copy()
if "Employee_ID" not in raw.columns:
    raw["Employee_ID"] = [f"EMP{i:04d}" for i in range(1, len(raw) + 1)]

# ----------------------------- DATA PREPARATION -------------------------------
data = raw.copy()
for col in NUMERIC:
    data[col] = pd.to_numeric(data[col], errors="coerce")
data["Department"] = data["Department"].fillna("Unknown").astype(str).str.strip()
data["Department"] = data["Department"].replace("", "Unknown")
data["Years_Experience"] = data["Years_Experience"].clip(0, 50)
data["Training_Hours"] = data["Training_Hours"].clip(lower=0)
data["Attendance_Rate"] = data["Attendance_Rate"].clip(0, 100)
data["Projects_Completed"] = data["Projects_Completed"].clip(lower=0)
data["Manager_Rating"] = data["Manager_Rating"].clip(1, 5)
data["Monthly_Salary"] = data["Monthly_Salary"].clip(lower=0)
for col in NUMERIC:
    median = data[col].median()
    data[col] = data[col].fillna(0 if pd.isna(median) else median)

data["Productivity_Index"] = (
    data["Projects_Completed"] * 10
    + data["Attendance_Rate"] * .35
    + data["Training_Hours"] * .25
)
data["Experience_Band"] = pd.cut(
    data["Years_Experience"], [-.001, 2, 5, 10, 50.001],
    labels=["Entry", "Junior", "Mid", "Senior"], include_lowest=True
).astype("object").fillna("Unknown")

generated_target = TARGET not in data.columns
if generated_target:
    score = (
        data["Attendance_Rate"] * .025
        + data["Training_Hours"] * .035
        + data["Projects_Completed"] * .28
        + data["Manager_Rating"] * .8
        + data["Years_Experience"] * .07
    )
    q1, q2 = score.quantile([.33, .67])
    data[TARGET] = np.where(score >= q2, "High", np.where(score >= q1, "Medium", "Low"))
else:
    data[TARGET] = data[TARGET].fillna("Unknown").astype(str).str.strip()
    data.loc[data[TARGET] == "", TARGET] = "Unknown"

data = data[data[TARGET] != "Unknown"].copy()
if len(data) < 4 or data[TARGET].nunique() < 2:
    st.error("At least 4 labeled records and 2 performance categories are required.")
    st.stop()

# ----------------------------- FILTERS -----------------------------------------
with st.sidebar:
    department_options = sorted(data["Department"].unique().tolist())
    selected_departments = st.multiselect(
        "Department", department_options, default=department_options
    )
    category_options = sorted(data[TARGET].unique().tolist())
    selected_categories = st.multiselect(
        "Performance category", category_options, default=category_options
    )
    st.markdown("---")
    st.caption("Source: " + ("Uploaded CSV" if uploaded is not None else "Included sample dataset"))
    st.caption(f"Dataset rows: {len(data):,}")

filtered = data[
    data["Department"].isin(selected_departments)
    & data[TARGET].isin(selected_categories)
]
if filtered.empty:
    st.warning("No records match the selected filters. Update the filters in the sidebar.")
    st.stop()

# ----------------------------- HERO --------------------------------------------
st.markdown("""
<div class="hero">
  <div class="hero-tag">FDS PROJECT • DATA SCIENCE + MACHINE LEARNING</div>
  <h1>Employee Performance Analytics</h1>
  <p>Transform workforce data into measurable insights using Pandas, feature engineering and a Scikit-learn classification model.</p>
  <div class="hero-live"><span class="live-dot"></span> LIVE ANALYTICS • Interactive dashboard</div>
</div>
<div class="motion-strip"><span></span></div>
""", unsafe_allow_html=True)

# KPI row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Employees analyzed", f"{len(filtered):,}", help="Records included after filters")
k2.metric("Departments", f"{filtered['Department'].nunique():,}")
k3.metric("Avg. attendance", f"{filtered['Attendance_Rate'].mean():.1f}%")
k4.metric("Avg. training", f"{filtered['Training_Hours'].mean():.1f} hrs")

# ----------------------------- MODEL PIPELINE ---------------------------------
FEATURES = [
    "Department", "Years_Experience", "Training_Hours", "Attendance_Rate",
    "Projects_Completed", "Manager_Rating", "Monthly_Salary",
    "Productivity_Index", "Experience_Band"
]
X, y = data[FEATURES].copy(), data[TARGET].copy()
class_counts = y.value_counts()
num_classes = y.nunique()
test_size = min(len(data) - 1, max(num_classes, int(np.ceil(len(data) * .25))))
stratify_arg = y if class_counts.min() >= 2 and test_size >= num_classes else None

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_arg
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

numeric_features = [c for c in FEATURES if c not in ["Department", "Experience_Band"]]
categorical_features = ["Department", "Experience_Band"]
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocess = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
])
model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    ))
])
try:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.stop()

accuracy = accuracy_score(y_test, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, predictions, average="weighted", zero_division=0
)

# ----------------------------- NAV TABS ----------------------------------------
tab_overview, tab_data, tab_model, tab_predict = st.tabs([
    "📊 Analytics Overview", "🗂️ Data Explorer", "🧠 ML Model", "🔮 Prediction Studio"
])

with tab_overview:
    st.markdown('<div class="section-title">Performance insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Interactive charts respond to dataset uploads and sidebar filters.</div>', unsafe_allow_html=True)

    chart1, chart2 = st.columns(2)
    with chart1:
        counts = filtered[TARGET].value_counts().rename_axis("Performance").reset_index(name="Employees")
        display_order = [x for x in ["High", "Medium", "Low"] if x in counts["Performance"].tolist()]
        display_order += [x for x in counts["Performance"].tolist() if x not in display_order]
        fig = px.bar(
            counts, x="Performance", y="Employees", color="Performance", text="Employees",
            category_orders={"Performance": display_order},
            color_discrete_map=CLASS_COLORS, title="Performance distribution"
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(**PLOT_LAYOUT, showlegend=False, xaxis_title="", yaxis_title="Employees")
        st.plotly_chart(fig, use_container_width=True)

    with chart2:
        metrics = filtered.groupby(TARGET)[
            ["Attendance_Rate", "Training_Hours", "Projects_Completed"]
        ].mean().reset_index().melt(
            id_vars=TARGET, var_name="Indicator", value_name="Average"
        )
        fig = px.bar(
            metrics, x=TARGET, y="Average", color="Indicator", barmode="group",
            text_auto=".1f", title="Average performance indicators",
            color_discrete_sequence=["#2563eb", "#8b5cf6", "#14b8a6"]
        )
        fig.update_layout(**PLOT_LAYOUT, xaxis_title="", yaxis_title="Average value",
                          legend_title="Indicator")
        st.plotly_chart(fig, use_container_width=True)

    chart3, chart4 = st.columns(2)
    with chart3:
        dept_counts = filtered.groupby(["Department", TARGET]).size().reset_index(name="Employees")
        fig = px.bar(
            dept_counts, x="Department", y="Employees", color=TARGET, barmode="stack",
            text="Employees", color_discrete_map=CLASS_COLORS,
            title="Department-wise performance"
        )
        fig.update_layout(**PLOT_LAYOUT, xaxis_title="", yaxis_title="Employees",
                          legend_title="Performance")
        st.plotly_chart(fig, use_container_width=True)

    with chart4:
        fig = px.scatter(
            filtered, x="Attendance_Rate", y="Productivity_Index", color=TARGET,
            size="Projects_Completed", hover_name="Employee_ID",
            hover_data=["Department", "Training_Hours", "Manager_Rating"],
            color_discrete_map=CLASS_COLORS, title="Attendance vs. productivity",
            size_max=18, opacity=.82
        )
        fig.update_layout(**PLOT_LAYOUT, xaxis_title="Attendance (%)",
                          yaxis_title="Productivity index", legend_title="Performance")
        st.plotly_chart(fig, use_container_width=True)

with tab_data:
    st.markdown('<div class="section-title">Employee data explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Review the cleaned dataset and export the processed records.</div>', unsafe_allow_html=True)
    search = st.text_input("Search employee ID or department", placeholder="e.g. EMP001 or Finance")
    view_data = filtered.copy()
    if search.strip():
        term = search.strip().lower()
        view_data = view_data[
            view_data["Employee_ID"].astype(str).str.lower().str.contains(term, na=False)
            | view_data["Department"].astype(str).str.lower().str.contains(term, na=False)
        ]
    st.dataframe(view_data, use_container_width=True, height=420, hide_index=True)
    st.download_button(
        "⬇️ Download cleaned dataset",
        view_data.to_csv(index=False).encode("utf-8"),
        "cleaned_employee_data.csv", "text/csv"
    )
    st.caption(f"Showing {len(view_data):,} records • {view_data.shape[1]} columns")

with tab_model:
    st.markdown('<div class="section-title">Model evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Random Forest classification evaluated on a held-out test set.</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test accuracy", f"{accuracy:.1%}")
    m2.metric("Weighted precision", f"{precision:.1%}")
    m3.metric("Weighted recall", f"{recall:.1%}")
    m4.metric("Weighted F1-score", f"{f1:.1%}")

    if generated_target:
        st.warning("Demo performance labels were generated using a scoring formula. Metrics demonstrate the ML workflow and are not evidence of real-world predictive accuracy.")
    else:
        st.info("Metrics are calculated from a held-out test set. Results depend on the quality and size of the labeled data.")

    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    with st.expander("📋 Classification report", expanded=True):
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

    model_left, model_right = st.columns(2)
    with model_left:
        labels = sorted(y.unique())
        matrix = confusion_matrix(y_test, predictions, labels=labels)
        fig = px.imshow(
            matrix, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
            labels={"x": "Predicted", "y": "Actual", "color": "Count"},
            title="Confusion matrix"
        )
        fig.update_layout(**PLOT_LAYOUT, xaxis_title="Predicted label", yaxis_title="Actual label")
        st.plotly_chart(fig, use_container_width=True)

    with model_right:
        try:
            names = model.named_steps["preprocess"].get_feature_names_out()
            values = model.named_steps["classifier"].feature_importances_
            importance = pd.DataFrame({"Feature": names, "Importance": values}).nlargest(12, "Importance")
            fig = px.bar(
                importance.sort_values("Importance"), x="Importance", y="Feature",
                orientation="h", text_auto=".3f", title="Top feature importance",
                color="Importance", color_continuous_scale="Blues"
            )
            fig.update_layout(**PLOT_LAYOUT, yaxis_title="", xaxis_title="Importance",
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.info(f"Feature importance could not be displayed: {exc}")

    with st.expander("🔁 Cross-validation"):
        folds = min(5, int(y.value_counts().min()))
        if folds >= 2:
            try:
                scores = cross_val_score(model, X, y, cv=folds, scoring="accuracy")
                st.metric(f"Mean {folds}-fold CV accuracy", f"{scores.mean():.1%}")
                cv_data = pd.DataFrame({
                    "Fold": [f"Fold {i+1}" for i in range(len(scores))],
                    "Accuracy": scores
                })
                fig = px.bar(cv_data, x="Fold", y="Accuracy", text_auto=".1%",
                             title="Accuracy across validation folds")
                fig.update_layout(**PLOT_LAYOUT, yaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.info(f"Cross-validation unavailable: {exc}")
        else:
            st.info("At least two examples per class are required for cross-validation.")

with tab_predict:
    st.markdown('<div class="section-title">Prediction studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Enter a hypothetical employee profile to estimate its performance category.</div>', unsafe_allow_html=True)
    departments = sorted(data["Department"].unique().tolist()) or ["Unknown"]
    with st.form("prediction_form"):
        p1, p2, p3 = st.columns(3)
        department = p1.selectbox("Department", departments)
        experience = p2.number_input("Years of experience", 0.0, 50.0, 3.0, .5)
        training = p3.number_input("Training hours", 0.0, 500.0, 20.0, 1.0)
        attendance = p1.number_input("Attendance rate (%)", 0.0, 100.0, 90.0, 1.0)
        projects = p2.number_input("Projects completed", 0.0, 100.0, 4.0, 1.0)
        rating = p3.slider("Manager rating", 1.0, 5.0, 3.0, .5)
        salary = p1.number_input("Monthly salary (₹)", 0.0, value=45000.0, step=1000.0)
        submit = st.form_submit_button("✨ Predict performance", use_container_width=True)

    if submit:
        productivity = projects * 10 + attendance * .35 + training * .25
        band = pd.cut(
            pd.Series([experience]), [-.001, 2, 5, 10, 50.001],
            labels=["Entry", "Junior", "Mid", "Senior"], include_lowest=True
        ).astype("object").fillna("Unknown").iloc[0]
        new_employee = pd.DataFrame([{
            "Department": department, "Years_Experience": experience,
            "Training_Hours": training, "Attendance_Rate": attendance,
            "Projects_Completed": projects, "Manager_Rating": rating,
            "Monthly_Salary": salary, "Productivity_Index": productivity,
            "Experience_Band": band
        }])[FEATURES]
        predicted = model.predict(new_employee)[0]
        probabilities = model.predict_proba(new_employee)[0]
        st.success(f"Predicted performance category: **{predicted}**")
        probability_df = pd.DataFrame({
            "Category": model.classes_, "Probability": probabilities
        })
        fig = px.bar(
            probability_df, x="Category", y="Probability", color="Category",
            text_auto=".1%", title="Prediction confidence by category",
            color_discrete_map=CLASS_COLORS
        )
        fig.update_layout(**PLOT_LAYOUT, yaxis_tickformat=".0%", yaxis_title="Probability")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown(
    '<div class="small-note">PeoplePulse • FDS academic prototype | '
    'Predictions are decision-support estimates and should not be used as the sole basis for employment decisions.</div>',
    unsafe_allow_html=True
)
