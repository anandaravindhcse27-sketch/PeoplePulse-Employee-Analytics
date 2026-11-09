import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------- MODERN UI -------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

    :root {
        --bg:#f6f7fb; --surface:#ffffff; --line:#e7e9f2; --ink:#0f172a; --muted:#64748b;
        --accent:#4f46e5; --accent-2:#7c3aed; --accent-soft:#eef2ff;
        --good:#10b981; --warn:#f59e0b; --bad:#f43f5e;
        --radius:18px; --shadow:0 1px 2px rgba(15,23,42,.04), 0 8px 24px -12px rgba(15,23,42,.10);
    }
    html, body, [class*="css"], .stApp { font-family:'Plus Jakarta Sans', system-ui, sans-serif; color:var(--ink); }
    .stApp { background:var(--bg); }
    [data-testid="stHeader"] { background:transparent; }
    footer { visibility:hidden; }
    .block-container { max-width:1480px; padding:2rem 2.6rem 3rem; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] { background:#0b1020; border-right:none; }
    [data-testid="stSidebar"] > div:first-child { padding-top:1.4rem; }
    [data-testid="stSidebar"] * { color:#cbd5e1; }
    [data-testid="stSidebar"] h3 { color:#f8fafc !important; font-size:12px; text-transform:uppercase; letter-spacing:1.4px; font-weight:700; margin-top:.4rem; }
    [data-testid="stSidebar"] hr { border-color:rgba(148,163,184,.16); }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section { background:rgba(255,255,255,.04); border:1.5px dashed rgba(148,163,184,.35); border-radius:14px; }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button { background:var(--accent); color:#fff; border:none; border-radius:10px; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div { background:rgba(255,255,255,.06); border:1px solid rgba(148,163,184,.25); border-radius:12px; }
    [data-testid="stSidebar"] [data-baseweb="tag"] { background:rgba(99,102,241,.28); border-radius:8px; }
    .brand { display:flex; align-items:center; gap:12px; margin-bottom:4px; }
    .brand-logo { width:38px; height:38px; border-radius:12px; display:grid; place-items:center; font-size:19px; background:linear-gradient(135deg,#6366f1,#a855f7); box-shadow:0 8px 20px -6px rgba(99,102,241,.7); }
    .brand-name { font-size:18px; font-weight:800; color:#fff !important; letter-spacing:-.4px; line-height:1.1; }
    .brand-sub { font-size:12px; color:#7c8aa5 !important; }

    /* ---------- Header ---------- */
    .topbar { display:flex; align-items:flex-end; justify-content:space-between; gap:24px; flex-wrap:wrap; margin-bottom:26px; }
    .eyebrow { display:inline-flex; align-items:center; gap:8px; font-size:12px; font-weight:700; color:var(--accent); background:var(--accent-soft); padding:5px 12px; border-radius:999px; margin-bottom:12px; }
    .eyebrow i { width:7px; height:7px; border-radius:50%; background:var(--good); display:inline-block; box-shadow:0 0 0 4px rgba(16,185,129,.18); }
    .topbar h1 { font-size:clamp(28px,3.2vw,40px); font-weight:800; letter-spacing:-1.3px; line-height:1.08; margin:0 0 8px; color:var(--ink); }
    .topbar h1 span { background:linear-gradient(90deg,var(--accent),var(--accent-2)); -webkit-background-clip:text; background-clip:text; color:transparent; }
    .topbar p { color:var(--muted); font-size:15px; margin:0; max-width:720px; line-height:1.55; }
    .chips { display:flex; gap:8px; flex-wrap:wrap; }
    .chip { font-size:12px; font-weight:600; color:#334155; background:var(--surface); border:1px solid var(--line); padding:7px 13px; border-radius:999px; box-shadow:var(--shadow); }

    /* ---------- KPI cards ---------- */
    div[data-testid="stMetric"] { background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); padding:20px 22px 18px; box-shadow:var(--shadow); position:relative; overflow:hidden; transition:transform .2s ease, box-shadow .2s ease; }
    div[data-testid="stMetric"]:before { content:""; position:absolute; left:0; top:18px; bottom:18px; width:4px; border-radius:0 4px 4px 0; background:linear-gradient(180deg,var(--accent),var(--accent-2)); }
    div[data-testid="stMetric"]:hover { transform:translateY(-2px); box-shadow:0 14px 34px -14px rgba(79,70,229,.28); }
    div[data-testid="stMetricLabel"] p { color:var(--muted); font-size:12.5px; font-weight:600; text-transform:uppercase; letter-spacing:.7px; }
    div[data-testid="stMetricValue"] { color:var(--ink); font-size:32px; font-weight:800; letter-spacing:-1px; }

    /* ---------- Tabs as segmented pills ---------- */
    div[data-testid="stTabs"] [role="tablist"] { gap:6px; background:var(--surface); border:1px solid var(--line); padding:6px; border-radius:16px; width:fit-content; box-shadow:var(--shadow); margin:26px 0 22px; border-bottom:none; }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"], div[data-testid="stTabs"] [data-baseweb="tab-border"] { display:none; }
    div[data-testid="stTabs"] button[role="tab"] { border-radius:11px; padding:9px 18px; height:auto; font-weight:600; color:var(--muted); background:transparent; transition:all .18s ease; }
    div[data-testid="stTabs"] button[role="tab"]:hover { color:var(--ink); background:#f1f3f9; }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { background:linear-gradient(135deg,var(--accent),var(--accent-2)); color:#fff !important; box-shadow:0 8px 18px -8px rgba(79,70,229,.8); }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p { color:#fff !important; }

    /* ---------- Section headings ---------- */
    .section-title { font-size:22px; font-weight:800; letter-spacing:-.5px; color:var(--ink); margin:2px 0 4px; }
    .section-subtitle { color:var(--muted); font-size:14px; margin-bottom:20px; }

    /* ---------- Panels, charts, tables ---------- */
    [data-testid="stPlotlyChart"] { background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); padding:10px 10px 4px; box-shadow:var(--shadow); overflow:hidden; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:var(--radius); overflow:hidden; box-shadow:var(--shadow); }
    div[data-testid="stExpander"] { background:var(--surface); border:1px solid var(--line) !important; border-radius:var(--radius); box-shadow:var(--shadow); overflow:hidden; }
    div[data-testid="stExpander"] summary { font-weight:700; }
    div[data-testid="stForm"] { background:var(--surface); border:1px solid var(--line); border-radius:22px; padding:26px 26px 18px; box-shadow:var(--shadow); }
    div[data-testid="stAlert"] { border-radius:14px; border:1px solid var(--line); }

    /* ---------- Inputs & buttons ---------- */
    input, textarea, [data-baseweb="select"] > div, [data-baseweb="input"] { border-radius:12px !important; }
    [data-testid="stTextInput"] input { background:var(--surface); border:1px solid var(--line); padding:12px 14px; }
    [data-testid="stTextInput"] input:focus { border-color:var(--accent); box-shadow:0 0 0 4px rgba(79,70,229,.12); }
    .stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button { border-radius:12px; font-weight:700; padding:.65rem 1.2rem; border:1px solid var(--line); transition:all .18s ease; }
    .stDownloadButton button { background:var(--surface); color:var(--ink); }
    .stDownloadButton button:hover { border-color:var(--accent); color:var(--accent); transform:translateY(-1px); }
    [data-testid="stFormSubmitButton"] button { background:linear-gradient(135deg,var(--accent),var(--accent-2)); color:#fff; border:none; box-shadow:0 12px 24px -10px rgba(79,70,229,.7); }
    [data-testid="stFormSubmitButton"] button:hover { transform:translateY(-2px); box-shadow:0 16px 30px -10px rgba(79,70,229,.8); color:#fff; }

    .small-note { color:#94a3b8; font-size:12.5px; text-align:center; padding-top:6px; }
    @media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation:none !important; transition:none !important; } }
</style>
""", unsafe_allow_html=True)

# ----------------------------- CONSTANTS --------------------------------------
REQUIRED = [
    "Department", "Years_Experience", "Training_Hours", "Attendance_Rate",
    "Projects_Completed", "Manager_Rating", "Monthly_Salary"
]
NUMERIC = REQUIRED[1:]
TARGET = "Performance_Label"
CLASS_COLORS = {"High": "#10b981", "Medium": "#f59e0b", "Low": "#f43f5e"}
ACCENT_SCALE = ["#eef2ff", "#a5b4fc", "#4f46e5", "#312e81"]
CHART_SEQUENCE = ["#4f46e5", "#a855f7", "#14b8a6"]

_axis = dict(showgrid=True, gridcolor="#eef0f6", zeroline=False, linecolor="#e2e8f0",
             tickfont=dict(size=12, color="#64748b"), title_font=dict(size=12, color="#64748b"))
_template = go.layout.Template()
_template.layout = go.Layout(
    colorway=["#4f46e5", "#a855f7", "#14b8a6", "#f59e0b", "#f43f5e", "#0ea5e9"],
    xaxis=dict(_axis, showgrid=False), yaxis=_axis,
    title=dict(font=dict(size=16, color="#0f172a", family="Plus Jakarta Sans, sans-serif"), x=0.02, xanchor="left"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=12, color="#475569"), title=dict(font=dict(size=12, color="#94a3b8"))),
    hoverlabel=dict(bgcolor="#0f172a", font=dict(color="white", size=12), bordercolor="#0f172a"),
)
PLOT_LAYOUT = dict(
    template=_template,
    margin=dict(l=18, r=18, t=64, b=22),
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#334155"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)

@st.cache_data
def load_sample():
    file_path = Path(__file__).parent / "data" / "employee_data.csv"
    if not file_path.exists():
        raise FileNotFoundError("Dataset not found. Place employee_data.csv in the data folder.")
    return pd.read_csv(file_path)

# ----------------------------- SIDEBAR ----------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-logo">◆</div>'
        '<div><div class="brand-name">PeoplePulse</div>'
        '<div class="brand-sub">Employee intelligence</div></div></div>',
        unsafe_allow_html=True
    )
    st.divider()
    st.markdown("### Data source")
    uploaded = st.file_uploader("Upload employee CSV", type=["csv"])
    st.markdown("### Filters")

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

# ----------------------------- HEADER ------------------------------------------
st.markdown("""
<div class="topbar">
  <div>
    <div class="eyebrow"><i></i> Live workspace</div>
    <h1>Employee <span>Performance</span> Analytics</h1>
    <p>Turn workforce data into clear insight with Pandas feature engineering and a Scikit-learn classification model.</p>
  </div>
  <div class="chips">
    <span class="chip">Pandas</span><span class="chip">Random Forest</span><span class="chip">Plotly</span><span class="chip">Streamlit</span>
  </div>
</div>
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
    "Overview", "Data Explorer", "ML Model", "Prediction Studio"
])

with tab_overview:
    st.markdown('<div class="section-title">Performance overview</div>', unsafe_allow_html=True)
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
            color_discrete_sequence=CHART_SEQUENCE
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
        "Download cleaned dataset",
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
    with st.expander("Classification report", expanded=True):
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

    model_left, model_right = st.columns(2)
    with model_left:
        labels = sorted(y.unique())
        matrix = confusion_matrix(y_test, predictions, labels=labels)
        fig = px.imshow(
            matrix, x=labels, y=labels, text_auto=True, color_continuous_scale=ACCENT_SCALE,
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
                color="Importance", color_continuous_scale=ACCENT_SCALE
            )
            fig.update_layout(**PLOT_LAYOUT, yaxis_title="", xaxis_title="Importance",
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.info(f"Feature importance could not be displayed: {exc}")

    with st.expander("Cross-validation"):
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
        submit = st.form_submit_button("Predict performance", use_container_width=True)

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

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div class="small-note">PeoplePulse • FDS academic prototype | '
    'Predictions are decision-support estimates and should not be used as the sole basis for employment decisions.</div>',
    unsafe_allow_html=True
)