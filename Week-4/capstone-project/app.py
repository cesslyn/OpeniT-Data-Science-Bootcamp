import os, json, sqlite3, datetime, warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from PIL import Image
warnings.filterwarnings("ignore")


#  CONFIG
CLASS_NAMES = sorted(["earthquake", "fire", "flood", "landslide", "normal"])
CLS_HEX = {
    "earthquake": "#F63E31", "fire": "#FF8C42",
    "flood":      "#3B9EBF", "landslide": "#9B8EA0", "normal": "#52B788",
}
ACCENT = "#F63E31"
CYAN   = "#3B9EBF"
GREEN  = "#52B788"
ORANGE = "#FF8C42"
PURPLE = "#9B8EA0"

PAL = dict(
    bg="#0b1820",       side_bg="#0d1f28",  card_bg="#0f2632",
    card_bdr="#1a3d50", text="#dce8ed",     sub="#5f8a9a",
    chart_bg="#0b1820", grid="#132533",
    blob1="rgba(246,62,49,0.07)", blob2="rgba(59,158,191,0.07)",
)

MODEL_PATH = os.path.join("models", "disaster_classifier.keras")
DB_PATH    = "predictions.db"
EVAL_DIR   = "eval"
IMG_SIZE   = 224
TIDY_CSV   = os.path.join("outputs", "cleaned_tidy.csv")
WIDE_CSV   = os.path.join("outputs", "cleaned_wide.csv")
SUMM_CSV   = os.path.join("outputs", "summary_by_hazard.csv")

PAGES = [
    ("Executive Dashboard", "executive"),
    ("Disaster Analytics",  "analytics"),
    ("Visualizations",      "visuals"),
    ("AI Classifier",       "classify"),
    ("Prediction History",  "history"),
    ("Dataset Explorer",    "explorer"),
]

PAGE_ICONS = {
    "executive": "🏠",
    "analytics": "📊",
    "visuals":   "📈",
    "classify":  "🔍",
    "history":   "🕐",
    "explorer":  "🗄️",
}

# Major hazard category mapping
HAZARD_CATEGORIES = {
    "Meteorological": [
        "cyclone", "typhoon", "tropical", "storm", "wind", "rain",
        "thunderstorm", "lightning", "squall", "tornado", "gale",
        "monsoon", "intertropical",
    ],
    "Hydrological": [
        "flood", "flashflood", "flash flood", "surge", "landslide",
        "mudslide", "debris", "lahar", "erosion",
    ],
    "Geophysical": [
        "earthquake", "seismic", "volcano", "volcanic", "eruption",
        "tsunami", "rockfall", "ground movement", "liquefaction",
    ],
    "Climatological": [
        "drought", "el nino", "el niño", "la nina", "la niña",
        "heat", "cold", "frost", "dry spell", "fire", "wildfire",
    ],
    "Biological": [
        "disease", "epidemic", "pest", "locust", "outbreak",
        "infection", "dengue", "cholera", "influenza",
    ],
    "Combined Events": [
        "combined", "multiple", "complex", "compound",
    ],
}

CAT_COLORS = {
    "Meteorological":  "#3B9EBF",
    "Hydrological":    "#F63E31",
    "Geophysical":     "#FF8C42",
    "Climatological":  "#F6D860",
    "Biological":      "#52B788",
    "Combined Events": "#9B8EA0",
}

def assign_category(hazard_type: str) -> str:
    ht = hazard_type.lower()
    for cat, keywords in HAZARD_CATEGORIES.items():
        if any(kw in ht for kw in keywords):
            return cat
    return "Combined Events"

#  PAGE CONFIG
st.set_page_config(
    page_title="DisasterVision",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  DATA LOADERS
@st.cache_data
def load_tidy():
    if not os.path.exists(TIDY_CSV):
        return None
    df = pd.read_csv(TIDY_CSV)
    df = df[df["persons_affected"] > 0].copy()
    return df

@st.cache_data
def derived_data():
    tidy = load_tidy()
    if tidy is None:
        return None, None, None, None
    yearly = tidy.groupby("year")[["families_affected", "persons_affected"]].sum().reset_index()
    summary = (
        tidy.groupby("hazard_type")["persons_affected"]
        .agg(
            total_persons="sum",
            max_year_persons="max",
            years_with_impact=lambda x: (x > 0).sum(),
        )
        .sort_values("total_persons", ascending=False)
        .reset_index()
    )
    heatmap_df = tidy.pivot_table(
        index="hazard_type", columns="year",
        values="persons_affected", aggfunc="sum", fill_value=0,
    )
    return tidy, yearly, summary, heatmap_df

#  DATABASE
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT, pred_class TEXT,
        confidence REAL, timestamp TEXT)""")
    conn.commit()
    conn.close()

def save_prediction(name, cls, conf):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions VALUES (NULL,?,?,?,?)",
        (name, cls, round(conf, 4), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()

def load_predictions(cls_filter="All", min_conf=0.0):
    conn = sqlite3.connect(DB_PATH)
    q, params = "SELECT * FROM predictions WHERE confidence >= ?", [min_conf]
    if cls_filter != "All":
        q += " AND pred_class = ?"
        params.append(cls_filter)
    q += " ORDER BY id DESC"
    df = pd.read_sql(q, conn, params=params)
    conn.close()
    return df

#  MODEL
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception:
        return None

def run_predict(model, img):
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input
    arr = np.array(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    arr = preprocess_input(np.expand_dims(arr, 0))
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs

#  PLOTLY THEME
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, DM Mono, monospace", color=PAL["text"], size=11),
    xaxis=dict(gridcolor=PAL["grid"], linecolor=PAL["card_bdr"], tickfont=dict(color=PAL["sub"], size=10)),
    yaxis=dict(gridcolor=PAL["grid"], linecolor=PAL["card_bdr"], tickfont=dict(color=PAL["sub"], size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=PAL["text"], size=10)),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(bgcolor=PAL["card_bg"], font_color=PAL["text"], bordercolor=PAL["card_bdr"]),
)

def apply_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

def hex_to_rgba(hex_color, alpha=0.7):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

#  CSS
def inject_css():
    bg       = PAL["bg"]
    side_bg  = PAL["side_bg"]
    card_bg  = PAL["card_bg"]
    card_bdr = PAL["card_bdr"]
    text     = PAL["text"]
    sub      = PAL["sub"]
    grid     = PAL["grid"]
    blob1    = PAL["blob1"]
    blob2    = PAL["blob2"]

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body, .stApp {{
    background: {bg} !important;
    color: {text} !important;
    font-family: 'Inter', sans-serif;
}}
.stApp::before {{
    content: '';
    position: fixed; inset: 0; z-index: -2;
    background-color: {bg};
    background-image:
        radial-gradient(ellipse 80% 60% at 5%  10%, {blob1} 0%, transparent 60%),
        radial-gradient(ellipse 55% 45% at 95% 90%, {blob2} 0%, transparent 55%),
        radial-gradient(ellipse 40% 40% at 50% 50%, rgba(59,158,191,0.06) 0%, transparent 70%);
}}
[data-testid="stHeader"], footer {{ display: none !important; }}
.block-container {{ padding: 1.8rem 2rem !important; max-width: 100% !important; }}

/* SIDEBAR */
[data-testid="stSidebar"] {{
    min-width: 72px !important; max-width: 72px !important;
    background: {side_bg} !important;
    backdrop-filter: blur(28px) !important;
    -webkit-backdrop-filter: blur(28px) !important;
    border-right: 1px solid {card_bdr} !important;
    box-shadow: 2px 0 20px rgba(0,0,0,0.35) !important;
}}
[data-testid="stSidebarContent"] {{
    padding: 14px 0 !important;
    display: flex !important; flex-direction: column !important;
    align-items: center !important; gap: 4px !important;
}}
[data-testid="stSidebarCollapseButton"] {{ display: none !important; }}
.sidebar-logo {{
    width: 44px; height: 44px; background: {ACCENT};
    border-radius: 13px; display: flex; align-items: center; justify-content: center;
    margin: 4px auto 20px; box-shadow: 0 4px 16px {ACCENT}66; flex-shrink: 0;
}}
[data-testid="stSidebar"] .stButton {{ width: 48px !important; margin: 2px auto !important; }}
[data-testid="stSidebar"] .stButton > button {{
    width: 48px !important; height: 48px !important; padding: 0 !important;
    border-radius: 13px !important; font-size: 22px !important; line-height: 1 !important;
    background: transparent !important; color: #ffffff !important;
    border: 1px solid transparent !important; box-shadow: none !important;
    outline: none !important; transition: all 0.15s ease !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    opacity: 0.6 !important;
}}
[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stButton > button span,
[data-testid="stSidebar"] .stButton > button * {{
    color: #ffffff !important; background: transparent !important;
    font-size: 22px !important; line-height: 1 !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(255,255,255,0.10) !important;
    border-color: rgba(255,255,255,0.15) !important; opacity: 1 !important;
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
    background: {ACCENT}33 !important; color: {ACCENT} !important;
    border-color: {ACCENT}66 !important; box-shadow: 0 0 12px {ACCENT}33 !important;
    opacity: 1 !important;
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"] p,
[data-testid="stSidebar"] .stButton > button[kind="primary"] span,
[data-testid="stSidebar"] .stButton > button[kind="primary"] * {{
    color: {ACCENT} !important;
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
    background: {ACCENT}44 !important; opacity: 1 !important;
}}

/* TYPOGRAPHY */
.pg-title {{
    font-family: 'Syne', sans-serif; font-size: 2.5rem; font-weight: 800;
    color: #ffffff; letter-spacing: -0.02em; line-height: 1.25;
}}
.pg-sub {{
    font-family: 'DM Mono', monospace; font-size: 0.7rem; color: {sub};
    letter-spacing: 0.05em; margin-top: 2px; margin-bottom: 22px;
}}

/* KPI CARDS */
.kpi-row {{ display: flex; gap: 14px; margin-bottom: 18px; flex-wrap: wrap; }}
.kpi {{
    flex: 1; min-width: 140px; background: {card_bg}cc;
    backdrop-filter: blur(20px); border: 1px solid {card_bdr};
    border-radius: 16px; padding: 20px 22px 18px;
    position: relative; overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    transition: box-shadow 0.2s, transform 0.2s;
}}
.kpi:hover {{ box-shadow: 0 8px 32px rgba(0,0,0,0.4); transform: translateY(-2px); }}
.kpi-accent {{ position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 16px 16px 0 0; }}
.kpi-label {{
    font-family: 'DM Mono', monospace; font-size: 0.5rem; color: {sub};
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;
}}
.kpi-val {{
    font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800;
    color: #fff; line-height: 1.15; margin-bottom: 4px;
}}
.kpi-delta {{ font-size: 0.68rem; color: {sub}; margin-top: 5px; }}

/* CHART CARDS */
.chart-card {{
    background: {card_bg}cc; backdrop-filter: blur(20px);
    border: 1px solid {card_bdr}; border-radius: 16px;
    padding: 20px 20px 14px; margin-bottom: 14px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.22);
}}
.chart-title {{
    font-family: 'Syne', sans-serif; font-size: 0.9rem; font-weight: 700;
    color: #fff; margin-bottom: 2px;
}}
.chart-sub {{
    font-family: 'DM Mono', monospace; font-size: 0.62rem; color: {sub};
    margin-bottom: 14px; letter-spacing: 0.04em;
}}

/* INSIGHT CARDS */
.insight-card {{
    background: linear-gradient(135deg, {card_bg}ee, {card_bg}aa);
    border: 1px solid {card_bdr}; border-left: 3px solid {ACCENT};
    border-radius: 12px; padding: 16px 18px; margin-bottom: 10px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.2); transition: border-color 0.2s;
}}
.insight-card:hover {{ border-left-color: {CYAN}; }}
.insight-label {{
    font-family: 'DM Mono', monospace; font-size: 0.58rem; color: {ACCENT};
    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 5px;
}}
.insight-text {{ font-size: 0.83rem; color: {text}; line-height: 1.55; }}

/* TIMELINE */
.timeline-item {{
    display: flex; gap: 16px; align-items: flex-start;
    padding: 12px 0; border-bottom: 1px solid {card_bdr};
}}
.timeline-item:last-child {{ border-bottom: none; }}
.timeline-dot {{
    width: 10px; height: 10px; border-radius: 50%;
    flex-shrink: 0; margin-top: 5px; box-shadow: 0 0 8px currentColor;
}}
.timeline-year {{
    font-family: 'Syne', sans-serif; font-size: 1.1rem;
    font-weight: 800; color: #fff; min-width: 44px;
}}

/* RESULT BOX */
.result-box {{
    background: {card_bg}cc; backdrop-filter: blur(20px);
    border: 1px solid {card_bdr}; border-radius: 16px;
    padding: 28px 24px; box-shadow: 0 4px 24px rgba(0,0,0,0.22);
}}
.result-cls {{
    font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800;
    letter-spacing: -0.02em; margin: 10px 0 4px;
}}
.result-conf {{ font-family: 'DM Mono', monospace; font-size: 0.78rem; color: {sub}; }}
.pbar-row {{ margin-bottom: 10px; }}
.pbar-meta {{
    display: flex; justify-content: space-between;
    font-size: 0.82rem; margin-bottom: 4px; color: {text};
}}
.pbar-meta span:last-child {{ font-family: 'DM Mono', monospace; font-weight: 500; }}
.pbar-track {{ background: {grid}; border-radius: 4px; height: 7px; }}
.pbar-fill  {{ border-radius: 4px; height: 7px; transition: width 0.6s cubic-bezier(.4,0,.2,1); }}

/* BADGES */
.badge {{
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.04em;
}}
.badge-hi  {{ background: #52B78820; color: #52B788; }}
.badge-lo  {{ background: {ACCENT}20; color: {ACCENT}; }}
.badge-med {{ background: #FF8C4220; color: #FF8C42; }}

/* WIDGETS */
label, .stSelectbox label, .stSlider label, .stFileUploader label {{
    font-family: 'DM Mono', monospace !important; font-size: 0.62rem !important;
    color: {sub} !important; text-transform: uppercase; letter-spacing: 0.1em;
}}
.stSelectbox > div > div {{
    background: {card_bg}cc !important; border-color: {card_bdr} !important;
    color: {text} !important; border-radius: 10px !important;
}}
[data-testid="stFileUploader"] {{
    background: {card_bg}88; border: 1px dashed {card_bdr};
    border-radius: 12px; padding: 4px;
}}
:not([data-testid="stSidebar"]) .stButton > button {{
    background: {ACCENT} !important; color: white !important;
    border: none !important; border-radius: 9px !important;
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    padding: 10px 24px !important; transition: opacity 0.2s !important;
}}
:not([data-testid="stSidebar"]) .stButton > button:hover {{ opacity: 0.86 !important; }}
.stDownloadButton > button {{
    background: transparent !important; border: 1px solid {card_bdr} !important;
    color: {text} !important; font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
}}
.stDownloadButton > button:hover {{ border-color: {ACCENT} !important; color: {ACCENT} !important; }}
[data-testid="stMetricValue"] {{ color: #fff !important; font-family: 'Syne', sans-serif !important; }}
[data-testid="stMetricLabel"] {{
    color: {sub} !important; font-family: 'DM Mono', monospace !important;
    font-size: 0.62rem !important;
}}
div[data-testid="stMarkdownContainer"] p {{ color: {text}; }}
.stAlert {{ border-radius: 10px !important; }}
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-thumb {{ background: {card_bdr}; border-radius: 2px; }}
.stDataFrame {{ border-radius: 12px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)


#  SIDEBAR NAV
def nav():
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-logo">
          <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24"
               fill="none" stroke="white" stroke-width="2.5"
               stroke-linecap="round" stroke-linejoin="round">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
            <line x1="12" y1="9"  x2="12"    y2="13"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
        </div>
        """, unsafe_allow_html=True)

        for label, key in PAGES:
            active = st.session_state.get("page") == label
            icon   = PAGE_ICONS[key]
            if st.button(icon, key=f"nav_{key}", help=label,
                         type="primary" if active else "secondary"):
                st.session_state.page = label
                st.rerun()


#  HELPERS
def kpi_card(label, val, hint, color):
    return (
        f'<div class="kpi">'
        f'<div class="kpi-accent" style="background:{color};"></div>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-val">{val}</div>'
        f'<div class="kpi-delta">{hint}</div>'
        f'</div>'
    )

def no_data_msg():
    st.markdown(
        f'<div class="chart-card" style="text-align:center;padding:52px;color:{PAL["sub"]};">'
        'Dataset not found. Run <code>data_cleaning.ipynb</code> first to generate '
        '<code>outputs/cleaned_tidy.csv</code>.</div>',
        unsafe_allow_html=True,
    )

def fmt_millions(n):
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.0f}K"
    return str(n)


#  PAGE 1: DASHBOARD
def page_executive():
    tidy, yearly, summary, heatmap_df = derived_data()
    if tidy is None: no_data_msg(); return

    st.markdown('<div class="pg-title">Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Philippines Disaster Impact · PSA Data · 2015–2023</div>', unsafe_allow_html=True)

    total_persons  = int(tidy["persons_affected"].sum())
    total_families = int(tidy["families_affected"].sum())
    top_hazard     = summary.iloc[0]["hazard_type"]
    peak_year      = int(yearly.loc[yearly["persons_affected"].idxmax(), "year"])
    peak_val       = int(yearly["persons_affected"].max())
    avg_annual     = int(yearly["persons_affected"].mean())
    n_hazards      = int(tidy["hazard_type"].nunique())
    top_pct        = summary.iloc[0]["total_persons"] / total_persons * 100

    cards = [
        ("Total Persons Affected",  fmt_millions(total_persons),           "2015–2023 cumulative",             ACCENT),
        ("Total Families Affected", fmt_millions(total_families),          "2015–2023 cumulative",             CYAN),
        ("Most Dangerous Hazard",   top_hazard.split("(")[0].strip()[:18], f"{top_pct:.0f}% of all incidents", ORANGE),
        ("Highest Impact Year",     str(peak_year),                        fmt_millions(peak_val)+" persons",  "#F63E31"),
        ("Avg Annual Impact",       fmt_millions(avg_annual),              "persons per year",                 GREEN),
        ("Hazard Types Tracked",    str(n_hazards),                        "unique categories",                PURPLE),
    ]
    st.markdown('<div class="kpi-row">' + "".join(kpi_card(l,v,h,c) for l,v,h,c in cards) + '</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2], gap="medium")
    with c1:
        st.markdown('<div class="chart-card"><div class="chart-title">Annual Impact Trend</div><div class="chart-sub">Total persons & families affected per year</div>', unsafe_allow_html=True)
        metric = st.radio("", ["Persons", "Families", "Both"], horizontal=True, key="exec_metric", label_visibility="collapsed")
        fig = go.Figure()
        if metric in ["Persons", "Both"]:
            fig.add_trace(go.Scatter(x=yearly["year"], y=yearly["persons_affected"],
                name="Persons", line=dict(color=ACCENT, width=2.5), mode="lines+markers",
                marker=dict(size=7, color=ACCENT, line=dict(color="white", width=1.5)),
                fill="tozeroy", fillcolor="rgba(246,62,49,0.08)",
                hovertemplate="<b>%{x}</b><br>Persons: %{y:,.0f}<extra></extra>"))
        if metric in ["Families", "Both"]:
            fig.add_trace(go.Scatter(x=yearly["year"], y=yearly["families_affected"],
                name="Families", line=dict(color=CYAN, width=2.5), mode="lines+markers",
                marker=dict(size=7, color=CYAN, line=dict(color="white", width=1.5)),
                fill="tozeroy", fillcolor="rgba(59,158,191,0.08)",
                hovertemplate="<b>%{x}</b><br>Families: %{y:,.0f}<extra></extra>"))
        fig.add_vline(x=peak_year, line_dash="dot", line_color=ACCENT, opacity=0.4)
        fig.add_annotation(x=peak_year, y=peak_val, text=f"Peak {peak_year}",
            showarrow=True, arrowhead=2, arrowcolor=ACCENT,
            font=dict(color=ACCENT, size=10), ax=30, ay=-30,
            bgcolor=PAL["card_bg"], bordercolor=ACCENT, borderwidth=1)
        apply_theme(fig)
        fig.update_layout(height=280, showlegend=(metric=="Both"), xaxis=dict(tickmode="linear", dtick=1))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card"><div class="chart-title">Hazard Share</div><div class="chart-sub">Top 6 by persons affected</div>', unsafe_allow_html=True)
        top6   = summary.head(6)
        others = total_persons - top6["total_persons"].sum()
        labels = list(top6["hazard_type"].str.split("(").str[0].str.strip()) + ["Others"]
        values = list(top6["total_persons"]) + [others]
        colors = [ACCENT, CYAN, ORANGE, GREEN, PURPLE, "#F6D860", PAL["card_bdr"]]
        fig = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.62, marker_colors=colors,
            textinfo="percent", textfont=dict(size=10, color="white"),
            hovertemplate="<b>%{label}</b><br>%{value:,.0f} persons<br>%{percent}<extra></extra>",
            sort=False))
        fig.add_annotation(
            text=f"<b>{fmt_millions(total_persons)}</b><br><span style='font-size:9px'>Total</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(color="white", size=14, family="Syne"))
        apply_theme(fig)
        fig.update_layout(height=280, showlegend=False, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    c3, c4 = st.columns([2, 3], gap="medium")
    with c3:
        st.markdown('<div class="chart-card"><div class="chart-title">Year-over-Year Change</div><div class="chart-sub">Δ persons affected vs prior year</div>', unsafe_allow_html=True)
        yoy_df = yearly.copy()
        yoy_df["yoy"] = yoy_df["persons_affected"].diff()
        yoy_df = yoy_df.dropna()
        fig = go.Figure(go.Bar(
            x=yoy_df["year"], y=yoy_df["yoy"],
            marker_color=[GREEN if v < 0 else ACCENT for v in yoy_df["yoy"]],
            hovertemplate="<b>%{x}</b><br>Δ %{y:+,.0f}<extra></extra>"))
        apply_theme(fig)
        fig.update_layout(height=240, xaxis=dict(tickmode="linear", dtick=1), yaxis=dict(tickformat=".2s"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="chart-card"><div class="chart-title">Key Insights</div><div class="chart-sub">Key findings from the dataset</div>', unsafe_allow_html=True)
        tc_pct = summary.iloc[0]["total_persons"] / total_persons * 100
        yoy_pct_df = yearly.copy()
        yoy_pct_df["yoy_pct"] = yoy_pct_df["persons_affected"].pct_change() * 100
        worst_jump     = yoy_pct_df.loc[yoy_pct_df["yoy_pct"].idxmax()]
        active_hazards = int((tidy.groupby("hazard_type")["had_impact"].sum() >= 5).sum())
        insights = [
            ("Dominant Threat",
             f"Tropical Cyclones account for {tc_pct:.0f}% of all affected persons from 2015 to 2023, "
             "making it the singular most impactful hazard category in the Philippines."),
            ("Peak Impact Year",
             f"{peak_year} recorded the highest annual toll with {fmt_millions(peak_val)} persons affected — "
             "driven by an intense typhoon season and compounding hazard events."),
            ("Escalating Trend",
             "Disaster impact increased significantly after 2018, with 2021–2023 consistently exceeding "
             "the 10 million persons affected mark each year."),
            ("Persistent Hazards",
             f"{active_hazards} hazard types recorded impact in 5 or more of the 9 years tracked, "
             "indicating recurring vulnerability across multiple categories."),
            ("Largest Single-Year Surge",
             f"{int(worst_jump['year'])} saw the sharpest year-over-year increase at "
             f"+{worst_jump['yoy_pct']:.0f}%, signalling the need for surge-response preparedness."),
        ]
        for lbl, txt in insights:
            st.markdown(
                f'<div class="insight-card"><div class="insight-label">{lbl}</div>'
                f'<div class="insight-text">{txt}</div></div>',
                unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


#  PAGE 2: DISASTER ANALYTICS
def page_analytics():
    tidy, yearly, summary, heatmap_df = derived_data()
    if tidy is None: no_data_msg(); return

    st.markdown('<div class="pg-title">Disaster Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Deep-dive into hazard categories, trends, and patterns</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1: top_n      = st.selectbox("Show top hazards", [5,10,15,20,"All"], index=1)
    with f2: metric     = st.selectbox("Metric", ["persons_affected","families_affected"])
    with f3: year_range = st.slider("Year range", 2015, 2023, (2015, 2023))

    tidy_f = tidy[(tidy["year"] >= year_range[0]) & (tidy["year"] <= year_range[1])].copy()
    summ_f = tidy_f.groupby("hazard_type")[metric].sum().sort_values(ascending=False).reset_index()
    if top_n != "All": summ_f = summ_f.head(int(top_n))
    summ_f = summ_f.sort_values(metric)

    st.markdown('<div class="chart-card"><div class="chart-title">Hazard Rankings</div><div class="chart-sub">Ranked by total affected. You may hover for details.</div>', unsafe_allow_html=True)
    n = len(summ_f)
    bar_colors = [f"rgba(246,62,49,{0.4 + 0.6*(i/max(n-1,1)):.2f})" for i in range(n)]
    fig = go.Figure(go.Bar(
        x=summ_f[metric], y=summ_f["hazard_type"].str[:35],
        orientation="h", marker_color=bar_colors,
        text=summ_f[metric].apply(fmt_millions), textposition="outside",
        textfont=dict(color=PAL["sub"], size=9),
        hovertemplate="<b>%{y}</b><br>%{x:,.0f}<extra></extra>"))
    apply_theme(fig)
    fig.update_layout(height=max(300, n*32), xaxis=dict(tickformat=".2s"), yaxis=dict(tickfont=dict(size=9)))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    # ── FIX: Trend Per-Hazard ─────────────────────────────────────────────
    # Only offer hazards that actually have data in the filtered year range
    st.markdown('<div class="chart-card"><div class="chart-title">Trend Per-Hazard</div><div class="chart-sub">Select hazards to compare over time</div>', unsafe_allow_html=True)
    all_hazards = sorted(tidy_f["hazard_type"].unique().tolist())
    # Default: top 3 hazards that exist in the current filter
    top_hazards_in_range = (
        tidy_f.groupby("hazard_type")[metric].sum()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )
    selected = st.multiselect(
        "Hazard types", all_hazards,
        default=[h for h in top_hazards_in_range if h in all_hazards],
        key="analytics_sel"
    )
    if selected:
        colors_map = [ACCENT, CYAN, ORANGE, GREEN, PURPLE, "#F6D860"]
        fig = go.Figure()
        sel_df = tidy_f[tidy_f["hazard_type"].isin(selected)].copy()

        for i, h in enumerate(selected):
            d = sel_df[sel_df["hazard_type"] == h].sort_values("year")
            if len(d) == 0:
                continue
            color = colors_map[i % len(colors_map)]
            label = h.split("(")[0].strip()[:25]

            # KEY FIX: if only 1 data point, switch to markers-only with a
            # larger marker so it is actually visible; add a text annotation
            if len(d) == 1:
                fig.add_trace(go.Scatter(
                    x=d["year"], y=d[metric],
                    name=label,
                    mode="markers+text",
                    marker=dict(size=14, color=color,
                                line=dict(color="white", width=2),
                                symbol="circle"),
                    text=[fmt_millions(int(d[metric].iloc[0]))],
                    textposition="top center",
                    textfont=dict(color=color, size=10),
                    hovertemplate=f"<b>{h[:30]}</b><br>Year: %{{x}}<br>{metric}: %{{y:,.0f}}<extra></extra>",
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=d["year"], y=d[metric],
                    name=label,
                    line=dict(color=color, width=2.5),
                    mode="lines+markers",
                    marker=dict(size=7, color=color,
                                line=dict(color="white", width=1.5)),
                    hovertemplate=f"<b>{h[:30]}</b><br>Year: %{{x}}<br>{metric}: %{{y:,.0f}}<extra></extra>",
                ))

        apply_theme(fig)
        fig.update_layout(
            height=320,
            xaxis=dict(
                tickmode="linear", dtick=1,
                range=[year_range[0] - 0.5, year_range[1] + 0.5],
            ),
            yaxis=dict(tickformat=".2s"),
            legend=dict(orientation="h", y=-0.22, font=dict(size=9)),
            margin=dict(l=10, r=10, t=30, b=50),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Select at least one hazard type above.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card"><div class="chart-title">Summary Statistics</div><div class="chart-sub">Aggregated metrics per hazard type for selected period</div>', unsafe_allow_html=True)
    agg = (tidy_f.groupby("hazard_type").agg(
        total_persons=("persons_affected","sum"), total_families=("families_affected","sum"),
        peak_persons=("persons_affected","max"), active_years=("had_impact","sum"))
        .sort_values("total_persons", ascending=False).reset_index())
    agg["total_persons"]  = agg["total_persons"].apply(lambda x: f"{x:,.0f}")
    agg["total_families"] = agg["total_families"].apply(lambda x: f"{x:,.0f}")
    agg["peak_persons"]   = agg["peak_persons"].apply(lambda x: f"{x:,.0f}")
    agg.columns = ["Hazard Type","Total Persons","Total Families","Peak Year Persons","Active Years"]
    st.dataframe(agg, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Major Hazard Category Breakdown ───────────────────────────────────
    st.markdown('<div class="chart-card"><div class="chart-title">Major Hazard Category Breakdown</div>'
                '<div class="chart-sub">Meteorological · Hydrological · Geophysical · '
                'Climatological · Biological · Combined Events</div>',
                unsafe_allow_html=True)

    cat_df = tidy_f.copy()
    cat_df["category"] = cat_df["hazard_type"].apply(assign_category)
    cat_summary = (
        cat_df.groupby("category")[metric]
        .sum()
        .reindex(list(HAZARD_CATEGORIES.keys()), fill_value=0)
        .reset_index()
    )
    cat_summary.columns = ["category", "value"]

    row1_l, row1_r = st.columns([1, 1], gap="medium")

    with row1_l:
        nonzero = cat_summary[cat_summary["value"] > 0]
        fig = go.Figure(go.Pie(
            labels=nonzero["category"],
            values=nonzero["value"],
            hole=0.60,
            marker_colors=[CAT_COLORS[c] for c in nonzero["category"]],
            textinfo="label+percent",
            textfont=dict(size=10, color="white"),
            insidetextorientation="radial",
            hovertemplate="<b>%{label}</b><br>%{value:,.0f} "
                          + ("persons" if metric == "persons_affected" else "families")
                          + "<br>%{percent}<extra></extra>",
            sort=True,
        ))
        total_cat = int(cat_summary["value"].sum())
        fig.add_annotation(
            text=f"<b>{fmt_millions(total_cat)}</b><br>"
                 f"<span style='font-size:9px'>{'Persons' if metric=='persons_affected' else 'Families'}</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="white", size=13, family="Syne"),
        )
        apply_theme(fig)
        fig.update_layout(
            height=320, showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            title=dict(text="Category Share", font=dict(size=11, color=PAL["sub"], family="DM Mono"), x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with row1_r:
        cat_sorted = cat_summary.sort_values("value")
        fig = go.Figure(go.Bar(
            x=cat_sorted["value"],
            y=cat_sorted["category"],
            orientation="h",
            marker_color=[CAT_COLORS[c] for c in cat_sorted["category"]],
            marker_line_width=0,
            text=cat_sorted["value"].apply(fmt_millions),
            textposition="outside",
            textfont=dict(color=PAL["sub"], size=10),
            hovertemplate="<b>%{y}</b><br>%{x:,.0f}<extra></extra>",
        ))
        apply_theme(fig)
        fig.update_layout(
            height=320,
            xaxis=dict(tickformat=".2s", showgrid=True),
            yaxis=dict(tickfont=dict(size=10)),
            margin=dict(l=10, r=60, t=30, b=10),
            title=dict(text="Total Affected by Category", font=dict(size=11, color=PAL["sub"], family="DM Mono"), x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    cat_year = (
        cat_df.groupby(["year", "category"])[metric]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=list(HAZARD_CATEGORIES.keys()), fill_value=0)
        .reset_index()
    )
    fig = go.Figure()
    for cat in list(HAZARD_CATEGORIES.keys()):
        if cat in cat_year.columns and cat_year[cat].sum() > 0:
            fig.add_trace(go.Bar(
                x=cat_year["year"], y=cat_year[cat], name=cat,
                marker_color=CAT_COLORS[cat],
                hovertemplate=f"<b>{cat}</b><br>Year: %{{x}}<br>%{{y:,.0f}}<extra></extra>",
            ))
    apply_theme(fig)
    fig.update_layout(
        barmode="stack", height=300,
        xaxis=dict(tickmode="linear", dtick=1),
        yaxis=dict(tickformat=".2s", title="Persons Affected" if metric == "persons_affected" else "Families Affected"),
        legend=dict(orientation="h", y=-0.22, font=dict(size=9), traceorder="normal"),
        margin=dict(l=10, r=10, t=30, b=40),
        title=dict(text="Category Impact by Year (Stacked)", font=dict(size=11, color=PAL["sub"], family="DM Mono"), x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    top_cat  = cat_summary.loc[cat_summary["value"].idxmax(), "category"]
    top_val  = int(cat_summary["value"].max())
    top_pct  = top_val / total_cat * 100 if total_cat > 0 else 0
    n_active = int((cat_summary["value"] > 0).sum())
    cat_year_top = cat_df[cat_df["category"] == top_cat].groupby("year")[metric].sum()
    peak_yr = int(cat_year_top.idxmax()) if len(cat_year_top) > 0 else "—"

    kpi_html = (
        kpi_card("Dominant Category",   top_cat,               f"{top_pct:.0f}% of total",    CAT_COLORS[top_cat])
        + kpi_card("Peak Impact",       fmt_millions(top_val), top_cat,                        ACCENT)
        + kpi_card("Active Categories", str(n_active),         f"out of {len(HAZARD_CATEGORIES)}", CYAN)
        + kpi_card("Peak Year",         str(peak_yr),          f"for {top_cat}",               ORANGE)
    )
    st.markdown(f'<div class="kpi-row" style="margin-top:14px;">{kpi_html}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


#  PAGE 3: VISUALIZATIONS
def page_visuals():
    tidy, yearly, summary, heatmap_df = derived_data()
    if tidy is None: no_data_msg(); return

    st.markdown('<div class="pg-title">Interactive Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Heatmaps, timelines, and advanced visual analytics</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card"><div class="chart-title">Disaster Intensity Heatmap</div><div class="chart-sub">Persons affected per hazard type per year — darker = higher impact</div>', unsafe_allow_html=True)
    top20 = summary.head(20)["hazard_type"].tolist()
    hm    = heatmap_df.loc[heatmap_df.index.isin(top20)]
    fig = go.Figure(go.Heatmap(
        z=hm.values, x=[str(c) for c in hm.columns], y=[h[:35] for h in hm.index],
        colorscale=[[0,"rgba(11,24,32,0.9)"],[0.3,"rgba(59,158,191,0.6)"],[0.7,"rgba(246,62,49,0.8)"],[1,"rgba(246,62,49,1)"]],
        hovertemplate="<b>%{y}</b><br>Year: %{x}<br>Persons: %{z:,.0f}<extra></extra>",
        showscale=True, colorbar=dict(tickfont=dict(color=PAL["sub"],size=9), bgcolor="rgba(0,0,0,0)", bordercolor=PAL["card_bdr"])))
    apply_theme(fig)
    fig.update_layout(height=520, yaxis=dict(tickfont=dict(size=9)))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    # ── FIX: Bubble Timeline ──────────────────────────────────────────────
    st.markdown('<div class="chart-card"><div class="chart-title">Impact Bubble Timeline</div>'
                '<div class="chart-sub">Bubble size = persons affected · Animated by year · '
                'Each hazard shown in every frame</div>', unsafe_allow_html=True)

    top10_h = summary.head(10)["hazard_type"].tolist()
    all_years = sorted(tidy["year"].unique())

    # Build a COMPLETE cross-product so every hazard appears in every frame.
    # Plotly animation requires the same set of traces in every frame or
    # bubbles disappear / the animation stalls.
    idx = pd.MultiIndex.from_product([top10_h, all_years], names=["hazard_type", "year"])
    full_grid = pd.DataFrame(index=idx).reset_index()

    raw = (
        tidy[tidy["hazard_type"].isin(top10_h)][["hazard_type", "year", "persons_affected", "families_affected"]]
        .groupby(["hazard_type", "year"], as_index=False)
        .sum()
    )

    bubble_df = full_grid.merge(raw, on=["hazard_type", "year"], how="left")
    bubble_df["persons_affected"]  = bubble_df["persons_affected"].fillna(0)
    bubble_df["families_affected"] = bubble_df["families_affected"].fillna(0)

    # Rank is fixed so hazards keep their horizontal position across frames
    rank_map = {h: i for i, h in enumerate(top10_h)}
    bubble_df["rank"] = bubble_df["hazard_type"].map(rank_map)

    # px.scatter needs a positive size column; use a tiny minimum so zero
    # values produce a visible (but tiny) placeholder bubble
    MIN_BUBBLE = bubble_df[bubble_df["persons_affected"] > 0]["persons_affected"].min() * 0.05
    bubble_df["bubble_size"] = bubble_df["persons_affected"].clip(lower=MIN_BUBBLE)

    short_labels = [h.split("(")[0].strip()[:20] for h in top10_h]
    bubble_df["label"] = bubble_df["hazard_type"].map(
        {h: short_labels[i] for i, h in enumerate(top10_h)}
    )

    bubble_colors = [ACCENT, CYAN, ORANGE, GREEN, PURPLE,
                     "#F6D860", "#A8DADC", "#E9C46A", "#264653", "#2A9D8F"]

    fig = px.scatter(
        bubble_df.sort_values("year"),
        x="rank",
        y="persons_affected",
        size="bubble_size",
        color="hazard_type",
        animation_frame="year",
        size_max=80,
        hover_name="label",
        hover_data={
            "persons_affected":  ":,.0f",
            "families_affected": ":,.0f",
            "rank":              False,
            "bubble_size":       False,
        },
        color_discrete_sequence=bubble_colors,
        labels={"persons_affected": "Persons Affected", "rank": "Hazard"},
        category_orders={"hazard_type": top10_h},
    )

    # Replace numeric x-axis with short hazard name labels
    apply_theme(fig)
    fig.update_layout(
        height=420,
        showlegend=False,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(top10_h))),
            ticktext=short_labels,
            tickfont=dict(size=9),
            title="",
        ),
        yaxis=dict(tickformat=".2s", title="Persons Affected"),
        margin=dict(l=10, r=10, t=40, b=60),
        # Speed up the animation slightly
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.15, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶ Play",  method="animate",
                     args=[None, {"frame": {"duration": 700, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 300}}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate", "transition": {"duration": 0}}]),
            ],
        )],
        sliders=[dict(
            currentvalue=dict(prefix="Year: ", font=dict(color=PAL["sub"], size=12)),
            pad=dict(t=10),
            font=dict(color=PAL["sub"]),
        )],
    )
    fig.update_traces(
        marker=dict(opacity=0.85, line=dict(width=1.5, color="white"))
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown('<div class="chart-card"><div class="chart-title">Disaster Timeline</div><div class="chart-sub">Peak years and impact magnitude</div>', unsafe_allow_html=True)
        max_val = int(yearly["persons_affected"].max())
        tl_html = ""
        for _, row in yearly.iterrows():
            yr  = int(row["year"]); val = int(row["persons_affected"]); pct = val / max_val
            dot = ACCENT if pct > 0.8 else (ORANGE if pct > 0.5 else CYAN)
            vc  = "#fff" if pct > 0.8 else PAL["sub"]
            tl_html += (
                f'<div class="timeline-item">'
                f'<div class="timeline-dot" style="background:{dot};color:{dot};"></div>'
                f'<div style="flex:1;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span class="timeline-year">{yr}</span>'
                f'<span style="font-family:\'DM Mono\',monospace;font-size:0.8rem;color:{vc};">{fmt_millions(val)} persons</span>'
                f'</div>'
                f'<div style="background:{PAL["grid"]};border-radius:3px;height:4px;margin-top:6px;">'
                f'<div style="background:{dot};width:{pct*100:.1f}%;height:4px;border-radius:3px;"></div>'
                f'</div></div></div>'
            )
        st.markdown(tl_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card"><div class="chart-title">Stacked Area — Top 5 Hazards</div><div class="chart-sub">Cumulative impact breakdown by year</div>', unsafe_allow_html=True)
        top5       = summary.head(5)["hazard_type"].tolist()
        area_pivot = tidy[tidy["hazard_type"].isin(top5)].pivot_table(
            index="year", columns="hazard_type", values="persons_affected", aggfunc="sum", fill_value=0)
        area_colors = [ACCENT, CYAN, ORANGE, GREEN, PURPLE]
        fig = go.Figure()
        for i, col in enumerate(top5):
            if col in area_pivot.columns:
                fig.add_trace(go.Scatter(
                    x=area_pivot.index, y=area_pivot[col], name=col[:22], mode="lines",
                    stackgroup="one", line=dict(color=area_colors[i], width=0),
                    fillcolor=hex_to_rgba(area_colors[i], 0.7),
                    hovertemplate=f"<b>{col[:25]}</b><br>%{{x}}: %{{y:,.0f}}<extra></extra>"))
        apply_theme(fig)
        fig.update_layout(height=300, xaxis=dict(tickmode="linear",dtick=1), legend=dict(orientation="h",y=-0.2,font=dict(size=9)))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card"><div class="chart-title">Trend & Moving Average</div><div class="chart-sub">3-year rolling average with simple linear forecast</div>', unsafe_allow_html=True)
    yr_vals = yearly.copy()
    yr_vals["rolling_avg"] = yr_vals["persons_affected"].rolling(3, min_periods=1).mean()
    coeffs = np.polyfit(np.arange(len(yr_vals)), yr_vals["persons_affected"], 1)
    f_years = [2024,2025,2026]
    f_vals  = [int(coeffs[0]*(len(yr_vals)+i)+coeffs[1]) for i in range(3)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=yr_vals["year"], y=yr_vals["persons_affected"], name="Actual",
        marker_color="rgba(246,62,49,0.35)", hovertemplate="<b>%{x}</b><br>Actual: %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=yr_vals["year"], y=yr_vals["rolling_avg"], name="3-yr Avg",
        line=dict(color=CYAN,width=2.5), hovertemplate="<b>%{x}</b><br>Rolling avg: %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=f_years, y=f_vals, name="Forecast",
        line=dict(color=ORANGE,width=2,dash="dot"), marker=dict(size=8,symbol="diamond",color=ORANGE),
        hovertemplate="<b>%{x}</b><br>Forecast: %{y:,.0f}<extra></extra>"))
    apply_theme(fig)
    fig.update_layout(height=300, xaxis=dict(tickmode="linear",dtick=1), yaxis=dict(tickformat=".2s"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


#  PAGE 4: CLASSIFIER
def page_classify(model):
    st.markdown('<div class="pg-title">Image Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Upload a disaster image for real-time disaster classification</div>', unsafe_allow_html=True)

    if model is None:
        st.warning("Model not found at `models/disaster_classifier.keras`. The analytics dashboard works independently of the model.")
    else:
        left, right = st.columns(2, gap="large")
        with left:
            uploaded = st.file_uploader("Drop image or click to browse", type=["jpg","jpeg","png"])
            if uploaded:
                st.image(Image.open(uploaded), caption=uploaded.name, use_container_width=True)
                run = st.button("Run Classification", use_container_width=True)
            else:
                st.markdown(
                    f'<div class="result-box" style="text-align:center;padding:52px;color:{PAL["sub"]};">'
                    'JPG · JPEG · PNG supported</div>', unsafe_allow_html=True)
                run = False

        with right:
            if uploaded and run:
                with st.spinner("Analyzing image…"):
                    cls, conf, probs = run_predict(model, Image.open(uploaded))
                    save_prediction(uploaded.name, cls, conf)
                lo      = conf < 0.60
                c_color = "#f97316" if lo else CLS_HEX.get(cls, ACCENT)
                warn_html = (
                    '<div style="background:#f9731618;border:1px solid #f9731644;border-radius:8px;'
                    'padding:9px 14px;margin-bottom:14px;font-size:0.75rem;color:#f97316;'
                    'font-family:DM Mono,monospace;">LOW CONFIDENCE — verify manually</div>'
                ) if lo else ""
                st.markdown(
                    f'<div class="result-box">{warn_html}'
                    f'<div style="font-family:DM Mono,monospace;font-size:0.6rem;color:{PAL["sub"]};'
                    f'text-transform:uppercase;letter-spacing:0.14em;">Predicted Class</div>'
                    f'<div class="result-cls" style="color:{c_color};">{cls.upper()}</div>'
                    f'<div class="result-conf">Confidence: <strong style="color:#fff;">{conf*100:.1f}%</strong></div></div>',
                    unsafe_allow_html=True)
                st.markdown(
                    f'<div style="font-family:DM Mono,monospace;font-size:0.6rem;color:{PAL["sub"]};'
                    f'text-transform:uppercase;letter-spacing:0.14em;margin:20px 0 12px;">All Class Probabilities</div>',
                    unsafe_allow_html=True)
                bars_html = ""
                for i, c in enumerate(CLASS_NAMES):
                    pv    = float(probs[i])
                    color = CLS_HEX.get(c, ACCENT)
                    bars_html += (
                        f'<div class="pbar-row">'
                        f'<div class="pbar-meta"><span>{c.title()}</span><span>{pv*100:.1f}%</span></div>'
                        f'<div class="pbar-track"><div class="pbar-fill" style="width:{pv*100:.1f}%;background:{color};"></div></div>'
                        f'</div>'
                    )
                st.markdown(bars_html, unsafe_allow_html=True)
            elif not uploaded:
                st.markdown(
                    f'<div class="result-box" style="text-align:center;padding:52px;color:{PAL["sub"]};">'
                    'Results will appear here after classification.</div>', unsafe_allow_html=True)

    st.markdown("---")
    hist_df = load_predictions()
    if len(hist_df) > 0:
        st.markdown('<div class="chart-card"><div class="chart-title">Classification Activity</div><div class="chart-sub">Frequency and confidence of all images classified so far</div>', unsafe_allow_html=True)

        ch1, ch2, ch3 = st.columns(3, gap="medium")

        with ch1:
            counts = hist_df["pred_class"].value_counts().reset_index()
            counts.columns = ["class", "count"]
            counts["color"] = counts["class"].map(CLS_HEX)
            fig = go.Figure(go.Pie(
                labels=[c.title() for c in counts["class"]],
                values=counts["count"], hole=0.6,
                marker_colors=list(counts["color"]),
                textinfo="percent", textfont=dict(size=10, color="white"),
                hovertemplate="<b>%{label}</b><br>%{value} images<br>%{percent}<extra></extra>",
                sort=False))
            fig.add_annotation(
                text=f"<b>{len(hist_df)}</b><br><span style='font-size:9px'>Total</span>",
                x=0.5, y=0.5, showarrow=False, font=dict(color="white", size=14, family="Syne"))
            apply_theme(fig)
            fig.update_layout(height=240, showlegend=True, margin=dict(l=0,r=0,t=10,b=0),
                legend=dict(orientation="h", y=-0.15, font=dict(size=9)))
            st.markdown('<div style="font-family:DM Mono,monospace;font-size:0.6rem;color:' + PAL["sub"] + ';text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Class Distribution</div>', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with ch2:
            avg_conf = hist_df.groupby("pred_class")["confidence"].mean().reset_index()
            avg_conf = avg_conf.sort_values("confidence")
            avg_conf["color"] = avg_conf["pred_class"].map(CLS_HEX)
            fig = go.Figure(go.Bar(
                x=avg_conf["confidence"] * 100,
                y=[c.title() for c in avg_conf["pred_class"]],
                orientation="h",
                marker_color=list(avg_conf["color"]),
                text=[f"{v*100:.1f}%" for v in avg_conf["confidence"]],
                textposition="outside",
                textfont=dict(color=PAL["sub"], size=9),
                hovertemplate="<b>%{y}</b><br>Avg confidence: %{x:.1f}%<extra></extra>"))
            apply_theme(fig)
            fig.update_layout(height=240, xaxis=dict(range=[0,115], ticksuffix="%"),
                yaxis=dict(tickfont=dict(size=10)), margin=dict(l=10,r=30,t=10,b=10))
            st.markdown('<div style="font-family:DM Mono,monospace;font-size:0.6rem;color:' + PAL["sub"] + ';text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Avg Confidence per Class</div>', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with ch3:
            hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
            hist_df["date"] = hist_df["timestamp"].dt.strftime("%Y-%m-%d")
            trend = hist_df.groupby(["date","pred_class"]).size().unstack(fill_value=0)
            trend = trend.reindex(columns=CLASS_NAMES, fill_value=0)
            fig = go.Figure()
            for c in CLASS_NAMES:
                if c in trend.columns and trend[c].sum() > 0:
                    fig.add_trace(go.Scatter(
                        x=trend.index, y=trend[c], name=c.title(), mode="lines+markers",
                        line=dict(color=CLS_HEX[c], width=2), marker=dict(size=5),
                        hovertemplate=f"<b>{c.title()}</b><br>%{{x}}<br>Count: %{{y}}<extra></extra>"))
            apply_theme(fig)
            fig.update_layout(height=240, showlegend=True,
                legend=dict(orientation="h", y=-0.25, font=dict(size=9)),
                xaxis=dict(tickangle=25, tickfont=dict(size=8)),
                margin=dict(l=10,r=10,t=10,b=30))
            st.markdown('<div style="font-family:DM Mono,monospace;font-size:0.6rem;color:' + PAL["sub"] + ';text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Classification Trend</div>', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="chart-card" style="text-align:center;padding:32px;color:{PAL["sub"]};">'
            'No classifications yet — upload and classify an image above to see activity charts.</div>',
            unsafe_allow_html=True)

    with st.expander("Model Performance Details", expanded=False):
        c5, c6 = st.columns(2, gap="medium")
        with c5:
            cm = os.path.join(EVAL_DIR, "confusion_matrix.png")
            if os.path.exists(cm):
                st.image(cm, use_container_width=True)
            else:
                st.info("confusion_matrix.png not found in eval/ folder.")
        with c6:
            mpath = os.path.join(EVAL_DIR, "metrics.json")
            if os.path.exists(mpath):
                with open(mpath) as f:
                    metrics = json.load(f)
                wr = metrics.get("classification_report", {}).get("weighted avg", {})
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0)*100:.2f}%")
                    st.metric("Precision",     f"{wr.get('precision', 0)*100:.2f}%")
                with m2:
                    st.metric("Recall",   f"{wr.get('recall', 0)*100:.2f}%")
                    st.metric("F1-Score", f"{wr.get('f1-score', 0)*100:.2f}%")
                rows = [{
                    "Class":     c.title(),
                    "Precision": f"{metrics['classification_report'].get(c, {}).get('precision', 0)*100:.1f}%",
                    "Recall":    f"{metrics['classification_report'].get(c, {}).get('recall', 0)*100:.1f}%",
                    "F1":        f"{metrics['classification_report'].get(c, {}).get('f1-score', 0)*100:.1f}%",
                    "Support":   int(metrics['classification_report'].get(c, {}).get('support', 0)),
                } for c in CLASS_NAMES]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("metrics.json not found in eval/ folder.")
        curve = os.path.join(EVAL_DIR, "training_curves.png")
        if os.path.exists(curve):
            st.image(curve, use_container_width=True)


#  PAGE 5: PREDICTION HISTORY
def page_history():
    st.markdown('<div class="pg-title">Prediction History</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">All AI classifications stored in local database</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: flt      = st.selectbox("Class", ["All"] + CLASS_NAMES)
    with c2: conf_min = st.slider("Min Confidence (%)", 0, 100, 0, 5) / 100
    with c3: sort_by  = st.selectbox("Sort", ["Newest First","Oldest First","Highest Confidence","Lowest Confidence"])

    df = load_predictions(flt, conf_min)
    if sort_by == "Oldest First":         df = df.sort_values("id")
    elif sort_by == "Highest Confidence": df = df.sort_values("confidence", ascending=False)
    elif sort_by == "Lowest Confidence":  df = df.sort_values("confidence")

    if len(df) == 0:
        st.markdown(
            f'<div class="chart-card" style="text-align:center;color:{PAL["sub"]};padding:32px;">'
            'No predictions yet — use the AI Classifier page to get started.</div>',
            unsafe_allow_html=True)
        return

    st.markdown(
        '<div class="kpi-row">'
        + kpi_card("Showing",         str(len(df)),                               "records",          ACCENT)
        + kpi_card("Avg Confidence",  f"{df['confidence'].mean()*100:.1f}%",      "across filtered",  CYAN)
        + kpi_card("High Confidence", str(int((df['confidence'] >= 0.80).sum())), "above 80%",        GREEN)
        + '</div>', unsafe_allow_html=True)

    disp = df[["image_name","pred_class","confidence","timestamp"]].copy()
    disp.columns = ["Image","Class","Confidence","Timestamp"]
    disp["Confidence"] = disp["Confidence"].apply(lambda x: f"{x*100:.1f}%")
    disp["Class"]      = disp["Class"].apply(str.title)
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.download_button("Export CSV", df.to_csv(index=False).encode(), "prediction_history.csv", "text/csv")


#  PAGE 6: DATASET EXPLORER
def page_explorer():
    tidy, yearly, summary, heatmap_df = derived_data()
    if tidy is None: no_data_msg(); return

    st.markdown('<div class="pg-title">Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Filter, search, and export the cleaned PSA dataset</div>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    with f1: hazard_sel  = st.selectbox("Hazard Type", ["All"] + sorted(tidy["hazard_type"].unique().tolist()))
    with f2: year_sel    = st.multiselect("Year(s)", sorted(tidy["year"].unique().tolist()), default=sorted(tidy["year"].unique().tolist()))
    with f3: min_persons = st.number_input("Min Persons Affected", min_value=0, value=0, step=1000)
    with f4: sort_col    = st.selectbox("Sort by", ["persons_affected","families_affected","year","hazard_type"])

    df_view = tidy.copy()
    if hazard_sel != "All": df_view = df_view[df_view["hazard_type"] == hazard_sel]
    if year_sel:            df_view = df_view[df_view["year"].isin(year_sel)]
    df_view = df_view[df_view["persons_affected"] >= min_persons]
    df_view = df_view.sort_values(sort_col, ascending=(sort_col in ["hazard_type","year"]))

    st.markdown(
        '<div class="kpi-row">'
        + kpi_card("Rows",           f"{len(df_view):,}",                                 "matching filters", ACCENT)
        + kpi_card("Total Persons",  fmt_millions(int(df_view["persons_affected"].sum())), "filtered total",   CYAN)
        + kpi_card("Total Families", fmt_millions(int(df_view["families_affected"].sum())),"filtered total",   GREEN)
        + kpi_card("Hazard Types",   str(df_view["hazard_type"].nunique()),                "in selection",     ORANGE)
        + '</div>', unsafe_allow_html=True)

    disp = df_view[["hazard_type","year","families_affected","persons_affected","avg_persons_per_family","persons_yoy_change"]].copy()
    disp.columns = ["Hazard Type","Year","Families Affected","Persons Affected","Avg Persons/Family","YoY Change"]
    disp["Families Affected"] = disp["Families Affected"].apply(lambda x: f"{x:,.0f}")
    disp["Persons Affected"]  = disp["Persons Affected"].apply(lambda x: f"{x:,.0f}")
    disp["YoY Change"]        = disp["YoY Change"].apply(lambda x: f"{x:+,.0f}")
    st.dataframe(disp, use_container_width=True, hide_index=True, height=420)
    st.download_button("Export Filtered CSV", df_view.to_csv(index=False).encode(), "filtered_dataset.csv", "text/csv")


#  MAIN
def main():
    init_db()
    if "page" not in st.session_state:
        st.session_state.page = "Executive Dashboard"
    inject_css()
    nav()

    model = load_model()
    pg    = st.session_state.page
    if   pg == "Executive Dashboard": page_executive()
    elif pg == "Disaster Analytics":  page_analytics()
    elif pg == "Visualizations":      page_visuals()
    elif pg == "AI Classifier":       page_classify(model)
    elif pg == "Prediction History":  page_history()
    elif pg == "Dataset Explorer":    page_explorer()


if __name__ == "__main__":
    main()