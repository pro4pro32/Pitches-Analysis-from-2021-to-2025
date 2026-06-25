"""
spray_zone_dashboard.py
═══════════════════════
Click a zone + select pitch type → instantly see:
  • Spray angle bins  (horizontal direction, 10° slices left→right)
  • Launch angle bins (0-10°, 10-20°, 20-30°, 30-35°, 35-45°, 45+°)
  • Joint 2-D heatmap (spray × launch)
  • Contingency tables per zone / per pitch type

Files needed (same folder): statcast_raw_YYYY.parquet  (e.g. 2021-2025)
Key columns used:
  plate_x, plate_z            pitch location (feet, catcher view)
  spray_angle                 horizontal batted-ball direction (−45..+45°)
  launch_angle                vertical batted-ball angle
  pitch_type                  FF / CH / SL / CU / SI …
  p_throws, stand             handedness
  release_speed               mph
  release_spin_rate           rpm
  launch_speed                exit velocity mph
  is_hit, estimated_woba_using_speedangle
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
DATA_DIR    = Path(".")
AVAIL_YEARS = [y for y in range(2015, 2027)
               if (DATA_DIR / f"statcast_raw_{y}.parquet").exists()]

# Zone grid — Statcast 3×3 (catcher's view, feet)
#   Z1 | Z2 | Z3   high
#   Z4 | Z5 | Z6   mid
#   Z7 | Z8 | Z9   low
X_EDGES = [-0.71, -0.237, 0.237, 0.71]
Z_EDGES = [ 1.50,  2.17,  2.83,  3.50]

ZONE_BOUNDS: dict = {}
for _r in range(3):
    for _c in range(3):
        _z = _r * 3 + _c + 1
        ZONE_BOUNDS[_z] = (
            X_EDGES[_c],     Z_EDGES[2 - _r],
            X_EDGES[_c + 1], Z_EDGES[2 - _r + 1],
        )

ZONE_DESC = {
    1: "High-Inside (RHB)",  2: "High-Center",      3: "High-Outside (RHB)",
    4: "Mid-Inside (RHB)",   5: "Center",            6: "Mid-Outside (RHB)",
    7: "Low-Inside (RHB)",   8: "Low-Center",        9: "Low-Outside (RHB)",
}

PITCH_NAMES = {
    "FF":"Four-Seam","SI":"Sinker","FC":"Cutter",
    "SL":"Slider","ST":"Sweeper","CH":"Changeup",
    "CU":"Curveball","KC":"Knuckle-C","SV":"Slurve","FS":"Splitter",
}
PITCH_COLORS = {
    "FF":"#ef4444","SI":"#f97316","FC":"#f59e0b","SL":"#eab308","ST":"#84cc16",
    "CH":"#22c55e","CU":"#06b6d4","KC":"#3b82f6","SV":"#8b5cf6","FS":"#ec4899",
}

# Spray-angle bins (horizontal, negative = LF)
SPRAY_BINS   = [-180, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 180]
SPRAY_LABELS = [
    "< -45°","-45/-35°","-35/-25°","-25/-15°","-15/-5°","-5/+5°",
    "+5/+15°","+15/+25°","+25/+35°","+35/+45°","> +45°",
]
SPRAY_COLORS = [
    "#dc2626","#ef4444","#f97316","#f59e0b","#eab308",
    "#22c55e","#16a34a","#0ea5e9","#3b82f6","#6366f1","#8b5cf6",
]

# Launch-angle bins (vertical)
LA_BINS   = [-90, 0, 10, 20, 30, 35, 45, 90]
LA_LABELS = [
    "< 0° (GB)","0-10°","10-20°","20-30°",
    "30-35°","35-45° ★","> 45° (PU)",
]
LA_COLORS = [
    "#64748b","#22d3ee","#34d399","#facc15",
    "#fb923c","#f87171","#c084fc",
]

# Dark-theme colours
BG   = "#0b0f17"
BG2  = "#111621"
GRID = "#1e2535"
TXT  = "#e2e8f0"
SUB  = "#8892a4"
ACC  = "#2f7cf6"

NEED_COLS = [
    "plate_x","plate_z","spray_angle","launch_angle",
    "pitch_type","p_throws","stand",
    "release_speed","release_spin_rate",
    "launch_speed","is_hit",
    "estimated_woba_using_speedangle",
    "player_name","game_date",
]

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG + CSS
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Zone → Spray & Launch", page_icon="⚾",
    layout="wide", initial_sidebar_state="expanded",
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[data-testid="stAppViewContainer"]{background:#0b0f17!important;color:#e2e8f0;font-family:'DM Sans',sans-serif;}
[data-testid="stSidebar"]{background:#111621!important;border-right:1px solid #1e2535;}
[data-testid="stSidebar"] *{color:#e2e8f0!important;}
p,div,label,span,td,th,h1,h2,h3{color:#e2e8f0;}
.ttl{font-size:1.75rem;font-weight:700;color:#f0f6ff;letter-spacing:-.4px;}
.sub{font-size:.82rem;color:#7a8494;margin:2px 0 16px;}
.sec{display:flex;align-items:center;gap:8px;
  background:linear-gradient(90deg,#1a3a6622,transparent);
  border-left:3px solid #2f7cf6;padding:8px 16px;border-radius:0 8px 8px 0;
  margin:20px 0 12px;color:#79b8ff!important;font-weight:600;font-size:.94rem;}
.icard{background:#111621;border:1px solid #1e2535;border-left:3px solid #2f7cf6;
  border-radius:4px;padding:.6rem .9rem;font-size:.82rem;color:#8fa8c8;margin:.4rem 0 .8rem;}
[data-testid="metric-container"]{background:#0f1923;border:1px solid #1e2535;border-radius:6px;padding:.65rem .85rem;}
[data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace!important;color:#57d9a3!important;font-size:1.4rem!important;}
[data-testid="stMetricLabel"]{color:#8fa8c8!important;font-size:.75rem!important;}
.stButton>button{background:#181f2e!important;border:1px solid #2a3545!important;color:#e2e8f0!important;
  border-radius:6px!important;font-size:.82rem!important;transition:all .15s!important;}
.stButton>button:hover{border-color:#2f7cf6!important;color:#79b8ff!important;}
[data-testid="stTabs"] [data-baseweb="tab"]{color:#8892a4!important;font-size:.86rem!important;}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"]{color:#79b8ff!important;border-bottom:2px solid #2f7cf6!important;}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════
if "sel_zone" not in st.session_state:
    st.session_state["sel_zone"] = 5

# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def to_f(s: pd.Series) -> pd.Series:
    """Convert any Series to plain float64 — kills nullable pd.NA."""
    return pd.to_numeric(s, errors="coerce").astype("float64")


@st.cache_data(show_spinner=False, ttl=3600)
def load_years(years: tuple) -> pd.DataFrame:
    import pyarrow.parquet as pq
    dfs = []
    for yr in years:
        p = DATA_DIR / f"statcast_raw_{yr}.parquet"
        if not p.exists():
            continue
        avail = set(pq.read_schema(p).names)
        cols  = [c for c in NEED_COLS if c in avail]
        df    = pd.read_parquet(p, columns=cols)
        df["year"] = yr
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    nullable_kinds = {
        "Int8","Int16","Int32","Int64",
        "UInt8","UInt16","UInt32","UInt64",
        "Float32","Float64","boolean",
    }
    for c in out.columns:
        if out[c].dtype.name in nullable_kinds:
            out[c] = to_f(out[c])
    return out


def assign_zones(df: pd.DataFrame) -> pd.DataFrame:
    if "zone" in df.columns:
        df["zone"] = to_f(df["zone"])
        return df
    if "plate_x" not in df.columns or "plate_z" not in df.columns:
        df["zone"] = np.nan
        return df
    px  = to_f(df["plate_x"]).values
    pz  = to_f(df["plate_z"]).values
    out = np.full(len(df), np.nan)
    for z, (x0, z0, x1, z1) in ZONE_BOUNDS.items():
        mask = (px >= x0) & (px < x1) & (pz >= z0) & (pz < z1)
        out[mask] = float(z)
    out[np.isnan(px) | np.isnan(pz)] = np.nan
    df = df.copy()
    df["zone"] = out
    return df


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "spray_angle" in df.columns:
        df["spray_bin"] = pd.cut(
            to_f(df["spray_angle"]),
            bins=SPRAY_BINS, labels=SPRAY_LABELS, right=True,
        )
    if "launch_angle" in df.columns:
        df["la_bin"] = pd.cut(
            to_f(df["launch_angle"]),
            bins=LA_BINS, labels=LA_LABELS, right=True,
        )
    return df


def _empty_fig(title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text="No data for current filters",
        x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=14, color="#4a5568"),
    )
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color=TXT),
        title=dict(text=title, font=dict(size=12, color="#c9d1d9")),
        height=340,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig

# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:10px 0 8px;">'
        '<span style="font-size:2rem;">⚾</span><br>'
        '<span style="color:#79b8ff;font-weight:700;font-size:.92rem;">Zone → Spray & Launch</span><br>'
        '<span style="color:#718096;font-size:.7rem;">Statcast · pitch-level analysis</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if not AVAIL_YEARS:
        st.error("No statcast_raw_YYYY.parquet files found in DATA_DIR.")
        st.stop()

    sel_years = st.multiselect(
        "Season(s)", AVAIL_YEARS, default=[AVAIL_YEARS[-1]],
    )
    if not sel_years:
        st.warning("Select at least one year.")
        st.stop()

    st.markdown("---")
    st.markdown("**🏏 Handedness**")
    bh = st.radio("Batter",  ["All","RHB","LHB"], horizontal=True, key="bh")
    ph = st.radio("Pitcher", ["All","RHP","LHP"], horizontal=True, key="ph")

    st.markdown("---")
    st.markdown("**⚡ Pitch parameters**")
    velo_rng = st.slider("Velocity (mph)",  60, 105, (60, 105))
    spin_rng = st.slider("Spin rate (rpm)", 1000, 3600, (1000, 3600))

    st.markdown("---")
    zone_color_by = st.selectbox(
        "Zone colour metric",
        ["Batted balls", "Pull %", "Avg Exit Velo", "Avg Launch Angle"],
    )
    st.markdown("---")
    if st.button("🔄 Clear cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Source: MLB Statcast")

# ══════════════════════════════════════════════════════════════════
#  LOAD + PREPARE DATA
# ══════════════════════════════════════════════════════════════════
with st.spinner("⏳ Loading data…"):
    df_raw = load_years(tuple(sorted(sel_years)))

if df_raw.empty:
    st.error(
        "No parquet files found. "
        "Place statcast_raw_YYYY.parquet files in the same folder as this script."
    )
    st.stop()

df_raw  = assign_zones(df_raw)
df_raw  = add_bins(df_raw)

df_base = df_raw.copy()
if bh != "All" and "stand"    in df_base.columns:
    df_base = df_base[df_base["stand"].astype(str)    == ("R" if bh == "RHB" else "L")]
if ph != "All" and "p_throws" in df_base.columns:
    df_base = df_base[df_base["p_throws"].astype(str) == ("R" if ph == "RHP" else "L")]
if "release_speed"     in df_base.columns:
    df_base = df_base[to_f(df_base["release_speed"]).between(*velo_rng)]
if "release_spin_rate" in df_base.columns:
    df_base = df_base[to_f(df_base["release_spin_rate"]).between(*spin_rng)]

avail_pt = sorted(df_base["pitch_type"].dropna().unique().tolist()) \
           if "pitch_type" in df_base.columns else []

# ══════════════════════════════════════════════════════════════════
#  HEADER + PITCH FILTER
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="ttl">⚾ Pitch Zone → Spray & Launch Analysis</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub">Click a zone · select pitch type(s) · '
    'see spray angle and launch angle distributions for those batted balls</div>',
    unsafe_allow_html=True,
)

sel_pitches = st.multiselect(
    "Pitch type(s)",
    options=avail_pt,
    default=[],
    format_func=lambda x: f"{PITCH_NAMES.get(x, x)} ({x})",
    placeholder="All pitch types",
    key="pitch_sel",
)

df_work = (df_base[df_base["pitch_type"].isin(sel_pitches)].copy()
           if sel_pitches else df_base.copy())

# ══════════════════════════════════════════════════════════════════
#  ZONE GRID FIGURE
# ══════════════════════════════════════════════════════════════════

def zone_metric_val(sub: pd.DataFrame, metric: str) -> float:
    if sub.empty:
        return np.nan
    if metric == "Batted balls":
        n = sub["spray_angle"].notna().sum() if "spray_angle" in sub.columns else len(sub)
        return float(n)
    if metric == "Pull %" and "spray_angle" in sub.columns and "stand" in sub.columns:
        sa   = to_f(sub["spray_angle"])
        st_r = sub["stand"].astype(str) == "R"
        pull = ((sa < -15) & st_r) | ((sa > 15) & ~st_r)
        n    = sa.notna().sum()
        return float(pull.sum() / n * 100) if n > 0 else np.nan
    if metric == "Avg Exit Velo" and "launch_speed" in sub.columns:
        return float(to_f(sub["launch_speed"]).mean())
    if metric == "Avg Launch Angle" and "launch_angle" in sub.columns:
        return float(to_f(sub["launch_angle"]).mean())
    return float(len(sub))


def build_zone_fig(df: pd.DataFrame, sel_zone: int, metric: str) -> go.Figure:
    zone_vals = {}
    if "zone" in df.columns:
        for z in range(1, 10):
            zone_vals[z] = zone_metric_val(df[df["zone"] == z], metric)

    vals  = [v for v in zone_vals.values() if not (isinstance(v, float) and np.isnan(v))]
    vmin  = float(min(vals)) if vals else 0.0
    vmax  = float(max(vals)) if vals else 1.0
    vspan = max(vmax - vmin, 1e-9)

    def cell_color(z: int) -> str:
        v = zone_vals.get(z, np.nan)
        if isinstance(v, float) and np.isnan(v):
            return "rgba(28,34,48,1)"
        t = (v - vmin) / vspan
        r = int(30  + t * 215)
        g = int(40  + max(0.0, 1.0 - abs(t - 0.5) * 2.5) * 175)
        b = int(195 - t * 175)
        return f"rgb({r},{g},{b})"

    def fmt_val(z: int) -> str:
        v = zone_vals.get(z, np.nan)
        if isinstance(v, float) and np.isnan(v):
            return "—"
        if metric == "Batted balls":
            return f"{int(v):,}"
        if metric == "Pull %":
            return f"{v:.1f}%"
        return f"{v:.1f}"

    fig = go.Figure()

    # Zone rectangles + annotations + click targets
    for z in range(1, 10):
        x0, z0, x1, z1 = ZONE_BOUNDS[z]
        cx  = (x0 + x1) / 2
        cz  = (z0 + z1) / 2
        sel = (z == sel_zone)

        fig.add_shape(
            type="rect",
            x0=x0, y0=z0, x1=x1, y1=z1,
            fillcolor=cell_color(z),
            line=dict(color="#79b8ff" if sel else "#2a3545", width=3 if sel else 1),
        )
        # Zone number
        fig.add_annotation(
            x=cx, y=cz + 0.09, xref="x", yref="y",
            text=f"<b>Z{z}</b>",
            showarrow=False,
            font=dict(size=12, color="#f0f6ff" if sel else "#c9d1d9", family="DM Sans"),
        )
        # Metric value
        fig.add_annotation(
            x=cx, y=cz - 0.10, xref="x", yref="y",
            text=fmt_val(z),
            showarrow=False,
            font=dict(
                size=10,
                color="#57d9a3" if sel else "#8892a4",
                family="JetBrains Mono",
            ),
        )
        # Invisible scatter for click detection
        fig.add_trace(go.Scatter(
            x=[cx], y=[cz],
            mode="markers",
            marker=dict(size=40, opacity=0, color="#000000"),
            customdata=[[z]],
            name=f"Zone {z}",
            showlegend=False,
            hovertemplate=(
                f"<b>Zone {z}</b> — {ZONE_DESC.get(z,'')}<br>"
                f"{metric}: {fmt_val(z)}<br>"
                "<i>Click to select</i><extra></extra>"
            ),
        ))

    # Home plate
    fig.add_trace(go.Scatter(
        x=[-0.24, 0.0, 0.24, 0.24, -0.24, -0.24],
        y=[1.16,  0.96, 1.16, 1.30,  1.30,  1.16],
        fill="toself",
        fillcolor="rgba(200,210,220,0.40)",
        line=dict(color="#e2e8f0", width=1),
        mode="lines",
        showlegend=False,
        hoverinfo="skip",
    ))

    # Outer dashed border
    fig.add_shape(
        type="rect",
        x0=-0.71, y0=1.50, x1=0.71, y1=3.50,
        fillcolor="rgba(0,0,0,0)",
        line=dict(color="#4a5568", width=1, dash="dot"),
    )

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG2,
        font=dict(color=TXT, family="DM Sans"),
        height=440,
        margin=dict(l=30, r=20, t=48, b=60),
        title=dict(
            text="Strike Zone  (catcher's view · click to select zone)",
            font=dict(size=13, color="#c9d1d9"),
        ),
        xaxis=dict(
            range=[-0.96, 0.96],
            tickvals=[-0.71, -0.237, 0.237, 0.71],
            ticktext=["In", "⅓", "⅔", "Out"],
            gridcolor=GRID,
            zeroline=False,
            title=dict(
                text="← 3B side (Inside to RHB)     catcher's view     1B side (Outside to RHB) →",
                font=dict(size=9, color=SUB),
            ),
            tickfont=dict(color=SUB, size=9),
        ),
        yaxis=dict(
            range=[0.82, 3.76],
            tickvals=[1.5, 2.17, 2.83, 3.5],
            ticktext=["Low", "⅓", "⅔", "High"],
            gridcolor=GRID,
            zeroline=False,
            title=dict(text="Height (ft)", font=dict(size=9, color=SUB)),
            tickfont=dict(color=SUB, size=9),
        ),
        dragmode="select",
        showlegend=False,
    )
    return fig

# ══════════════════════════════════════════════════════════════════
#  DISTRIBUTION CHART BUILDERS
# ══════════════════════════════════════════════════════════════════

def spray_bar_chart(df: pd.DataFrame, zone: int, pitches: list) -> go.Figure:
    sub = df[df["zone"] == zone].copy() if "zone" in df.columns else df.copy()
    if "spray_bin" not in sub.columns or sub.empty:
        return _empty_fig("Spray Angle Distribution")

    counts = (
        sub["spray_bin"].astype(str)
        .replace("nan", pd.NA)
        .value_counts()
        .reindex(SPRAY_LABELS, fill_value=0)
    )
    total = int(counts.sum())
    pcts  = (counts / max(total, 1) * 100).round(1)

    title_parts = [f"Zone {zone}"]
    if pitches:
        title_parts.append(", ".join(pitches))
    title_parts.append("— Spray Angle Distribution")
    title = "  ·  ".join(title_parts[:2]) + "  " + title_parts[-1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=SPRAY_LABELS,
        y=counts.values,
        marker_color=SPRAY_COLORS,
        text=[f"{c:,}<br>({p:.1f}%)" for c, p in zip(counts.values, pcts.values)],
        textposition="outside",
        textfont=dict(size=9, color=TXT),
        hovertemplate="Bin: %{x}<br>Count: %{y:,}<br>%{text}<extra></extra>",
    ))
    fig.add_vrect(
        x0=-0.5, x1=3.5,
        fillcolor="rgba(239,68,68,0.06)", line_width=0,
        annotation_text="← Pull / LF", annotation_position="top left",
        annotation_font=dict(size=8, color="#ef4444"),
    )
    fig.add_vrect(
        x0=6.5, x1=10.5,
        fillcolor="rgba(99,102,241,0.06)", line_width=0,
        annotation_text="Oppo / RF →", annotation_position="top right",
        annotation_font=dict(size=8, color="#6366f1"),
    )
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG2,
        font=dict(color=TXT, family="DM Sans"),
        title=dict(text=title, font=dict(size=12, color="#c9d1d9")),
        xaxis=dict(
            title="Spray Angle Bin  (− = Left Field · + = Right Field)",
            tickangle=-38,
            gridcolor=GRID,
            tickfont=dict(color=SUB, size=8.5),
            zeroline=False,
        ),
        yaxis=dict(
            title="Batted Balls",
            gridcolor=GRID,
            zeroline=False,
            tickfont=dict(color=SUB),
        ),
        height=390,
        margin=dict(l=55, r=20, t=58, b=100),
        showlegend=False,
    )
    return fig


def launch_angle_bar_chart(df: pd.DataFrame, zone: int, pitches: list) -> go.Figure:
    sub = df[df["zone"] == zone].copy() if "zone" in df.columns else df.copy()
    if "la_bin" not in sub.columns or sub.empty:
        return _empty_fig("Launch Angle Distribution")

    counts = (
        sub["la_bin"].astype(str)
        .replace("nan", pd.NA)
        .value_counts()
        .reindex(LA_LABELS, fill_value=0)
    )
    total = int(counts.sum())
    pcts  = (counts / max(total, 1) * 100).round(1)

    bar_colors = [
        "#f87171" if "35-45" in lbl else c
        for lbl, c in zip(LA_LABELS, LA_COLORS)
    ]
    line_c = ["#ffffff" if "35-45" in lbl else GRID for lbl in LA_LABELS]
    line_w = [2        if "35-45" in lbl else 1    for lbl in LA_LABELS]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=LA_LABELS,
        y=counts.values,
        marker=dict(color=bar_colors, line=dict(color=line_c, width=line_w)),
        text=[f"{c:,}<br>({p:.1f}%)" for c, p in zip(counts.values, pcts.values)],
        textposition="outside",
        textfont=dict(size=9, color=TXT),
        hovertemplate="Bin: %{x}<br>Count: %{y:,}<br>%{text}<extra></extra>",
    ))

    peak = int(counts.max()) if total > 0 else 1
    if "35-45° ★" in LA_LABELS:
        idx = LA_LABELS.index("35-45° ★")
        fig.add_annotation(
            x=LA_LABELS[idx],
            y=int(counts.iloc[idx]) + peak * 0.09,
            text="HR zone",
            showarrow=False,
            font=dict(size=9, color="#f87171"),
        )

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG2,
        font=dict(color=TXT, family="DM Sans"),
        title=dict(
            text=f"Zone {zone} — Launch Angle Distribution",
            font=dict(size=12, color="#c9d1d9"),
        ),
        xaxis=dict(
            title="Launch Angle Bin",
            tickangle=-20,
            gridcolor=GRID,
            tickfont=dict(color=SUB, size=9),
            zeroline=False,
        ),
        yaxis=dict(
            title="Batted Balls",
            gridcolor=GRID,
            zeroline=False,
            tickfont=dict(color=SUB),
        ),
        height=390,
        margin=dict(l=55, r=20, t=58, b=90),
        showlegend=False,
    )
    return fig


def joint_heatmap_chart(df: pd.DataFrame, zone: int) -> go.Figure:
    sub = df[df["zone"] == zone].copy() if "zone" in df.columns else df.copy()
    need = ["spray_bin", "la_bin"]
    if any(c not in sub.columns for c in need) or sub.empty:
        return _empty_fig("Spray × Launch Angle Heatmap")

    sub2 = sub.dropna(subset=need).copy()
    sub2["spray_bin"] = sub2["spray_bin"].astype(str)
    sub2["la_bin"]    = sub2["la_bin"].astype(str)

    piv = (
        sub2.groupby(["la_bin", "spray_bin"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=LA_LABELS, columns=SPRAY_LABELS, fill_value=0)
    )
    total   = int(piv.values.sum())
    piv_pct = (piv / max(total, 1) * 100).round(1)

    cell_text = [
        [f"{int(piv.iloc[r, c])}<br>{piv_pct.iloc[r, c]:.1f}%"
         for c in range(piv.shape[1])]
        for r in range(piv.shape[0])
    ]

    fig = go.Figure(go.Heatmap(
        z=piv.values.tolist(),
        x=SPRAY_LABELS,
        y=LA_LABELS,
        text=cell_text,
        texttemplate="%{text}",
        textfont=dict(size=8),
        colorscale="YlOrRd",
        colorbar=dict(title="Count", tickfont=dict(color=TXT, size=9)),
        hovertemplate="Spray: %{x}<br>LA: %{y}<br>Count: %{z}<extra></extra>",
    ))

    if "35-45° ★" in LA_LABELS:
        la_idx = LA_LABELS.index("35-45° ★")
        fig.add_shape(
            type="rect",
            x0=-0.5, y0=la_idx - 0.5,
            x1=float(len(SPRAY_LABELS)) - 0.5, y1=la_idx + 0.5,
            fillcolor="rgba(248,113,113,0.10)",
            line=dict(color="#f87171", width=2, dash="dot"),
        )

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG2,
        font=dict(color=TXT, family="DM Sans"),
        title=dict(
            text=f"Zone {zone} — Spray Angle × Launch Angle  (count · %)",
            font=dict(size=12, color="#c9d1d9"),
        ),
        xaxis=dict(
            title="Spray Angle  (− = Left Field · + = Right Field)",
            tickangle=-40,
            gridcolor=GRID,
            tickfont=dict(color=SUB, size=8),
            zeroline=False,
        ),
        yaxis=dict(
            title="Launch Angle Bin",
            gridcolor=GRID,
            tickfont=dict(color=SUB, size=9),
            zeroline=False,
        ),
        height=400,
        margin=dict(l=110, r=20, t=58, b=100),
    )
    return fig

# ══════════════════════════════════════════════════════════════════
#  PIVOT TABLE BUILDERS
# ══════════════════════════════════════════════════════════════════

def pivot_zone_vs_bins(df: pd.DataFrame, bin_col: str, labels: list) -> pd.DataFrame:
    if "zone" not in df.columns or bin_col not in df.columns:
        return pd.DataFrame()
    sub = df[to_f(df["zone"]).between(1, 9)].copy()
    sub["_z"]  = to_f(sub["zone"]).astype(int)
    sub["_bc"] = sub[bin_col].astype(str).replace("nan", pd.NA)
    piv = (
        sub.groupby(["_z", "_bc"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=labels, fill_value=0)
    )
    piv.index = [f"Z{z} — {ZONE_DESC.get(z,'')}" for z in piv.index]
    piv.insert(0, "TOTAL", piv.sum(axis=1))
    return piv


def pivot_pitch_vs_bins(df: pd.DataFrame, bin_col: str, labels: list) -> pd.DataFrame:
    if "pitch_type" not in df.columns or bin_col not in df.columns:
        return pd.DataFrame()
    sub = df.copy()
    sub["_bc"] = sub[bin_col].astype(str).replace("nan", pd.NA)
    piv = (
        sub.groupby(["pitch_type", "_bc"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=labels, fill_value=0)
    )
    piv.index = [f"{PITCH_NAMES.get(p, p)} ({p})" for p in piv.index]
    piv.insert(0, "TOTAL", piv.sum(axis=1))
    return piv


def style_pivot(df: pd.DataFrame, highlight: str = None):
    if df.empty:
        return df
    num = [c for c in df.columns if c != "TOTAL"]
    styled = (
        df.style
        .background_gradient(cmap="YlOrRd", subset=num, axis=None)
        .format("{:,.0f}")
        .set_table_styles([
            {"selector": "th",
             "props": [("background","#111621"),("color","#a0aec0"),
                       ("font-size","0.74rem"),("white-space","nowrap")]},
            {"selector": "td",
             "props": [("font-size","0.78rem"),("color","#e2e8f0"),
                       ("white-space","nowrap")]},
        ])
    )
    if highlight and highlight in df.columns:
        styled = styled.apply(
            lambda s: [
                "background:#f8717144;color:#f87171;font-weight:700"
                if (not pd.isna(v) and v == s.max()) else ""
                for v in s
            ],
            subset=[highlight],
        )
    return styled

# ══════════════════════════════════════════════════════════════════
#  PULL-RATE BAR
# ══════════════════════════════════════════════════════════════════

def pull_rate_bar(df: pd.DataFrame) -> go.Figure:
    if "zone" not in df.columns or "spray_angle" not in df.columns:
        return _empty_fig("Pull % by Zone")

    rows = []
    for z in range(1, 10):
        sub = df[df["zone"] == z]
        if sub.empty:
            continue
        sa = to_f(sub["spray_angle"])
        if "stand" in sub.columns:
            st_r = sub["stand"].astype(str) == "R"
            pull = ((sa < -15) & st_r) | ((sa > 15) & ~st_r)
        else:
            pull = sa < -15
        n = int(sa.notna().sum())
        rows.append({
            "Zone": f"Z{z}",
            "Pull %": float(pull.sum()) / max(n, 1) * 100,
            "N": n,
            "Desc": ZONE_DESC.get(z, ""),
        })

    if not rows:
        return _empty_fig("Pull % by Zone")

    df_p   = pd.DataFrame(rows)
    inside = {"Z1","Z4","Z7"}
    outside= {"Z3","Z6","Z9"}
    colors = [
        "#ef4444" if r["Zone"] in inside  else
        "#3b82f6" if r["Zone"] in outside else
        "#22c55e"
        for _, r in df_p.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_p["Zone"],
        y=df_p["Pull %"],
        marker_color=colors,
        text=[f"{v:.1f}%" for v in df_p["Pull %"]],
        textposition="outside",
        textfont=dict(size=9, color=TXT),
        customdata=df_p[["N","Desc"]].values,
        hovertemplate=(
            "<b>%{x}</b>  %{customdata[1]}<br>"
            "Pull%%: %{y:.1f}%%  (n=%{customdata[0]})<extra></extra>"
        ),
    ))
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG2,
        font=dict(color=TXT, family="DM Sans"),
        title=dict(
            text="Pull % by Zone  (red = inside RHB · green = center · blue = outside RHB)",
            font=dict(size=12, color="#c9d1d9"),
        ),
        xaxis=dict(
            title="Zone",
            gridcolor=GRID,
            zeroline=False,
            tickfont=dict(color=SUB),
        ),
        yaxis=dict(
            title="Pull %",
            gridcolor=GRID,
            zeroline=False,
            tickfont=dict(color=SUB),
        ),
        height=320,
        margin=dict(l=55, r=20, t=55, b=50),
        showlegend=False,
    )
    return fig

# ══════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════
col_zone, col_right = st.columns([1, 2], gap="large")

# ── LEFT: Zone grid ──────────────────────────────────────────────
with col_zone:
    st.markdown('<div class="sec">🎯 Click a Zone</div>', unsafe_allow_html=True)

    fig_zone = build_zone_fig(df_work, st.session_state["sel_zone"], zone_color_by)
    event = st.plotly_chart(
        fig_zone,
        use_container_width=True,
        on_select="rerun",
        key="zone_grid",
    )

    # Handle Plotly click
    if (event
            and hasattr(event, "selection")
            and event.selection
            and event.selection.get("points")):
        pts = event.selection["points"]
        if pts:
            cd = pts[0].get("customdata")
            if cd and len(cd) > 0:
                clicked = int(cd[0])
                if 1 <= clicked <= 9:
                    st.session_state["sel_zone"] = clicked

    sel_zone = st.session_state["sel_zone"]

    # Fallback: 3×3 button grid
    st.markdown("**Or select directly:**")
    for zone_row in [[1,2,3],[4,5,6],[7,8,9]]:
        row_cols = st.columns(3)
        for col_el, z in zip(row_cols, zone_row):
            with col_el:
                label = f"Z{z}" + (" ✓" if z == sel_zone else "")
                if st.button(label, key=f"btn_z{z}", use_container_width=True):
                    st.session_state["sel_zone"] = z
                    st.rerun()

    # Zone summary
    sub_sel = (df_work[df_work["zone"] == sel_zone].copy()
               if "zone" in df_work.columns else pd.DataFrame())
    n_bip = int(sub_sel["spray_angle"].notna().sum()) \
            if "spray_angle" in sub_sel.columns else 0
    pitch_str = ", ".join(sel_pitches) if sel_pitches else "All pitches"

    st.markdown("---")
    st.markdown(
        f'<div class="icard">'
        f'<b>Zone {sel_zone}</b> — {ZONE_DESC.get(sel_zone,"")}<br>'
        f'Pitch: <b>{pitch_str}</b><br>'
        f'Total rows: <b>{len(sub_sel):,}</b> · BIP: <b>{n_bip:,}</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if n_bip > 0:
        sa_s = to_f(sub_sel["spray_angle"]) if "spray_angle" in sub_sel.columns \
               else pd.Series(dtype=float)
        la_s = to_f(sub_sel["launch_angle"]) if "launch_angle" in sub_sel.columns \
               else pd.Series(dtype=float)

        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("BIP", f"{n_bip:,}")
            hr_cnt = int(((la_s >= 35) & (la_s <= 45)).sum()) \
                     if len(la_s.dropna()) > 0 else 0
            st.metric("35–45° LA (★)", f"{hr_cnt:,}")
        with mc2:
            if len(sa_s.dropna()) > 0 and "stand" in sub_sel.columns:
                st_r = sub_sel["stand"].astype(str) == "R"
                pull = ((sa_s < -15) & st_r) | ((sa_s > 15) & ~st_r)
                st.metric("Pull %", f"{pull.mean()*100:.1f}%")
            if len(la_s.dropna()) > 0:
                st.metric("Avg LA", f"{la_s.mean():.1f}°")

# ── RIGHT: Distributions ─────────────────────────────────────────
with col_right:
    st.markdown('<div class="sec">📊 Distributions for selected zone + pitch filter</div>',
                unsafe_allow_html=True)

    tab_sp, tab_la, tab_jt = st.tabs(
        ["↔ Spray Angle", "↕ Launch Angle", "🔲 Joint View"]
    )
    with tab_sp:
        st.plotly_chart(
            spray_bar_chart(df_work, sel_zone, sel_pitches),
            use_container_width=True,
        )
        st.caption(
            "Negative bins = Left Field · Positive bins = Right Field · "
            "Pull for RHB ≈ bins −45..−15°"
        )
    with tab_la:
        st.plotly_chart(
            launch_angle_bar_chart(df_work, sel_zone, sel_pitches),
            use_container_width=True,
        )
        st.caption(
            "★ 35–45° = prime HR zone · < 0° = ground ball · > 45° = pop-up"
        )
    with tab_jt:
        st.plotly_chart(
            joint_heatmap_chart(df_work, sel_zone),
            use_container_width=True,
        )
        st.caption(
            "Each cell = count + % of BIP with that exact spray + launch-angle combo. "
            "Dashed red box = 35–45° LA row."
        )

# ══════════════════════════════════════════════════════════════════
#  PIVOT TABLES
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="sec">📋 Contingency Tables — All Zones</div>',
            unsafe_allow_html=True)

ptab_la, ptab_sp, ptab_pt = st.tabs([
    "Zone × Launch Angle", "Zone × Spray Angle", "Pitch Type × Launch Angle",
])

with ptab_la:
    pla = pivot_zone_vs_bins(df_work, "la_bin", LA_LABELS)
    if not pla.empty:
        hi_col = "35-45° ★" if "35-45° ★" in pla.columns else None
        st.dataframe(style_pivot(pla, hi_col), use_container_width=True, height=370)
        st.caption(
            "Rows = zones · Columns = launch angle bins · Values = batted ball count · "
            "Red = zone with the most balls in each column."
        )
        st.download_button(
            "📥 Export Zone × Launch Angle",
            data=pla.to_csv().encode(),
            file_name="zone_launch_angle.csv",
            mime="text/csv",
        )
    else:
        st.info("No data available.")

with ptab_sp:
    psp = pivot_zone_vs_bins(df_work, "spray_bin", SPRAY_LABELS)
    if not psp.empty:
        st.dataframe(style_pivot(psp), use_container_width=True, height=370)
        st.caption(
            "Rows = zones · Columns = spray angle bins · "
            "Negative bins = Left Field · Positive bins = Right Field."
        )
        st.download_button(
            "📥 Export Zone × Spray Angle",
            data=psp.to_csv().encode(),
            file_name="zone_spray_angle.csv",
            mime="text/csv",
        )
    else:
        st.info("No data available.")

with ptab_pt:
    ppt = pivot_pitch_vs_bins(df_work, "la_bin", LA_LABELS)
    if not ppt.empty:
        hi_col = "35-45° ★" if "35-45° ★" in ppt.columns else None
        st.dataframe(style_pivot(ppt, hi_col), use_container_width=True, height=400)
        st.caption(
            "Rows = pitch types · Columns = launch angle bins. "
            "Use zone selector + pitch filter above to narrow down."
        )
        st.download_button(
            "📥 Export Pitch Type × Launch Angle",
            data=ppt.to_csv().encode(),
            file_name="pitch_launch_angle.csv",
            mime="text/csv",
        )
    else:
        st.info("No data available.")

# ══════════════════════════════════════════════════════════════════
#  PULL-RATE STRIP
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="sec">📉 Inside → Pull Tendency  (all zones comparison)</div>',
            unsafe_allow_html=True)

pr_l, pr_r = st.columns([1.6, 1.4])

with pr_l:
    st.plotly_chart(pull_rate_bar(df_work), use_container_width=True)

with pr_r:
    if "zone" in df_work.columns:
        summary_rows = []
        for z in range(1, 10):
            sub = df_work[df_work["zone"] == z]
            la  = to_f(sub["launch_angle"]) if "launch_angle" in sub.columns \
                  else pd.Series(dtype=float)
            sa  = to_f(sub["spray_angle"])  if "spray_angle"  in sub.columns \
                  else pd.Series(dtype=float)
            ev  = to_f(sub["launch_speed"]) if "launch_speed"  in sub.columns \
                  else pd.Series(dtype=float)

            n_bip_z = int(sa.notna().sum())
            if "stand" in sub.columns:
                st_r = sub["stand"].astype(str) == "R"
                pull = ((sa < -15) & st_r) | ((sa > 15) & ~st_r)
                pull_p = float(pull.sum()) / max(n_bip_z, 1) * 100
            else:
                pull_p = np.nan

            hr_cnt_z = int(((la >= 35) & (la <= 45)).sum()) \
                       if len(la.dropna()) > 0 else 0
            avg_la   = round(float(la.mean()), 1) if len(la.dropna()) > 0 else np.nan
            avg_ev   = round(float(ev.mean()), 1) if len(ev.dropna()) > 0 else np.nan

            summary_rows.append({
                "Zone": f"Z{z}",
                "Description": ZONE_DESC.get(z, ""),
                "BIP": n_bip_z,
                "Pull %": round(pull_p, 1) if not np.isnan(pull_p) else np.nan,
                "35-45° LA": hr_cnt_z,
                "Avg LA°": avg_la,
                "Avg EV": avg_ev,
            })

        df_sum = pd.DataFrame(summary_rows).set_index("Zone")

        styled_sum = (
            df_sum.style
            .background_gradient(cmap="RdYlGn", subset=["Pull %"], axis=0)
            .background_gradient(cmap="YlOrRd",  subset=["35-45° LA"], axis=0)
            .format({
                "Pull %":  "{:.1f}",
                "Avg LA°": "{:.1f}",
                "Avg EV":  "{:.1f}",
                "BIP":     "{:,.0f}",
                "35-45° LA": "{:,.0f}",
            })
            .set_table_styles([
                {"selector": "th",
                 "props": [("background","#111621"),("color","#a0aec0"),
                           ("font-size","0.73rem"),("white-space","nowrap")]},
                {"selector": "td",
                 "props": [("font-size","0.77rem"),("color","#e2e8f0"),
                           ("white-space","nowrap")]},
            ])
        )
        st.dataframe(styled_sum, use_container_width=True, height=350)
        st.caption(
            "Pull % = share of BIP pulled. "
            "35-45° LA = HR-zone ball count. "
            "Avg EV = average exit velocity."
        )

st.markdown("---")
st.caption(
    "Source: MLB Statcast  ·  "
    "Zones 1-9 = standard Statcast 3×3 grid, catcher's view  ·  "
    "Negative spray angle = Left Field"
)