import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# Fix Pandas Styler cell limit before anything else
pd.set_option("styler.render.max_elements", 10_000_000)

# ════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MLB Statcast Pro Dashboard",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════
# CUSTOM CSS
# ════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Base ─────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0b0f17 !important;
    color: #e2e8f0;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #111621 !important;
    border-right: 1px solid #1e2535;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Global text override – everything visible ─────── */
p, div, label, span, li, td, th, h1, h2, h3, h4, h5, h6,
.stMarkdown, .stText, [data-testid="stMarkdownContainer"] * {
    color: #e2e8f0;
}

/* ── Header ───────────────────────────────────────── */
.dash-header {
    padding: 18px 0 4px 0;
    border-bottom: 1px solid #1e2535;
    margin-bottom: 20px;
}
.dash-title {
    font-size: 1.75rem; font-weight: 700;
    color: #f0f6ff !important;
    letter-spacing: -0.4px; line-height: 1.1;
}
.dash-sub { font-size: 0.82rem; color: #7a8494 !important; margin-top: 4px; }

/* ── Section header ───────────────────────────────── */
.sec-hdr {
    display: flex; align-items: center; gap: 10px;
    background: linear-gradient(90deg, #1a3a6622, transparent);
    border-left: 3px solid #2f7cf6;
    padding: 10px 18px; border-radius: 0 8px 8px 0;
    margin: 24px 0 16px 0;
    color: #79b8ff !important;
    font-weight: 600; font-size: 1rem;
}

/* ── Ref cards ────────────────────────────────────── */
.ref-card  { background: #111621; border: 1px solid #1e2535; border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; }
.ref-title { color: #79b8ff; font-weight: 600; font-size: 0.83rem; margin-bottom: 8px; }
.ref-body  { color: #a0aec0; font-size: 0.78rem; line-height: 1.55; }
.ref-badge { display: inline-block; background: #181f2e; border: 1px solid #2a3545; border-radius: 5px; padding: 3px 9px; margin: 2px 2px 4px 0; font-size: 0.73rem; color: #63b3ff; font-family: 'JetBrains Mono', monospace; }
.ref-badge-dim { color: #718096 !important; }

/* ── Sidebar logo ─────────────────────────────────── */
.sb-logo      { text-align: center; padding: 12px 0 8px 0; border-bottom: 1px solid #1e2535; margin-bottom: 14px; }
.sb-logo-icon { font-size: 2.2rem; display: block; margin-bottom: 4px; }
.sb-logo-name { color: #79b8ff !important; font-weight: 700; font-size: 0.95rem; letter-spacing: 0.5px; }
.sb-logo-sub  { color: #718096 !important; font-size: 0.72rem; margin-top: 3px; }

/* ── Divider ──────────────────────────────────────── */
.dash-divider { height: 1px; background: #1e2535; margin: 28px 0; border: none; }

/* ── All widget labels → white ────────────────────── */
.stSelectbox label,
.stSlider label,
.stCheckbox label,
.stRadio label,
div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] span {
    color: #e2e8f0 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

/* ── Selectbox ────────────────────────────────────── */
.stSelectbox > div > div {
    background: #111621 !important;
    border-color: #2a3545 !important;
    color: #e2e8f0 !important;
    font-size: 0.86rem !important;
}
/* Dropdown option text */
[data-baseweb="select"] [role="option"] span { color: #e2e8f0 !important; }

/* ── Radio buttons ────────────────────────────────── */
/* wrapper row */
div[data-testid="stRadio"] > div {
    gap: 8px !important;
    flex-wrap: wrap;
}
/* every radio label = pill */
div[data-testid="stRadio"] label {
    display: inline-flex !important;
    align-items: center !important;
    background: #181f2e !important;
    border: 1px solid #2a3545 !important;
    border-radius: 20px !important;
    padding: 5px 16px !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    margin: 0 !important;
}
/* text inside pill */
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span {
    color: #e2e8f0 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    line-height: 1 !important;
}
/* hide native radio circle */
div[data-testid="stRadio"] input[type="radio"] {
    width: 0 !important; height: 0 !important;
    opacity: 0 !important; position: absolute !important;
}
/* active / selected pill */
div[data-testid="stRadio"] label:has(input:checked) {
    background: #1a3a66 !important;
    border-color: #2f7cf6 !important;
}
div[data-testid="stRadio"] label:has(input:checked) p,
div[data-testid="stRadio"] label:has(input:checked) span {
    color: #79b8ff !important;
    font-weight: 600 !important;
}

/* ── Sliders ──────────────────────────────────────── */
.stSlider [data-baseweb="thumb"]      { background: #2f7cf6 !important; border-color: #2f7cf6 !important; }
.stSlider [data-baseweb="track-fill"] { background: #2f7cf6 !important; }
/* slider current value label */
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"],
div[data-testid="stSlider"] span { color: #a0aec0 !important; }

/* ── Expander ─────────────────────────────────────── */
details > summary {
    background: #111621 !important;
    border: 1px solid #1e2535 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    padding: 10px 16px !important;
}
details[open] > summary { border-radius: 8px 8px 0 0 !important; }
details > div {
    background: #0b0f17 !important;
    border: 1px solid #1e2535 !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    padding: 14px !important;
}
/* Streamlit expander header text */
.streamlit-expanderHeader p,
.streamlit-expanderHeader span {
    color: #e2e8f0 !important;
    font-size: 0.84rem !important;
}

/* ── Dataframe ────────────────────────────────────── */
[data-testid="stDataFrame"] { border: 1px solid #1e2535 !important; border-radius: 8px !important; }
.dataframe th { background: #111621 !important; color: #a0aec0 !important; font-size: 0.77rem !important; font-weight: 600 !important; }
.dataframe td { font-size: 0.82rem !important; color: #e2e8f0 !important; }

/* ── Buttons ──────────────────────────────────────── */
.stButton > button {
    background: #181f2e; border: 1px solid #2a3545;
    color: #e2e8f0 !important; border-radius: 7px;
    font-size: 0.83rem; font-weight: 500; transition: all 0.15s;
}
.stButton > button:hover { background: #1e2535; border-color: #2f7cf6; color: #79b8ff !important; }

/* ── Alerts ───────────────────────────────────────── */
.stWarning, .stInfo, .stError, .stSuccess {
    border-radius: 8px !important; font-size: 0.83rem !important;
}
.stWarning p, .stInfo p, .stError p, .stSuccess p { color: inherit !important; }

/* ── Caption / small text ─────────────────────────── */
.stCaption, small, [data-testid="stCaptionContainer"] p {
    color: #718096 !important; font-size: 0.76rem !important;
}

/* ── Mobile ───────────────────────────────────────── */
@media (max-width: 768px) {
    [data-testid="column"] { min-width: 100% !important; flex: 100% !important; }
    .dash-title { font-size: 1.3rem !important; }
    .sec-hdr    { font-size: 0.88rem; padding: 8px 14px; }
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# PITCH GROUPS & NAMES
# ════════════════════════════════════════════════════════
PITCH_GROUPS = {
    "Fastballs": ["FF", "SI", "FC"],
    "Offspeed":  ["CH", "FS", "SV"],
    "Breaking":  ["SL", "ST", "CU", "KC"],
}

PITCH_NAMES = {
    "FF": "Four-Seam Fastball (FF)",
    "SI": "Sinker (SI)",
    "FC": "Cutter (FC)",
    "SL": "Slider (SL)",
    "ST": "Sweeper (ST)",
    "CH": "Changeup (CH)",
    "CU": "Curveball (CU)",
    "KC": "Knuckle Curve (KC)",
    "SV": "Slurve (SV)",
    "FS": "Splitter (FS)",
}
CODE_FROM_DISP = {v: k for k, v in PITCH_NAMES.items()}

# ════════════════════════════════════════════════════════
# STATISTICS
# ════════════════════════════════════════════════════════
FIXED_RANGES = {
    "Whiff %":       (0,   70),
    "Swing %":       (10,  90),
    "Contact %":     (30, 100),
    "xwOBA":         (0.10, 0.55),
    "Exit Velo":     (70,  98),
    "Launch Angle":  (-15, 45),
    "Barrel %":      (0,   25),
    "Hard Hit %":    (0,   60),
    "Sweet Spot %":  (0,   50),
    "GB %":          (0,   80),
    "LD %":          (0,   80),
    "FB %":          (0,   80),
}
STAT_COL = {
    "Whiff %":       "whiff_pct",
    "Swing %":       "swing_pct",
    "Contact %":     "contact_pct",
    "xwOBA":         "avg_xwoba",
    "Exit Velo":     "avg_ev",
    "Launch Angle":  "avg_la",
    "Barrel %":      "barrel_pct",
    "Hard Hit %":    "hard_hit_pct",
    "Sweet Spot %":  "sweet_spot_pct",
    "GB %":          "gb_pct",
    "LD %":          "ld_pct",
    "FB %":          "fb_pct",
}
STAT_LABELS = list(FIXED_RANGES.keys())

ALL_COUNTS = [
    "0-0","0-1","0-2",
    "1-0","1-1","1-2",
    "2-0","2-1","2-2",
    "3-0","3-1","3-2",
]

DATA_FILE       = "statcast_all_pitches_2021-2025.parquet"
MAX_STYLED_ROWS = 5_000   # gradient styling only up to this many rows

# ════════════════════════════════════════════════════════
# MOVEMENT REFERENCE
# ════════════════════════════════════════════════════════
MOVEMENT_REF = {
    "FF": {
        "h_range": '+3" to +10"',  "h_dir": "arm-side",
        "v_range": '+10" to +17"', "v_dir": "backspin rise",
        "note_en": "High backspin fights gravity — appears to 'rise' vs. expected trajectory. Elite four-seamers exceed 14\" induced vertical break. Most effective high in zone (Zones 1–3). Velocity matters: 96+ mph creates a non-linear jump in whiff rate.",
        "note_pl": "Wysoki backspin przeciw grawitacji – efekt 'wznoszenia'. Elitarne FF przekraczają 14\" IVB. Najskuteczniejszy wysoko w strefie (1–3). 96+ mph = nieliniowy skok whiff%.",
    },
    "SI": {
        "h_range": '+8" to +17"',  "h_dir": "strong arm-side tail",
        "v_range": '−2" to +6"',   "v_dir": "gravity-aided sink",
        "note_en": "Pronounced arm-side run + gravity sink generates ground balls. GB rate 58–68% in zones 7–9. Slower sinkers (88–91 mph) produce more downward angle than 93+. Pairs devastatingly with a cutter on the glove side.",
        "note_pl": "Silny ruch arm-side + opadanie. GB% 58–68% w strefach 7–9. Wolniejszy (88–91 mph) ma większy kąt opadania. Idealny w parze z cutterem.",
    },
    "FC": {
        "h_range": '−9" to −2"',   "h_dir": "glove-side cut",
        "v_range": '+4" to +11"',  "v_dir": "slight rise",
        "note_en": "Late glove-side cut 'saws off' the handle of same-handed batters. Optimal velocity 88–91 mph. Best deployed away vs. same-handed batter (zones 3/6/9). Devastating paired with a sinker running the opposite direction.",
        "note_pl": "Późny ruch glove-side odcina trzonek kija. Optymalna prędkość 88–91 mph. Najlepszy zewnątrz vs. ta sama ręka (3/6/9). Zabójczy w parze z sinkerem.",
    },
    "SL": {
        "h_range": '−13" to −4"',  "h_dir": "glove-side break",
        "v_range": '−5" to +4"',   "v_dir": "flat / mild drop",
        "note_en": "Lateral sweep with moderate depth — highest overall MLB whiff rate. Most effective vs. same-handed batters in shadow zones 13–14. Spin 2200–2400 rpm at 84–88 mph generates peak whiff. Enormous platoon split — avoid as out-pitch vs. opposite hand.",
        "note_pl": "Boczny ruch z umiarkowanym opadem – najwyższy whiff% w MLB. Najskuteczniejszy vs. ta sama ręka w shadow 13–14. Gigantyczny split platoonowy.",
    },
    "ST": {
        "h_range": '−19" to −8"',  "h_dir": "extreme glove-side sweep",
        "v_range": '−1" to +7"',   "v_dir": "slight rise",
        "note_en": "Revolutionary horizontal movement pitch (~2022). Largest platoon split of any pitch — devastating vs. opposite-handed batters. Whiff 40–48% in away shadow zone vs. opposite hand. Vary velocity and entry angle to maintain effectiveness.",
        "note_pl": "Rewolucyjny poziomy zamach (od 2022). Największy split platoonowy ze wszystkich. Whiff 40–48% w shadow zewnętrznym vs. przeciwna ręka.",
    },
    "CH": {
        "h_range": '+6" to +15"',  "h_dir": "arm-side fade",
        "v_range": '−3" to +6"',   "v_dir": "fade / mild sink",
        "note_en": "Mimics fastball arm action — velocity differential (8–12 mph gap) is the weapon. Most effective vs. opposite-handed batter, low zone (zones 7–9). 1800–2000 rpm at 78–83 mph forces early commitment and generates ground balls.",
        "note_pl": "Naśladuje ruch fastballa – kluczowa różnica 8–12 mph. Najskuteczniejszy vs. przeciwna ręka, dolna strefa (7–9). 1800–2000 rpm przy 78–83 mph = early commitment i groundery.",
    },
    "CU": {
        "h_range": '−6" to +5"',   "h_dir": "neutral",
        "v_range": '−15" to −4"',  "v_dir": "sharp 12-to-6 drop",
        "note_en": "Pure topspin vertical dive — sharpest drop of all breaking balls. Shadow zones 13/14 are the primary target. Zone 5 (middle) leads to 12–15% barrel rate — avoid. Spin 2600–2800 rpm creates a sharper spike and more missed swings.",
        "note_pl": "Czysty topspin pionowy – najostrzejszy opad. Shadow 13/14 to główny cel. Strefa 5 = 12–15% barrel% – unikaj. 2600–2800 rpm = ostrzejszy spike i więcej whiff.",
    },
    "KC": {
        "h_range": '−5" to +4"',   "h_dir": "neutral",
        "v_range": '−17" to −6"',  "v_dir": "extreme tumbling drop",
        "note_en": "Deepest downward tumble of any breaking ball. Elite out-pitch in 2-strike counts — whiff 35–40% in low shadow with tunneling. Optimal spin 2600–2800 rpm. Reserve for finishing counts, limited value as a setup pitch.",
        "note_pl": "Najgłębszy tumble ze wszystkich łamanych. Elitarny out-pitch przy 2 strike'ach – whiff 35–40% w shadow z tunelowaniem. Zarezerwowany do kończenia.",
    },
    "SV": {
        "h_range": '−9" to 0"',    "h_dir": "glove-side",
        "v_range": '−11" to −2"',  "v_dir": "drop + lateral blend",
        "note_en": "Slider-curveball hybrid — moderate drop blended with lateral break. More versatile across platoon matchups. Useful for pitchers without an elite slider or curveball. Effective in low shadow zones from both handedness matchups.",
        "note_pl": "Hybryda slider-curve – umiarkowany opad + ruch boczny. Wszechstronny platoonowo. Skuteczny w dolnych shadow zones z obu stron.",
    },
    "FS": {
        "h_range": '+2" to +10"',  "h_dir": "arm-side",
        "v_range": '−13" to −2"',  "v_dir": "heavy tumbling sink",
        "note_en": "Low spin creates a heavy tumbling drop — looks like a fastball then falls off a table. Optimal: below 1800 rpm, 84–88 mph, low zone and shadow (13/14). GB rate above 65% in zones 7–9. Avoid high in zone (1–3): barrel rate climbs to 14–18%.",
        "note_pl": "Niski spin = opadający tumble. Optimum: poniżej 1800 rpm, 84–88 mph, dolna strefa i shadow (13/14). GB% powyżej 65% w strefach 7–9. Unikać góry (1–3).",
    },
}

# ════════════════════════════════════════════════════════
# TRANSLATIONS
# ════════════════════════════════════════════════════════
T_ALL = {
    "en": {
        "title":        "MLB Statcast Pro Dashboard",
        "subtitle":     "Pitch Analytics · Statcast Database 2021–2025",
        "main_sec":     "📊  Main Visualization",
        "cmp_sec":      "🔄  Side-by-Side Comparison",
        "sel_mode":     "Selection Mode",
        "mode_grp":     "Pitch Group",
        "mode_ind":     "Individual Pitch",
        "pitch_grp":    "Pitch Group",
        "pitch_ind":    "Pitch Type",
        "p_hand":       "Pitcher Handedness",
        "b_hand":       "Batter Handedness",
        "statistic":    "Statistic",
        "spin_lbl":     "Spin Rate (rpm)",
        "vel_lbl":      "Velocity Bin (mph)",
        "count_lbl":    "Count",
        "strikes_lbl":  "Strikes Filter",
        "year_lbl":     "Season",
        "hbrk_lbl":     "Horizontal Break (inches)",
        "vbrk_lbl":     "Vertical Break (inches)",
        "adv_flt":      "⚙️  Advanced Filters",
        "ref_hdr":      "📐  Pitch Movement Reference",
        "ref_sub":      "Typical values in inches — catcher's POV, RHP default",
        "h_break":      "Horizontal Break",
        "v_break":      "Vertical Break",
        "pitch_note":   "Scouting Notes",
        "zone_sum":     "Zone-by-Zone Summary",
        "data_tbl":     "Full Raw Data Table",
        "data_tbl_sub": "Showing up to 5,000 rows. Gradient styling applied when ≤ 5,000 rows.",
        "no_data":      "No data available for this combination of filters.",
        "no_file":      "CSV data file not found",
        "caption":      "Color scales are fixed across all configurations — comparisons are apples-to-apples.",
        "all":          "All",
        "cfg_a":        "⬅️  Configuration A",
        "cfg_b":        "➡️  Configuration B",
        "reload":       "🔄  Clear Cache & Reload",
        "reload_ok":    "✅  Cache cleared — reloading…",
        "n_pitches":    "pitches",
        "src":          "Source: MLB Statcast via pybaseball",
        "grp_hint":     "Spin / velocity bin filters are available in Individual Pitch mode only.",
        "rows_shown":   "rows shown",
        "of":           "of",
    },
    "pl": {
        "title":        "MLB Statcast Pro Dashboard",
        "subtitle":     "Analiza narzutów · Baza Statcast 2021–2025",
        "main_sec":     "📊  Główna wizualizacja",
        "cmp_sec":      "🔄  Porównanie konfiguracji",
        "sel_mode":     "Tryb wyboru",
        "mode_grp":     "Grupa narzutów",
        "mode_ind":     "Konkretny narzut",
        "pitch_grp":    "Grupa narzutów",
        "pitch_ind":    "Rodzaj narzutu",
        "p_hand":       "Ręka miotacza",
        "b_hand":       "Ręka pałkarza",
        "statistic":    "Statystyka",
        "spin_lbl":     "Obroty (rpm)",
        "vel_lbl":      "Prędkość (mph)",
        "count_lbl":    "Stan pojedynku",
        "strikes_lbl":  "Filtr strike'ów",
        "year_lbl":     "Sezon",
        "hbrk_lbl":     "H-Break (cale)",
        "vbrk_lbl":     "V-Break (cale)",
        "adv_flt":      "⚙️  Zaawansowane filtry",
        "ref_hdr":      "📐  Ruch narzutów — referencja",
        "ref_sub":      "Typowe wartości w calach — perspektywa łapacza, dla RHP",
        "h_break":      "Ruch poziomy",
        "v_break":      "Ruch pionowy",
        "pitch_note":   "Opis i strategia",
        "zone_sum":     "Podsumowanie stref",
        "data_tbl":     "Pełna tabela danych",
        "data_tbl_sub": "Pokazuje do 5 000 wierszy. Kolorowanie gradientowe przy ≤ 5 000 wierszy.",
        "no_data":      "Brak danych dla tej kombinacji filtrów.",
        "no_file":      "Nie znaleziono pliku CSV",
        "caption":      "Skala kolorów jest stała — porównania są uczciwe i czytelne.",
        "all":          "Wszystkie",
        "cfg_a":        "⬅️  Konfiguracja A",
        "cfg_b":        "➡️  Konfiguracja B",
        "reload":       "🔄  Wyczyść cache i przeładuj",
        "reload_ok":    "✅  Cache wyczyszczony — ładowanie…",
        "n_pitches":    "narzutów",
        "src":          "Źródło: MLB Statcast via pybaseball",
        "grp_hint":     "Filtry spin/velocity dostępne tylko w trybie pojedynczego narzutu.",
        "rows_shown":   "wierszy wyświetlono",
        "of":           "z",
    },
}

# ════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <span class="sb-logo-icon">⚾</span>
        <div class="sb-logo-name">MLB Statcast Pro</div>
        <div class="sb-logo-sub">Advanced Pitch Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    lang_sel = st.selectbox(
        "Language / Język",
        ["English 🇬🇧", "Polski 🇵🇱"],
        index=0,
        key="lang_sel",
    )
    lang     = "en" if "English" in lang_sel else "pl"
    T        = T_ALL[lang]
    note_key = "note_en" if lang == "en" else "note_pl"

    st.markdown("---")

    st.markdown(
        f'<div style="color:#79b8ff;font-weight:600;font-size:0.88rem;'
        f'margin-bottom:6px;">{T["ref_hdr"]}</div>',
        unsafe_allow_html=True,
    )
    st.caption(T["ref_sub"])

    ref_disp = st.selectbox(
        " ", list(PITCH_NAMES.values()),
        key="sb_ref_pitch", label_visibility="collapsed",
    )
    ref_code = CODE_FROM_DISP[ref_disp]
    ref      = MOVEMENT_REF[ref_code]

    st.markdown(f"""
    <div class="ref-card">
        <div class="ref-title">↔ {T['h_break']}</div>
        <span class="ref-badge">{ref['h_range']}</span>
        <span class="ref-badge ref-badge-dim">{ref['h_dir']}</span>
        <div class="ref-title" style="margin-top:10px;">↕ {T['v_break']}</div>
        <span class="ref-badge">{ref['v_range']}</span>
        <span class="ref-badge ref-badge-dim">{ref['v_dir']}</span>
    </div>
    <div class="ref-card">
        <div class="ref-title">📋 {T['pitch_note']}</div>
        <div class="ref-body">{ref[note_key]}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button(T["reload"], use_container_width=True):
        st.cache_data.clear()
        st.success(T["reload_ok"])
        st.rerun()
    st.markdown("---")
    st.caption(T["src"])

# ════════════════════════════════════════════════════════
# PAGE HEADER
# ════════════════════════════════════════════════════════
st.markdown(f"""
<div class="dash-header">
    <div class="dash-title">⚾ {T['title']}</div>
    <div class="dash-sub">{T['subtitle']}</div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════
@st.cache_data(show_spinner="⏳  Loading Statcast database…")
def load_all_data(path: str):
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df.columns = df.columns.str.strip()
    num_cols = [
        "total_pitches","swing_pct","whiff_pct","contact_pct",
        "barrel_pct","hard_hit_pct","sweet_spot_pct",
        "gb_pct","ld_pct","fb_pct",
        "avg_xwoba","avg_ev","avg_la","avg_hbreak","avg_vbreak",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "zone" in df.columns:
        df["zone"] = pd.to_numeric(df["zone"], errors="coerce").astype("Int64")
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df

df_all = load_all_data(DATA_FILE)

if df_all is None:
    st.error(f"⚠️  {T['no_file']}: `{DATA_FILE}`")
    st.info("Run the Jupyter notebook first, then place the CSV in the same folder as this script.")
    st.stop()

# Year options — safe, built after T is set
available_years = (
    sorted([str(y) for y in df_all["year"].dropna().unique().tolist()])
    if "year" in df_all.columns else []
)
YEAR_OPTIONS = [T["all"]] + available_years

# ════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════
def sort_key(b: str) -> float:
    b = str(b).strip()
    try:
        if b.startswith("<"): return float(b[1:]) - 0.1
        if b.startswith(">"): return float(b[1:])
        return float(b.split("-")[0])
    except Exception:
        return float("inf")

def get_bin_options(df_sub, col: str, all_text: str) -> list:
    if df_sub is None or df_sub.empty or col not in df_sub.columns:
        return [all_text]
    vals = sorted(df_sub[col].dropna().unique().tolist(), key=sort_key)
    return [all_text] + vals

def apply_filters(
    df, pitch_codes: list,
    phand, bhand,
    year_val, strikes_val, count_val,
    spin_val, vel_val,
    h_range, v_range,
    all_text: str,
):
    if df is None or not pitch_codes:
        return None

    mask = df["pitch_type"].isin(pitch_codes)

    if phand != all_text:
        mask &= df["p_throws"] == ("R" if "RHP" in phand else "L")
    if bhand != all_text:
        mask &= df["stand"]    == ("R" if "RHB" in bhand else "L")

    # Season — safe int conversion (prevents ValueError on "All")
    if year_val != all_text and str(year_val).strip().isdigit():
        mask &= df["year"] == int(year_val)

    if strikes_val != all_text:
        s = int(str(strikes_val).split()[0])
        mask &= df["count_state"].str.endswith(f"-{s}")

    if count_val != all_text:
        mask &= df["count_state"] == count_val

    if spin_val != all_text:
        mask &= df["spin_bin"]     == spin_val
    if vel_val  != all_text:
        mask &= df["velocity_bin"] == vel_val

    if "avg_hbreak" in df.columns:
        mask &= df["avg_hbreak"].between(h_range[0], h_range[1])
    if "avg_vbreak" in df.columns:
        mask &= df["avg_vbreak"].between(v_range[0], v_range[1])

    out = df[mask].copy()
    return out if not out.empty else None

def build_title(pitch_label, phand, bhand, year_val,
                strikes_val, count_val, stat_label, all_text) -> str:
    parts = [stat_label, pitch_label]
    if phand       != all_text: parts.append(phand)
    if bhand       != all_text: parts.append(bhand)
    if year_val    != all_text: parts.append(year_val)
    if strikes_val != all_text: parts.append(strikes_val)
    if count_val   != all_text: parts.append(f"Count {count_val}")
    return "  ·  ".join(parts)

# ════════════════════════════════════════════════════════
# HEATMAP — black text on coloured cells
# ════════════════════════════════════════════════════════
def draw_heatmap(df_f, stat_label: str, title: str):
    if df_f is None or df_f.empty:
        st.warning(T["no_data"])
        return

    col = STAT_COL[stat_label]
    if col not in df_f.columns:
        st.warning(f"Column `{col}` not found in data.")
        return

    pv = df_f.groupby("zone")[col].mean()
    pp = df_f.groupby("zone")["total_pitches"].sum()

    vmin, vmax = FIXED_RANGES[stat_label]
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    brd  = 0.85;  ms = 3.3
    mx   = brd;   my = brd
    top  = my + ms
    rx   = mx + ms
    hlf  = ms / 2
    sy   = 2.5
    cell = ms / 3

    fig, ax = plt.subplots(figsize=(7, 7.2))
    fig.patch.set_facecolor("#0b0f17")
    ax.set_facecolor("#0b0f17")

    def _fill(z):
        v, t = pv.get(z, np.nan), pp.get(z, 0)
        if pd.isna(v) or t == 0:
            return "#1c2230"
        return cmap(np.clip((v - vmin) / (vmax - vmin), 0, 1))

    def _label(z):
        v, t = pv.get(z, np.nan), pp.get(z, 0)
        if t == 0:
            return str(z)
        if stat_label == "xwOBA":
            return f"{z}\n{v:.3f}"
        if stat_label in ("Exit Velo", "Launch Angle"):
            return f"{z}\n{v:.1f}"
        return f"{z}\n{v:.1f}%"

    def _text_color(z):
        v = pv.get(z, np.nan)
        t = pp.get(z, 0)
        # dark empty cell → muted grey; any coloured cell → solid black
        if pd.isna(v) or t == 0:
            return "#4a5568"
        return "#111111"

    # Zones 1–9
    for i in range(3):
        for j in range(3):
            z = i * 3 + j + 1
            x = mx + j * cell
            y = my + (2 - i) * cell
            ax.add_patch(plt.Rectangle(
                (x, y), cell, cell,
                facecolor=_fill(z), edgecolor="#1e2535", linewidth=2.0,
            ))
            ax.text(
                x + cell / 2, y + cell / 2, _label(z),
                ha="center", va="center",
                fontsize=10.5, fontweight="bold",
                color=_text_color(z),
            )

    # Shadow zones 11–14
    shadows = [
        (11, [(0,sy),(brd,sy),(brd,top),(mx,top),(mx+hlf,top),(mx+hlf,5),(0,5),(0,sy)]),
        (12, [(rx,sy),(rx,top),(mx+hlf,top),(mx+hlf,5),(5,5),(5,sy),(rx,sy)]),
        (13, [(0,sy),(brd,sy),(brd,my),(mx,my),(mx+hlf,my),(mx+hlf,0),(0,0),(0,sy)]),
        (14, [(rx,sy),(rx,my),(mx+hlf,my),(mx+hlf,0),(5,0),(5,sy),(rx,sy)]),
    ]
    for z, verts in shadows:
        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
        ax.add_patch(PathPatch(
            Path(verts, codes),
            facecolor=_fill(z), edgecolor="#1e2535", linewidth=2.0,
        ))

    ax.add_patch(plt.Rectangle(
        (mx, my), ms, ms,
        fill=False, edgecolor="#f85149", linewidth=3.0, zorder=10,
    ))

    for z, xt, yt in [
        (11, brd / 2,     5 - brd / 2),
        (12, 5 - brd / 2, 5 - brd / 2),
        (13, brd / 2,     brd / 2),
        (14, 5 - brd / 2, brd / 2),
    ]:
        ax.text(
            xt, yt, _label(z),
            ha="center", va="center",
            fontsize=10.5, fontweight="bold",
            color=_text_color(z),
        )

    ax.set_xlim(0, 5)
    ax.set_ylim(-0.7, 5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11.5, pad=14, color="#c9d1d9", fontweight="600")

    sm   = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, ax=ax, shrink=0.68, pad=0.03)
    cbar.set_label(stat_label, fontsize=9, color="#8892a4")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8892a4", fontsize=8)
    cbar.outline.set_edgecolor("#1e2535")

    n_total = int(pp.sum())
    ax.text(2.5, -0.35, f"n = {n_total:,} {T['n_pitches']}",
            ha="center", fontsize=8.5, color="#718096", style="italic")

    if "avg_hbreak" in df_f.columns and "avg_vbreak" in df_f.columns:
        mh = df_f["avg_hbreak"].mean()
        mv = df_f["avg_vbreak"].mean()
        if not (pd.isna(mh) or pd.isna(mv)):
            ax.text(2.5, -0.56,
                    f"H-Break: {mh:+.1f}\"   V-Break: {mv:+.1f}\"",
                    ha="center", fontsize=8, color="#63b3ff",
                    fontfamily="monospace")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ════════════════════════════════════════════════════════
# ZONE SUMMARY TABLE
# ════════════════════════════════════════════════════════
def show_zone_summary(df_f):
    if df_f is None or df_f.empty:
        return
    st.markdown(f'<div class="sec-hdr">{T["zone_sum"]}</div>', unsafe_allow_html=True)

    wanted   = ["total_pitches","swing_pct","contact_pct","whiff_pct",
                "barrel_pct","hard_hit_pct","sweet_spot_pct",
                "gb_pct","ld_pct","fb_pct","avg_xwoba","avg_ev","avg_la"]
    present  = [c for c in wanted if c in df_f.columns]
    agg_dict = {c: ("sum" if c == "total_pitches" else "mean") for c in present}
    summary  = df_f.groupby("zone")[present].agg(agg_dict).round(2)
    summary  = summary.rename(columns={
        "total_pitches":"Pitches",
        "swing_pct":"Swing %","contact_pct":"Contact %","whiff_pct":"Whiff %",
        "barrel_pct":"Barrel %","hard_hit_pct":"Hard Hit %","sweet_spot_pct":"Sweet Spot %",
        "gb_pct":"GB %","ld_pct":"LD %","fb_pct":"FB %",
        "avg_xwoba":"xwOBA","avg_ev":"Exit Velo","avg_la":"Launch °",
    })

    def sfmt(fmt):
        def f(x):
            if pd.isna(x): return "—"
            try:   return fmt.format(x)
            except: return str(x)
        return f

    fmt_map = {
        "Pitches":    "{:.0f}",
        "Swing %":    sfmt("{:.1f}%"), "Contact %":    sfmt("{:.1f}%"),
        "Whiff %":    sfmt("{:.1f}%"), "Barrel %":     sfmt("{:.1f}%"),
        "Hard Hit %": sfmt("{:.1f}%"), "Sweet Spot %": sfmt("{:.1f}%"),
        "GB %":       sfmt("{:.1f}%"), "LD %":         sfmt("{:.1f}%"),
        "FB %":       sfmt("{:.1f}%"), "xwOBA":        sfmt("{:.3f}"),
        "Exit Velo":  sfmt("{:.1f}"),  "Launch °":     sfmt("{:.1f}"),
    }
    fmt_filtered = {k: v for k, v in fmt_map.items() if k in summary.columns}
    st.dataframe(summary.style.format(fmt_filtered), use_container_width=True, height=420)

# ════════════════════════════════════════════════════════
# FULL RAW DATA TABLE  — safe rendering
# ════════════════════════════════════════════════════════
def show_raw_table(df_f, stat_label: str):
    if df_f is None or df_f.empty:
        return

    show_cols = [c for c in [
        "zone","total_pitches","pitch_type","year","count_state",
        "p_throws","stand",
        "swing_pct","contact_pct","whiff_pct",
        "barrel_pct","hard_hit_pct","sweet_spot_pct",
        "gb_pct","ld_pct","fb_pct",
        "avg_xwoba","avg_ev","avg_la",
        "avg_hbreak","avg_vbreak",
        "spin_bin","velocity_bin",
    ] if c in df_f.columns]

    df_display = df_f[show_cols].sort_values("zone").reset_index(drop=True)
    total_rows = len(df_display)

    # Cap rows for gradient styling to stay well inside Streamlit's cell limit
    grad_col   = STAT_COL.get(stat_label, "")
    grad_cols  = [grad_col] if grad_col in df_display.columns else []

    st.caption(T["data_tbl_sub"])
    st.caption(f"{total_rows:,} {T['rows_shown']}")

    if total_rows <= MAX_STYLED_ROWS and grad_cols:
        # Safe to apply gradient
        st.dataframe(
            df_display.head(MAX_STYLED_ROWS)
            .style.background_gradient(subset=grad_cols, cmap="YlOrRd"),
            use_container_width=True,
            height=420,
        )
    else:
        # Too many rows — show plain dataframe (no styling crash)
        st.dataframe(
            df_display.head(MAX_STYLED_ROWS),
            use_container_width=True,
            height=420,
        )

# ════════════════════════════════════════════════════════
# PITCH SELECTION WIDGET
# ════════════════════════════════════════════════════════
def pitch_selection_widget(prefix: str):
    """Returns (pitch_codes, pitch_label, single_code | None)."""
    mode = st.radio(
        T["sel_mode"],
        [T["mode_grp"], T["mode_ind"]],
        horizontal=True,
        key=f"{prefix}_mode",
    )

    if mode == T["mode_grp"]:
        grp   = st.selectbox(T["pitch_grp"], list(PITCH_GROUPS.keys()), key=f"{prefix}_grp")
        codes = PITCH_GROUPS[grp]
        label = f"{grp}  ({' · '.join(codes)})"
        single_code = None
        badges = "".join(
            f'<span class="ref-badge">{PITCH_NAMES[c]}</span>'
            for c in codes
        )
        st.markdown(
            f'<div style="margin:-2px 0 10px 0;">{badges}</div>',
            unsafe_allow_html=True,
        )
    else:
        pitch_disp  = st.selectbox(T["pitch_ind"], list(PITCH_NAMES.values()), key=f"{prefix}_ind")
        single_code = CODE_FROM_DISP[pitch_disp]
        codes       = [single_code]
        label       = pitch_disp.split("(")[0].strip()

    return codes, label, single_code

# ════════════════════════════════════════════════════════
# FULL FILTER PANEL
# ════════════════════════════════════════════════════════
def filter_panel(prefix: str, default_stat: str = "Whiff %") -> dict:
    all_text = T["all"]

    # Pitch selection
    pitch_codes, pitch_label, single_code = pitch_selection_widget(prefix)

    # Row 1: Pitcher / Batter / Statistic
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        phand = st.selectbox(T["p_hand"], [all_text, "RHP", "LHP"], key=f"{prefix}_ph")
    with c2:
        bhand = st.selectbox(T["b_hand"], [all_text, "RHB", "LHB"], key=f"{prefix}_bh")
    with c3:
        def_idx    = STAT_LABELS.index(default_stat) if default_stat in STAT_LABELS else 0
        stat_label = st.selectbox(T["statistic"], STAT_LABELS, index=def_idx, key=f"{prefix}_stat")

    # Row 2: Season / Strikes / Count
    c4, c5, c6 = st.columns(3)
    with c4:
        year_val    = st.selectbox(T["year_lbl"],    YEAR_OPTIONS,                                   key=f"{prefix}_yr")
    with c5:
        strikes_val = st.selectbox(T["strikes_lbl"], [all_text,"0 strikes","1 strike","2 strikes"], key=f"{prefix}_str")
    with c6:
        count_val   = st.selectbox(T["count_lbl"],   [all_text] + ALL_COUNTS,                       key=f"{prefix}_cnt")

    # Advanced filters
    with st.expander(T["adv_flt"], expanded=False):

        df_sub = df_all[df_all["pitch_type"].isin(pitch_codes)].copy() if df_all is not None else None
        if df_sub is not None:
            if phand != all_text:
                df_sub = df_sub[df_sub["p_throws"] == ("R" if "RHP" in phand else "L")]
            if bhand != all_text:
                df_sub = df_sub[df_sub["stand"]    == ("R" if "RHB" in bhand else "L")]

        if single_code is not None:
            ca, cb = st.columns(2)
            with ca:
                spin_val = st.selectbox(
                    T["spin_lbl"],
                    get_bin_options(df_sub, "spin_bin", all_text),
                    key=f"{prefix}_sp",
                )
            with cb:
                vel_val = st.selectbox(
                    T["vel_lbl"],
                    get_bin_options(df_sub, "velocity_bin", all_text),
                    key=f"{prefix}_vl",
                )
        else:
            spin_val = all_text
            vel_val  = all_text
            st.caption(T["grp_hint"])

        # H / V break sliders
        h_min, h_max = -25.0, 25.0
        v_min, v_max = -25.0, 25.0
        if df_sub is not None:
            if "avg_hbreak" in df_sub.columns:
                hv = df_sub["avg_hbreak"].dropna()
                if not hv.empty:
                    h_min = float(np.floor(hv.min()))
                    h_max = float(np.ceil(hv.max()))
            if "avg_vbreak" in df_sub.columns:
                vv = df_sub["avg_vbreak"].dropna()
                if not vv.empty:
                    v_min = float(np.floor(vv.min()))
                    v_max = float(np.ceil(vv.max()))

        if single_code:
            ref_m  = MOVEMENT_REF.get(single_code, {})
            h_help = f"Typical: {ref_m.get('h_range','?')}  ({ref_m.get('h_dir','')})"
            v_help = f"Typical: {ref_m.get('v_range','?')}  ({ref_m.get('v_dir','')})"
        else:
            h_help = v_help = "Filter by average horizontal / vertical movement across the group"

        cc, cd = st.columns(2)
        with cc:
            h_range = st.slider(
                T["hbrk_lbl"],
                min_value=h_min, max_value=h_max,
                value=(h_min, h_max), step=0.5,
                key=f"{prefix}_hr", help=h_help,
            )
        with cd:
            v_range = st.slider(
                T["vbrk_lbl"],
                min_value=v_min, max_value=v_max,
                value=(v_min, v_max), step=0.5,
                key=f"{prefix}_vr", help=v_help,
            )

    return dict(
        pitch_codes=pitch_codes, pitch_label=pitch_label,
        phand=phand, bhand=bhand, stat_label=stat_label,
        year_val=year_val, strikes_val=strikes_val, count_val=count_val,
        spin_val=spin_val, vel_val=vel_val,
        h_range=h_range, v_range=v_range,
    )

# ════════════════════════════════════════════════════════
# MAIN VISUALIZATION
# ════════════════════════════════════════════════════════
st.markdown(f'<div class="sec-hdr">{T["main_sec"]}</div>', unsafe_allow_html=True)

cfg = filter_panel("main", default_stat="Whiff %")

df_f = apply_filters(
    df_all,
    cfg["pitch_codes"], cfg["phand"], cfg["bhand"],
    cfg["year_val"],    cfg["strikes_val"], cfg["count_val"],
    cfg["spin_val"],    cfg["vel_val"],
    cfg["h_range"],     cfg["v_range"],
    T["all"],
)

draw_heatmap(
    df_f, cfg["stat_label"],
    build_title(
        cfg["pitch_label"], cfg["phand"], cfg["bhand"],
        cfg["year_val"], cfg["strikes_val"], cfg["count_val"],
        cfg["stat_label"], T["all"],
    ),
)

show_zone_summary(df_f)

if df_f is not None:
    with st.expander(T["data_tbl"], expanded=False):
        show_raw_table(df_f, cfg["stat_label"])

# ════════════════════════════════════════════════════════
# SIDE-BY-SIDE COMPARISON
# ════════════════════════════════════════════════════════
st.markdown('<div class="dash-divider"></div>', unsafe_allow_html=True)
st.markdown(f'<div class="sec-hdr">{T["cmp_sec"]}</div>', unsafe_allow_html=True)

col_A, col_B = st.columns(2, gap="large")

with col_A:
    st.markdown(
        f'<div style="color:#79b8ff;font-weight:600;font-size:0.9rem;'
        f'margin-bottom:10px;">{T["cfg_a"]}</div>',
        unsafe_allow_html=True,
    )
    cfg_A = filter_panel("cmp_a", default_stat="Whiff %")
    df_A  = apply_filters(
        df_all,
        cfg_A["pitch_codes"], cfg_A["phand"], cfg_A["bhand"],
        cfg_A["year_val"],    cfg_A["strikes_val"], cfg_A["count_val"],
        cfg_A["spin_val"],    cfg_A["vel_val"],
        cfg_A["h_range"],     cfg_A["v_range"],
        T["all"],
    )
    draw_heatmap(
        df_A, cfg_A["stat_label"],
        build_title(
            cfg_A["pitch_label"], cfg_A["phand"], cfg_A["bhand"],
            cfg_A["year_val"], cfg_A["strikes_val"], cfg_A["count_val"],
            cfg_A["stat_label"], T["all"],
        ),
    )

with col_B:
    st.markdown(
        f'<div style="color:#79b8ff;font-weight:600;font-size:0.9rem;'
        f'margin-bottom:10px;">{T["cfg_b"]}</div>',
        unsafe_allow_html=True,
    )
    cfg_B = filter_panel("cmp_b", default_stat="xwOBA")
    df_B  = apply_filters(
        df_all,
        cfg_B["pitch_codes"], cfg_B["phand"], cfg_B["bhand"],
        cfg_B["year_val"],    cfg_B["strikes_val"], cfg_B["count_val"],
        cfg_B["spin_val"],    cfg_B["vel_val"],
        cfg_B["h_range"],     cfg_B["v_range"],
        T["all"],
    )
    draw_heatmap(
        df_B, cfg_B["stat_label"],
        build_title(
            cfg_B["pitch_label"], cfg_B["phand"], cfg_B["bhand"],
            cfg_B["year_val"], cfg_B["strikes_val"], cfg_B["count_val"],
            cfg_B["stat_label"], T["all"],
        ),
    )

# ════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════
st.markdown('<div class="dash-divider"></div>', unsafe_allow_html=True)
st.caption(T["caption"])
st.caption(T["src"])