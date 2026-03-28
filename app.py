import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from matplotlib.path import Path
from matplotlib.patches import PathPatch

# ================================================
# LANGUAGE SELECTION
# ================================================
if "language" not in st.session_state:
    st.session_state.language = "en"

lang_options = ["English 🇬🇧", "Polski 🇵🇱", "Español 🇪🇸", "Français 🇫🇷", "日本語 🇯🇵"]
selected_lang_display = st.selectbox(
    "Język / Language / Idioma / Langue / 言語",
    options=lang_options,
    index=0
)

lang_map = {
    "English 🇬🇧": "en",
    "Polski 🇵🇱": "pl",
    "Español 🇪🇸": "es",
    "Français 🇫🇷": "fr",
    "日本語 🇯🇵": "ja"
}
st.session_state.language = lang_map[selected_lang_display]
lang = st.session_state.language

# ================================================
# PITCH NAMES
# ================================================
PITCH_FULL_NAMES = {
    "en": {
        "FF": "Four-seam fastball (FF)", "SL": "Slider (SL)", "CH": "Changeup (CH)",
        "CU": "Curveball (CU)", "SV": "Slurve (SV)", "SI": "Sinker (SI)",
        "FC": "Cutter (FC)", "ST": "Sweeper (ST)", "KC": "Knuckle curve (KC)",
        "FS": "Split-finger (FS)"
    },
    "pl": {
        "FF": "Four-seam fastball (FF)", "SL": "Slider (SL)", "CH": "Changeup (CH)",
        "CU": "Curveball (CU)", "SV": "Slurve (SV)", "SI": "Sinker (SI)",
        "FC": "Cutter (FC)", "ST": "Sweeper (ST)", "KC": "Knuckle curve (KC)",
        "FS": "Split-finger (FS)"
    }
}

PITCH_FULL_NAMES_LANG = PITCH_FULL_NAMES.get(lang, PITCH_FULL_NAMES["en"])
pitch_display_to_code = {v: k for k, v in PITCH_FULL_NAMES_LANG.items()}
pitch_display_list = list(PITCH_FULL_NAMES_LANG.values())

# ================================================
# TRANSLATIONS
# ================================================
TRANSLATIONS = {
    "en": {
        "title": "📊 Statcast Dashboard – All Pitches",
        "main_section": "Main Visualization",
        "pitch_type": "Pitch Type",
        "pitcher_hand": "Pitcher Hand",
        "batter_hand": "Batter Hand",
        "statistic": "Statistic",
        "spin_bin": "Spin Rate Range (spin bin)",
        "velocity": "Pitch Velocity (mph bin)",
        "data_table": "Full Data Table",
        "comparison_section": "Comparison of Two Configurations",
        "left_config": "**Left Configuration**",
        "right_config": "**Right Configuration**",
        "no_data": "No data for this configuration",
        "no_file": "No file found for selected configuration",
        "caption": "Color scale is fixed – comparisons are fair and clear.",
        "all": "All",
        "zone_summary": "Zone Summary"
    },
    "pl": {
        "title": "📊 Dashboard Statcast – Wszystkie narzuty",
        "main_section": "Główna wizualizacja",
        "pitch_type": "Rodzaj rzutu",
        "pitcher_hand": "Ręka miotacza",
        "batter_hand": "Ręka pałkarza",
        "statistic": "Statystyka",
        "spin_bin": "Zakres obrotów (spin bin)",
        "velocity": "Prędkość narzutu (mph bin)",
        "data_table": "Pełna tabela danych",
        "comparison_section": "Porównanie dwóch konfiguracji",
        "left_config": "**Lewa konfiguracja**",
        "right_config": "**Prawa konfiguracja**",
        "no_data": "Brak danych dla tej konfiguracji",
        "no_file": "Brak pliku dla wybranej konfiguracji",
        "caption": "Skala kolorów jest stała – porównania są uczciwe i czytelne.",
        "all": "Wszystkie",
        "zone_summary": "Podsumowanie według stref"
    }
}

T = TRANSLATIONS.get(lang, TRANSLATIONS["en"])

# ================================================
# HELPER FUNCTIONS
# ================================================
def get_lower_bound(b):
    b_str = str(b).strip()
    try:
        if b_str.startswith('>'):
            return float(b_str[1:])
        if b_str.startswith('<'):
            return float(b_str[1:]) - 0.1
        if '-' in b_str:
            return float(b_str.split('-')[0].strip())
        return float(b_str)
    except:
        return float('inf')

def get_spin_bin_options(df, all_text):
    if df is None or df.empty:
        return [all_text]
    bins = df['spin_bin'].dropna().unique().tolist()
    sorted_bins = sorted(bins, key=get_lower_bound)
    return [all_text] + sorted_bins

def get_velocity_bin_options(df, all_text, pitch_code):
    if df is None or df.empty:
        return [all_text]
    
    bins = df['velocity_bin'].dropna().unique().tolist()
    
    slow_pitches = {"SL", "KC", "CH", "ST"}
    
    if pitch_code == "CU":
        special = [">90"] if ">90" in bins else []
    elif pitch_code in slow_pitches:
        special = ["<70"] if "<70" in bins else []
        if "70-75" in bins:
            special.append("70-75")
    else:
        special = [">70"] if ">70" in bins else []
        if "70-75" in bins:
            special.append("70-75")
    
    rest = [b for b in bins if b not in special]
    rest_sorted = sorted(rest, key=get_lower_bound)
    
    return [all_text] + special + rest_sorted

# ================================================
# APP CONFIGURATION
# ================================================
st.set_page_config(page_title="Baseball Dashboard", layout="wide")
st.title(T["title"])

# Force reload fresh data (clears cache when new CSV files are used)
force_reload = st.checkbox("🔄 Force reload fresh CSV data (clear cache)", value=False)

if force_reload:
    st.cache_data.clear()
    st.success("✅ Cache cleared – loading latest CSV files with updated spin bins...")
    st.rerun()

DATA_FOLDER = ""

fixed_ranges = {
    "whiff_pct": (0, 70), "swing_pct": (10, 90), "avg_xwoba": (0.10, 0.55),
    "avg_ev": (70, 98), "avg_la": (-15, 45),
    "barrel_pct": (0, 25), "hard_hit_pct": (0, 60), "sweet_spot_pct": (0, 50),
    "gb_pct": (0, 80), "ld_pct": (0, 80), "fb_pct": (0, 80),
}

pitcher_hands = ["RHP", "LHP"]
batter_hands = ["RHB", "LHB"]

filename_map = {
    "FF": "fastballs", "SL": "sliders", "CH": "changeups", "CU": "curveballs",
    "SV": "slurves", "SI": "sinkers", "FC": "cutters", "ST": "sweepers",
    "KC": "knucklecurves", "FS": "splitters"
}

# ================================================
# DATA LOADING
# ================================================
@st.cache_data
def load_data(pitch_code, phand, bhand, force_reload=False):
    pitch_name = filename_map[pitch_code]
    filename = f"{pitch_name}_{phand}_vs_{bhand}_2021-2025_all_zones_spin_swing_whiff.csv"
    filepath = os.path.join(DATA_FOLDER, filename)
    if not os.path.exists(filepath):
        st.error(f"{T['no_file']}: {filename}")
        return None
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    numeric_cols = ['total_pitches','swing_pct','whiff_pct','barrel_pct','hard_hit_pct',
                    'sweet_spot_pct','gb_pct','ld_pct','fb_pct','avg_xwoba','avg_ev','avg_la']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['zone'] = pd.to_numeric(df['zone'], errors='coerce').astype('Int64')
    return df

# ================================================
# HEATMAP DRAWING
# ================================================
def draw_heatmap(df_filtered, stat, title):
    if df_filtered is None or df_filtered.empty:
        st.warning(T["no_data"])
        return

    pivot_stat = df_filtered.groupby('zone')[stat].mean()
    pivot_pitches = df_filtered.groupby('zone')['total_pitches'].sum()

    vmin, vmax = fixed_ranges.get(stat, (0, 100))
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    fig, ax = plt.subplots(figsize=(8.6, 8.6))

    border = 0.85
    main_size = 3.3
    main_x = border
    main_y = border
    top_y = main_y + main_size
    right_x = main_x + main_size
    half = main_size / 2
    split_y = 2.5

    for i in range(3):
        for j in range(3):
            zone = i * 3 + j + 1
            val = pivot_stat.get(zone, np.nan)
            total_p = pivot_pitches.get(zone, 0)
            x = main_x + j * (main_size / 3)
            y = main_y + (2 - i) * (main_size / 3)

            color = (0.96, 0.96, 0.96) if pd.isna(val) or total_p == 0 else cmap(np.clip((val - vmin) / (vmax - vmin), 0, 1))
            ax.add_patch(plt.Rectangle((x, y), main_size / 3, main_size / 3, facecolor=color, edgecolor='black', linewidth=2.8))

            if total_p == 0:
                text = str(zone)
            else:
                text = f"{zone}\n{val:.3f}" if stat == 'avg_xwoba' else f"{zone}\n{val:.1f}" if stat in ['avg_ev', 'avg_la'] else f"{zone}\n{val:.1f}%"
            ax.text(x + (main_size / 6), y + (main_size / 6), text, ha='center', va='center', fontsize=12, fontweight='bold')

    def get_val_pitches(z):
        return pivot_stat.get(z, np.nan), pivot_pitches.get(z, 0)

    for z, verts, codes_len in [(11, [(0, split_y),(border,split_y),(border,top_y),(main_x,top_y),(main_x+half,top_y),(main_x+half,5),(0,5),(0,split_y)], 7),
                                (12, [(right_x,split_y),(right_x,top_y),(main_x+half,top_y),(main_x+half,5),(5,5),(5,split_y),(right_x,split_y)], 6),
                                (13, [(0,split_y),(border,split_y),(border,main_y),(main_x,main_y),(main_x+half,main_y),(main_x+half,0),(0,0),(0,split_y)], 7),
                                (14, [(right_x,split_y),(right_x,main_y),(main_x+half,main_y),(main_x+half,0),(5,0),(5,split_y),(right_x,split_y)], 6)]:
        val, total = get_val_pitches(z)
        color = (0.96, 0.96, 0.96) if pd.isna(val) or total == 0 else cmap(np.clip((val - vmin) / (vmax - vmin), 0, 1))
        ax.add_patch(PathPatch(Path(verts, [Path.MOVETO] + [Path.LINETO] * codes_len), facecolor=color, edgecolor='black', linewidth=2.8))

    ax.add_patch(plt.Rectangle((main_x, main_y), main_size, main_size, fill=False, edgecolor='red', linewidth=4.2))

    offset = border * 0.38
    for z, val, total, x_text, y_text in [(11, *get_val_pitches(11), offset, 5-offset),
                                          (12, *get_val_pitches(12), 5-offset, 5-offset),
                                          (13, *get_val_pitches(13), offset, offset),
                                          (14, *get_val_pitches(14), 5-offset, offset)]:
        text = str(z) if total == 0 else (f"{z}\n{val:.3f}" if stat == 'avg_xwoba' else f"{z}\n{val:.1f}" if stat in ['avg_ev', 'avg_la'] else f"{z}\n{val:.1f}%")
        ax.text(x_text, y_text, text, ha='center', va='center', fontsize=12, fontweight='bold')

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=17, pad=25)

    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, ax=ax, shrink=0.78, pad=0.04)
    cbar.set_label(stat.upper().replace('_PCT', ' %'), fontsize=12)
    st.pyplot(fig, use_container_width=True)

# ================================================
# MAIN VISUALIZATION
# ================================================
st.subheader(T["main_section"])
c1, c2, c3, c4 = st.columns(4)
with c1:
    selected_pitch_display = st.selectbox(T["pitch_type"], pitch_display_list, key="main_pitch")
    main_pitch_code = pitch_display_to_code[selected_pitch_display]
with c2:
    main_phand = st.selectbox(T["pitcher_hand"], pitcher_hands, key="main_phand")
with c3:
    main_bhand = st.selectbox(T["batter_hand"], batter_hands, key="main_bhand")
with c4:
    main_stat = st.selectbox(T["statistic"], list(fixed_ranges.keys()), key="main_stat")

df_main = load_data(main_pitch_code, main_phand, main_bhand, force_reload)

if df_main is not None:
    all_text = T["all"]
    
    spin_options = get_spin_bin_options(df_main, all_text)
    main_spin = st.selectbox(T["spin_bin"], spin_options, key="main_spin")
    
    vel_options = get_velocity_bin_options(df_main, all_text, main_pitch_code)
    main_vel = st.selectbox(T["velocity"], vel_options, key="main_vel")

    df_main_filtered = df_main.copy()
    if main_spin != all_text:
        df_main_filtered = df_main_filtered[df_main_filtered['spin_bin'] == main_spin]
    if main_vel != all_text:
        df_main_filtered = df_main_filtered[df_main_filtered['velocity_bin'] == main_vel]

    draw_heatmap(df_main_filtered, main_stat, f"{main_stat.upper()} – {selected_pitch_display} | {main_phand} vs {main_bhand}")

    st.subheader(T["zone_summary"])
    summary = df_main_filtered.groupby('zone').agg({
        'total_pitches': 'sum', 'swing_pct': 'mean', 'whiff_pct': 'mean',
        'barrel_pct': 'mean', 'hard_hit_pct': 'mean', 'sweet_spot_pct': 'mean',
        'gb_pct': 'mean', 'ld_pct': 'mean', 'fb_pct': 'mean',
        'avg_xwoba': 'mean', 'avg_ev': 'mean', 'avg_la': 'mean'
    }).round(2)

    for col in summary.columns:
        if col != 'total_pitches':
            summary[col] = summary[col].where(summary['total_pitches'] > 0, "-")

    summary = summary[['total_pitches', 'swing_pct', 'whiff_pct', 'barrel_pct', 'hard_hit_pct',
                       'sweet_spot_pct', 'gb_pct', 'ld_pct', 'fb_pct', 'avg_xwoba', 'avg_ev', 'avg_la']]

    def safe_fmt(fmt_str):
        def formatter(x):
            if x == "-" or pd.isna(x):
                return "-"
            try:
                return fmt_str.format(x)
            except:
                return str(x)
        return formatter

    st.dataframe(summary.style.format({
        'total_pitches': '{:.0f}',
        'swing_pct': safe_fmt('{:.1f}%'), 'whiff_pct': safe_fmt('{:.1f}%'),
        'barrel_pct': safe_fmt('{:.1f}%'), 'hard_hit_pct': safe_fmt('{:.1f}%'),
        'sweet_spot_pct': safe_fmt('{:.1f}%'), 'gb_pct': safe_fmt('{:.1f}%'),
        'ld_pct': safe_fmt('{:.1f}%'), 'fb_pct': safe_fmt('{:.1f}%'),
        'avg_xwoba': safe_fmt('{:.3f}'), 'avg_ev': safe_fmt('{:.1f}'),
        'avg_la': safe_fmt('{:.1f}')
    }), use_container_width=True)

    st.subheader(T["data_table"])
    st.dataframe(df_main_filtered.style.background_gradient(subset=[main_stat], cmap='YlOrRd'))

else:
    st.error(T["no_file"])

# ================================================
# COMPARISON SECTION
# ================================================
st.subheader(T["comparison_section"])
col_left, col_right = st.columns(2)

with col_left:
    st.markdown(T["left_config"])
    cl1, cl2 = st.columns([3, 2])
    with cl1:
        left_pitch_display = st.selectbox(T["pitch_type"], pitch_display_list, key="left_pitch")
    left_pitch_code = pitch_display_to_code[left_pitch_display]
    with cl2:
        left_phand = st.selectbox(T["pitcher_hand"], pitcher_hands, key="left_phand")
    cl3, cl4 = st.columns([3, 2])
    with cl3:
        left_bhand = st.selectbox(T["batter_hand"], batter_hands, key="left_bhand")
    with cl4:
        left_stat = st.selectbox(T["statistic"], list(fixed_ranges.keys()), key="left_stat")
    
    df_left = load_data(left_pitch_code, left_phand, left_bhand, force_reload)
    if df_left is not None:
        all_text = T["all"]
        left_spin = st.selectbox(T["spin_bin"], get_spin_bin_options(df_left, all_text), key="left_spin")
        left_vel = st.selectbox(T["velocity"], get_velocity_bin_options(df_left, all_text, left_pitch_code), key="left_vel")
        df_left_f = df_left.copy()
        if left_spin != all_text:
            df_left_f = df_left_f[df_left_f['spin_bin'] == left_spin]
        if left_vel != all_text:
            df_left_f = df_left_f[df_left_f['velocity_bin'] == left_vel]
        draw_heatmap(df_left_f, left_stat, f"LEFT: {left_stat.upper()} – {left_pitch_display} | {left_phand} vs {left_bhand}")

with col_right:
    st.markdown(T["right_config"])
    cr1, cr2 = st.columns([3, 2])
    with cr1:
        right_pitch_display = st.selectbox(T["pitch_type"], pitch_display_list, key="right_pitch")
    right_pitch_code = pitch_display_to_code[right_pitch_display]
    with cr2:
        right_phand = st.selectbox(T["pitcher_hand"], pitcher_hands, key="right_phand")
    cr3, cr4 = st.columns([3, 2])
    with cr3:
        right_bhand = st.selectbox(T["batter_hand"], batter_hands, key="right_bhand")
    with cr4:
        right_stat = st.selectbox(T["statistic"], list(fixed_ranges.keys()), key="right_stat")
    
    df_right = load_data(right_pitch_code, right_phand, right_bhand, force_reload)
    if df_right is not None:
        all_text = T["all"]
        right_spin = st.selectbox(T["spin_bin"], get_spin_bin_options(df_right, all_text), key="right_spin")
        right_vel = st.selectbox(T["velocity"], get_velocity_bin_options(df_right, all_text, right_pitch_code), key="right_vel")
        df_right_f = df_right.copy()
        if right_spin != all_text:
            df_right_f = df_right_f[df_right_f['spin_bin'] == right_spin]
        if right_vel != all_text:
            df_right_f = df_right_f[df_right_f['velocity_bin'] == right_vel]
        draw_heatmap(df_right_f, right_stat, f"RIGHT: {right_stat.upper()} – {right_pitch_display} | {right_phand} vs {right_bhand}")

st.caption(T["caption"])
