"""
app.py
======
Interface Streamlit du French PPA Model.

Pages :
  🏠  Accueil        — présentation du modèle et navigation
  📊  Data Explorer  — visualisation des données (reprise du notebook explore_ppa_data_france)
  ⚙️  Scénario       — configuration et lancement du calcul
  📈  Résultats       — cashflow, VAN, métriques de sites
  🗺️  Carte GIS       — carte interactive des parcelles candidates

Lancement :
  streamlit run app.py
  streamlit run app.py -- --db database/   # chemin custom
"""

from __future__ import annotations
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="French PPA Model",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLE GLOBAL — Palette industrielle sobre, typography DM Sans / JetBrains Mono
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1923;
    border-right: 1px solid #1e2d3d;
}
section[data-testid="stSidebar"] * {
    color: #c8d8e8 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.88rem;
    padding: 4px 0;
}

/* Métriques */
[data-testid="stMetric"] {
    background: #1a2332;
    border: 1px solid #243447;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace;
    font-size: 1.6rem !important;
    color: #4fc3f7 !important;
}
[data-testid="stMetricLabel"] {
    color: #7a9bb5 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Onglets */
.stTabs [data-baseweb="tab-list"] {
    background: #1a2332;
    border-radius: 8px;
    padding: 4px;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    color: #7a9bb5;
    border-radius: 6px;
    font-weight: 500;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: #243447 !important;
    color: #4fc3f7 !important;
}

/* Fond principal */
.main .block-container {
    background: #0d1521;
    color: #c8d8e8;
    padding-top: 1.5rem;
}

/* Titres */
h1, h2, h3 { color: #e8f4fd; }
h1 { font-weight: 700; letter-spacing: -0.02em; }

/* Boutons */
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.55rem 1.8rem;
    font-family: 'DM Sans', sans-serif;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1976d2, #1565c0);
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(21,101,192,0.4);
}

/* Selectbox / sliders */
.stSelectbox > div, .stSlider > div {
    background: #1a2332;
}

/* Séparateur */
hr { border-color: #1e2d3d !important; }

/* Badge statut */
.badge-green  { background:#1b3a2a; color:#4caf50; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-orange { background:#3a2a0a; color:#ff9800; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-red    { background:#3a0a0a; color:#f44336; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }

/* Cards info */
.info-card {
    background: #1a2332;
    border: 1px solid #243447;
    border-left: 3px solid #4fc3f7;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
    color: #c8d8e8;
}

/* Plotly dark override */
.js-plotly-plot { border-radius: 10px; overflow: hidden; }

/* Expander */
details { background: #1a2332; border-radius: 8px; padding: 0 12px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHEMINS & CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

_HERE   = Path(__file__).resolve().parent
_DB_DIR = _HERE / "database"
_GIS_DIR = _HERE / "gisdata"

PLOTLY_TEMPLATE = "plotly_dark"
MONTH_NAMES = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]
COLORS_TECH = {
    "solar":   "#f5a623",
    "offshore":"#4fc3f7",
    "onshore": "#81c784",
    "hybrid":  "#ce93d8",
    "load":    "#ef9a9a",
    "grid":    "#607d8b",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DONNÉES
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_grid_france(db_dir: str) -> pd.DataFrame | None:
    p = Path(db_dir) / "grid_france.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, index_col="year")


@st.cache_data(ttl=300)
def load_solar_db(db_dir: str) -> pd.DataFrame | None:
    p = Path(db_dir) / "solar_patterns.db"
    if not p.exists():
        return None
    conn = sqlite3.connect(str(p))
    try:
        df = pd.read_sql("SELECT * FROM solar_patterns", conn, index_col="datetime",
                         parse_dates=["datetime"])
        conn.close()
        df.index = df.index.floor("h")
        return df[~df.index.duplicated(keep="first")]
    except Exception:
        conn.close()
        return None


@st.cache_data(ttl=300)
def load_wind_db(db_dir: str) -> pd.DataFrame | None:
    p = Path(db_dir) / "wind_patterns.db"
    if not p.exists():
        return None
    conn = sqlite3.connect(str(p))
    try:
        df = pd.read_sql("SELECT * FROM wind_patterns", conn, index_col="datetime",
                         parse_dates=["datetime"])
        conn.close()
        df.index = df.index.floor("h")
        return df[~df.index.duplicated(keep="first")]
    except Exception:
        conn.close()
        return None


@st.cache_data(ttl=300)
def load_load_db(db_dir: str) -> pd.DataFrame | None:
    p = Path(db_dir) / "load_patterns.db"
    if not p.exists():
        return None
    conn = sqlite3.connect(str(p))
    try:
        df = pd.read_sql("SELECT * FROM load_patterns", conn, index_col="datetime",
                         parse_dates=["datetime"])
        conn.close()
        return df
    except Exception:
        conn.close()
        return None


@st.cache_data(ttl=300)
def load_epex_db(db_dir: str, year: int = 2020) -> pd.Series | None:
    p = Path(db_dir) / "epex_profiles.db"
    if not p.exists():
        return None
    conn = sqlite3.connect(str(p))
    try:
        df = pd.read_sql(f"SELECT * FROM epex_{year}", conn,
                         index_col="datetime", parse_dates=["datetime"])
        conn.close()
        return df["price_eur_mwh"]
    except Exception:
        conn.close()
        return None


@st.cache_data(ttl=300)
def load_capture_rates(db_dir: str) -> pd.DataFrame | None:
    p = Path(db_dir) / "capture_rates.db"
    if not p.exists():
        return None
    conn = sqlite3.connect(str(p))
    try:
        df = pd.read_sql("SELECT * FROM capture_rates", conn)
        conn.close()
        return df
    except Exception:
        conn.close()
        return None


@st.cache_data(ttl=300)
def load_wind_grid(db_dir: str) -> pd.DataFrame | None:
    p = Path(db_dir) / "wind_grid_france.xlsx"
    if not p.exists():
        return None
    return pd.read_excel(p)


@st.cache_data(ttl=300)
def load_scenario_defaults(xlsx_path: str) -> pd.DataFrame | None:
    p = Path(xlsx_path)
    if not p.exists():
        return None
    try:
        df = pd.read_excel(p, header=3)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception:
        return None


def _db_status(db_dir: str) -> dict:
    """Vérifie l'existence des fichiers de données."""
    ddir = Path(db_dir)
    files = {
        "grid_france.csv":      "Trajectoires CAPEX/CO2/ENR",
        "solar_patterns.db":    "Profils solaires (10 sites)",
        "wind_patterns.db":     "Profils éoliens (10 zones)",
        "load_patterns.db":     "Profils de charge (4 types)",
        "epex_profiles.db":     "Prix EPEX spot horaire",
        "capture_rates.db":     "Capture rates calculés",
        "wind_grid_france.xlsx":"Parcs éoliens (12 parcs)",
        "TURPE_france.xlsx":    "Tarifs TURPE HTB",
    }
    return {f: (ddir / f).exists() for f in files}, files


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(db_dir_state: str) -> tuple[str, str]:
    with st.sidebar:
        st.markdown("## ⚡ French PPA Model")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["🏠 Accueil", "📊 Data Explorer", "⚙️ Scénario & Calcul",
             "📈 Résultats", "🗺️ Carte GIS"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**📂 Répertoire database**")
        db_dir = st.text_input("", value=db_dir_state, label_visibility="collapsed",
                                key="db_dir_input")

        # Status rapide des fichiers
        status_map, labels = _db_status(db_dir)
        n_ok  = sum(status_map.values())
        n_tot = len(status_map)
        color = "badge-green" if n_ok == n_tot else ("badge-orange" if n_ok >= 4 else "badge-red")
        st.markdown(
            f'<span class="{color}">{n_ok}/{n_tot} fichiers présents</span>',
            unsafe_allow_html=True
        )

        with st.expander("Détail des fichiers"):
            for fname, ok in status_map.items():
                icon = "✅" if ok else "❌"
                st.markdown(f"{icon} `{fname}`")

        st.markdown("---")
        st.markdown(
            "<small style='color:#556b7d'>French PPA Model v2.0<br>"
            "Données : PVGIS · ERA5 · EPEX · CRE</small>",
            unsafe_allow_html=True
        )

    return page, db_dir


# ─────────────────────────────────────────────────────────────────────────────
# PAGE : ACCUEIL
# ─────────────────────────────────────────────────────────────────────────────

def page_accueil(db_dir: str) -> None:
    st.markdown("# ⚡ French PPA Model")
    st.markdown(
        "<p style='color:#7a9bb5;font-size:1.1rem;margin-top:-0.5rem'>"
        "Modèle d'optimisation des contrats d'achat d'électricité renouvelable "
        "pour industriels français — HTB / EPEX Spot / TURPE / Garanties d'Origine"
        "</p>", unsafe_allow_html=True
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    status_map, _ = _db_status(db_dir)
    n_ok = sum(status_map.values())

    with col1:
        st.metric("Fichiers données", f"{n_ok}/8", "base de données")
    with col2:
        solar_df = load_solar_db(db_dir)
        n_sites = len(solar_df.columns) if solar_df is not None else 0
        st.metric("Sites solaires", n_sites, "PVGIS / synthétique")
    with col3:
        wind_df = load_wind_db(db_dir)
        n_wind = len(wind_df.columns) if wind_df is not None else 0
        st.metric("Zones éoliennes", n_wind, "ERA5 / synthétique")
    with col4:
        cr_df = load_capture_rates(db_dir)
        n_cr = len(cr_df) if cr_df is not None else 0
        st.metric("Capture rates", n_cr, "sites calculés")

    st.markdown("---")

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("### Architecture du modèle")
        st.markdown("""
<div class="info-card">
<b>Pipeline de calcul</b><br>
<code>download_france_data.py</code> → données marché (PVGIS, ERA5, EPEX, TURPE)<br>
<code>build_land_grid.py</code> → données foncières (RPG IGN, DVF, INPN)<br>
<code>FranceGridUtils.py</code> → métriques de sites (capture rate, corrélation, screening)<br>
<code>ppamodule.py</code> → optimisation LP mix PPA + settlement + NPV<br>
<code>app.py</code> → interface Streamlit (ce fichier)
</div>
<div class="info-card">
<b>Concepts clés</b><br>
• <b>Capture Rate</b> : valeur de marché relative de la production (CR > 1 = éolien hivernal)<br>
• <b>TURPE HTB</b> : composante réseau (puissance + énergie), fixée par la CRE<br>
• <b>GO</b> : Garanties d'Origine (équivalent français des RECs), ~5-15 €/MWh<br>
• <b>Sleeving</b> : fournisseur intermédiaire gérant les écarts de production (3-5 €/MWh)<br>
• <b>vPPA</b> : contrat financier (CfD), pas de transfert physique d'électrons
</div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("### Scénarios disponibles")
        xlsx = _DB_DIR / "scenario_defaults.xlsx"
        sd = load_scenario_defaults(str(xlsx))
        if sd is not None:
            scen_cols = [c for c in sd.columns
                         if c not in ("Scenario Name", "Source / Explication")
                         and not c.startswith("Unnamed")]
            for sc in scen_cols:
                load_row = sd[sd["Scenario Name"].astype(str).str.contains("Load", case=False, na=False)]
                load_val = ""
                if not load_row.empty and sc in load_row.columns:
                    vals = load_row[sc].dropna().tolist()
                    if vals:
                        load_val = f" — {vals[0]} MW"
                st.markdown(
                    f'<div class="info-card" style="border-left-color:#f5a623;">'
                    f'<b>{sc}</b>{load_val}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.warning("scenario_defaults.xlsx non trouvé dans ce répertoire.")

        st.markdown("---")
        st.markdown("### ▶ Démarrage rapide")
        st.code("""# 1. Générer les données
python download_france_data.py --offline

# 2. Données foncières (optionnel)
python build_land_grid.py --lat 50.93 --lon 2.38

# 3. Lancer l'interface
streamlit run app.py""", language="bash")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE : DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────

def page_data_explorer(db_dir: str) -> None:
    st.markdown("# 📊 Data Explorer")
    st.markdown(
        "<p style='color:#7a9bb5'>Visualisation de toutes les données générées par "
        "<code>download_france_data.py</code></p>", unsafe_allow_html=True
    )

    tabs = st.tabs([
        "📈 Grid France",
        "☀️ Profils Solaires",
        "💨 Profils Éoliens",
        "🏭 Charge Industrielle",
        "⚡ EPEX Spot",
        "🎯 Capture Rates",
        "🌬️ Parcs Éoliens",
        "🔄 Complémentarité",
        "📋 Qualité données",
    ])

    # ── Tab 1 : Grid France ──────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("#### Trajectoires CAPEX / CO2 / ENR — 2023→2050")
        df = load_grid_france(db_dir)
        if df is None:
            st.error("grid_france.csv non trouvé — exécuter `download_france_data.py`")
            return

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("CO2 réseau 2024", f"{df.loc[2024,'co2_intensity_g_kwh']:.0f} gCO2/kWh", "vs Corée ~450")
        with c2: st.metric("ENR 2024", f"{df.loc[2024,'ren_share']*100:.0f}%", "→ 80% en 2050")
        with c3: st.metric("CAPEX PV 2024", f"{df.loc[2024,'solar_capex']/1000:.0f} k€/MW", "→ 300 k€/MW en 2050")
        with c4: st.metric("EPEX moyen 2024", f"{df.loc[2024,'epex_spot_mean']:.0f} €/MWh", "vs 96 €/MWh 2023")

        fig = make_subplots(rows=2, cols=3,
            subplot_titles=[
                "CO2 réseau (gCO2/kWh)", "Part ENR (%)", "Prix EPEX moyen (€/MWh)",
                "CAPEX PV (k€/MW)", "CAPEX Éolien Offshore (k€/MW)", "CAPEX BESS (k€/MWh)",
            ])

        def add_line(fig, x, y, name, color, row, col, fill=False):
            fig.add_trace(go.Scatter(
                x=x, y=y, name=name, line=dict(color=color, width=2),
                fill="tozeroy" if fill else None,
                fillcolor=color.replace(")", ",0.12)").replace("rgb", "rgba") if fill else None,
            ), row=row, col=col)

        add_line(fig, df.index, df["co2_intensity_g_kwh"], "CO2", "#ef5350", 1, 1, True)
        add_line(fig, df.index, df["ren_share"]*100, "ENR", "#66bb6a", 1, 2, True)
        add_line(fig, df.index, df["epex_spot_mean"], "EPEX", "#42a5f5", 1, 3)
        add_line(fig, df.index, df["solar_capex"]/1000, "PV", "#ffa726", 2, 1)
        add_line(fig, df.index, df["wind_offshore_capex"]/1000, "Offshore", "#29b6f6", 2, 2)
        add_line(fig, df.index, df["bess_capex_per_mwh"]/1000, "BESS", "#ab47bc", 2, 3)

        fig.update_layout(template=PLOTLY_TEMPLATE, height=480,
                          showlegend=False, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Données brutes"):
            st.dataframe(df.style.format("{:.2f}"), use_container_width=True)

    # ── Tab 2 : Profils Solaires ─────────────────────────────────────────────
    with tabs[1]:
        st.markdown("#### Profils solaires horaires — PVGIS ou synthétique")
        solar_df = load_solar_db(db_dir)
        if solar_df is None:
            st.error("solar_patterns.db non trouvé")
            return

        sites = list(solar_df.columns)
        c1, c2 = st.columns([2, 1])
        with c1:
            selected_sites = st.multiselect("Sites à afficher", sites,
                                             default=sites[:3] if len(sites) >= 3 else sites)
        with c2:
            view_month = st.selectbox("Mois de référence", list(range(1, 13)),
                                       format_func=lambda m: MONTH_NAMES[m-1], index=5)

        if not selected_sites:
            st.info("Sélectionner au moins un site")
            return

        fig = make_subplots(rows=2, cols=2,
            subplot_titles=[
                f"Profil horaire — {MONTH_NAMES[view_month-1]} (1 semaine)",
                "CF mensuel moyen par site",
                "Heatmap CF — heure × mois",
                "Distribution annuelle des CF",
            ])

        colors_sites = px.colors.qualitative.Set2
        for i, site in enumerate(selected_sites[:6]):
            c = colors_sites[i % len(colors_sites)]
            s = solar_df[site]

            # Semaine type — moyenne par (jour_semaine, heure) sur tout le mois
            if not hasattr(s.index, 'month') or not hasattr(s.index, 'hour'):
                s.index = pd.date_range(start="2024-01-01", periods=len(s), freq="h")

            month_data = s[s.index.month == view_month].copy()

            if len(month_data) > 0:
                week_avg = (
                    month_data.groupby([month_data.index.dayofweek, month_data.index.hour])
                    .mean())
                week_avg.index.names = ["dow", "hour"]
                week_avg = week_avg.reset_index()
                week_avg.columns = ["dow", "hour", "cf"]
                week_avg = week_avg.sort_values(["dow", "hour"])
                DAYS = ["Lu", "Ma", "Me", "Je", "Ve", "Sa", "Di"]
                x_labels = [f"{DAYS[d]} {h:02d}h" for d, h in zip(week_avg["dow"], week_avg["hour"])]

                fig.add_trace(go.Scatter(
                    x=list(range(len(week_avg))),
                    y=week_avg["cf"].values * 100,
                    name=site,
                    line=dict(color=c, width=1.5),
                    fill="tozeroy" if i == 0 else None,
                    hovertext=x_labels,
                    hovertemplate="%{hovertext}: %{y:.1f}%<extra></extra>"
                ), row=1, col=1)

        # Heatmap premier site
        s0 = solar_df[selected_sites[0]]
        if not hasattr(s0.index, 'hour'):
            s0.index = pd.date_range(start="2024-01-01", periods=len(s0), freq="h")
        pivot = pd.DataFrame({"h": s0.index.hour, "m": s0.index.month, "v": s0.values})
        hm   = pivot.groupby(["h","m"])["v"].mean().unstack()
        fig.add_trace(go.Heatmap(z=hm.values*100, x=MONTH_NAMES,
            y=[f"{h}h" for h in range(24)], colorscale="YlOrRd",
            showscale=True, name=selected_sites[0]), row=2, col=1)

        # Distribution
        for i, site in enumerate(selected_sites[:6]):
            c = colors_sites[i % len(colors_sites)]
            vals = solar_df[site].values * 100
            vals = vals[vals > 0]
            fig.add_trace(go.Histogram(x=vals, name=site, marker_color=c,
                opacity=0.7, showlegend=False, nbinsx=40), row=2, col=2)

        fig.update_layout(template=PLOTLY_TEMPLATE, height=620,
                          legend=dict(orientation="h", y=1.02),
                          margin=dict(t=50, b=10))
        fig.update_yaxes(title_text="CF (%)", row=1, col=1)
        fig.update_xaxes(title_text="Heure du jour", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # Tableau récapitulatif CF
        st.markdown("**CF moyen annuel par site**")
        summary = pd.DataFrame({
            "Site": sites,
            "CF moyen (%)": [solar_df[s].mean()*100 for s in sites],
            "CF max (%)": [solar_df[s].max()*100 for s in sites],
            "Heures prod. > 10%": [(solar_df[s] > 0.10).sum() for s in sites],
        }).set_index("Site").round(2)
        st.dataframe(summary, use_container_width=True)

    # ── Tab 3 : Profils Éoliens ──────────────────────────────────────────────
    with tabs[2]:
        st.markdown("#### Profils éoliens horaires — ERA5 ou synthétique")
        wind_df = load_wind_db(db_dir)
        if wind_df is None:
            st.error("wind_patterns.db non trouvé")
            return

        zones = list(wind_df.columns)
        selected_zones = st.multiselect("Zones à afficher", zones,
                                         default=zones[:5] if len(zones) >= 5 else zones)
        if not selected_zones:
            return

        colors_z = px.colors.qualitative.Bold
        fig = make_subplots(rows=2, cols=2,
            subplot_titles=[
                "CF mensuel moyen par zone",
                "Distribution des CF (boxplot)",
                "Profil éolien — Semaine type Janvier",
                "Courbe de durée annuelle",
            ])

        for i, zone in enumerate(selected_zones[:8]):
            c = colors_z[i % len(colors_z)]
            s = wind_df[zone]

            monthly = s.groupby(s.index.month).mean() * 100
            fig.add_trace(go.Scatter(x=MONTH_NAMES, y=monthly.values,
                name=zone, line=dict(color=c), mode="lines+markers"), row=1, col=1)
            
            fig.add_trace(go.Box(y=s.values*100, name=zone, marker_color=c,
                showlegend=False), row=1, col=2)
            
            DAYS = ["Lu", "Ma", "Me", "Je", "Ve", "Sa", "Di"]
            jan_data = s[s.index.month == 1]
            if len(jan_data) > 0:
                week_avg_w = jan_data.groupby([jan_data.index.dayofweek, jan_data.index.hour]).mean()
                week_avg_w.index.names = ["dow", "hour"]
                week_avg_w = week_avg_w.reset_index().sort_values(["dow", "hour"])
                fig.add_trace(go.Scatter(
                    x=list(range(len(week_avg_w))),
                    y=week_avg_w.iloc[:, 2].values * 100,
                    name=zone, line=dict(color=c), showlegend=False,
                    hovertext=[f"{DAYS[d]} {h:02d}h" for d, h in zip(week_avg_w["dow"], week_avg_w["hour"])],
                    hovertemplate="%{hovertext}: %{y:.1f}%<extra></extra>"
                ), row=2, col=1)
            
            sorted_cf = np.sort(s.values)[::-1] * 100
            fig.add_trace(go.Scatter(x=np.arange(1, len(sorted_cf)+1), y=sorted_cf,
                name=zone, line=dict(color=c), showlegend=False), row=2, col=2)

        fig.update_layout(template=PLOTLY_TEMPLATE, height=600,
                          legend=dict(orientation="h", y=1.02))
        fig.update_yaxes(title_text="CF (%)", row=1, col=1)
        fig.update_xaxes(title_text="Heures (rang)", row=2, col=2)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Statistiques CF annuel**")
        stats = wind_df[selected_zones].describe().T[["mean","std","min","max"]]*100
        stats.columns = ["CF moy (%)", "Écart-type (%)", "CF min (%)", "CF max (%)"]
        st.dataframe(stats.round(2), use_container_width=True)

    # ── Tab 4 : Charge Industrielle ──────────────────────────────────────────
    with tabs[3]:
        st.markdown("#### Profils de charge industriels")
        load_df = load_load_db(db_dir)
        if load_df is None:
            st.error("load_patterns.db non trouvé")
            return

        load_cols = list(load_df.columns)
        selected_loads = st.multiselect("Profils à afficher", load_cols,
                                         default=load_cols[:4] if len(load_cols) >= 4 else load_cols)
        if not selected_loads:
            return

        colors_l = {"siderurgie": "#ef5350", "chimie": "#42a5f5",
                    "papier": "#66bb6a", "agroalim": "#ffa726", "load_reel": "#ce93d8"}

        fig = make_subplots(rows=2, cols=2,
            subplot_titles=[
                "Profil type 2 semaines",
                "CF mensuel moyen",
                "Distribution des charges",
                "Profil journalier moyen",
            ])

        for load_name in selected_loads:
            if load_name not in load_df.columns:
                continue
            c = colors_l.get(load_name, "#90a4ae")
            s = load_df[load_name].dropna()

            # 2 semaines
            two_w = s.iloc[:336]
            fig.add_trace(go.Scatter(x=list(range(len(two_w))), y=two_w.values*100,
                name=load_name, line=dict(color=c, width=1.5),
                fill="tozeroy" if load_name == selected_loads[0] else None), row=1, col=1)

            # CF mensuel
            if hasattr(s.index, "month"):
                monthly = s.groupby(s.index.month).mean() * 100
                fig.add_trace(go.Scatter(x=MONTH_NAMES, y=monthly.values,
                    name=load_name, line=dict(color=c), mode="lines+markers",
                    showlegend=False), row=1, col=2)

                # Distribution
                fig.add_trace(go.Histogram(x=s.values*100, name=load_name,
                    marker_color=c, opacity=0.7, showlegend=False, nbinsx=30), row=2, col=1)

                # Profil journalier
                hourly = s.groupby(s.index.hour).mean() * 100
                fig.add_trace(go.Scatter(x=list(range(24)), y=hourly.values,
                    name=load_name, line=dict(color=c), showlegend=False), row=2, col=2)

        fig.update_layout(template=PLOTLY_TEMPLATE, height=600,
                          legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Load Factor par profil**")
        lf_data = {lc: load_df[lc].mean() for lc in load_cols if lc in load_df.columns}
        lf_df = pd.DataFrame({"Profil": list(lf_data.keys()),
                               "Load Factor": [f"{v*100:.1f}%" for v in lf_data.values()]})
        st.dataframe(lf_df.set_index("Profil"), use_container_width=True)

    # ── Tab 5 : EPEX Spot ────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("#### Prix EPEX Day-Ahead France — profil horaire")
        year_sel = st.selectbox("Année", list(range(2018, 2026))[::-1], index=1)
        epex = load_epex_db(db_dir, year_sel)

        if epex is None:
            st.warning(f"epex_profiles.db absent ou année {year_sel} non calculée. "
                       "Exécuter `download_france_data.py`")
            return

        c1, c2, c3, c4 = st.columns(4)
        neg_h = int((epex < 0).sum())
        with c1: st.metric("Prix moyen", f"{epex.mean():.1f} €/MWh")
        with c2: st.metric("Prix max", f"{epex.max():.0f} €/MWh")
        with c3: st.metric("Prix min", f"{epex.min():.0f} €/MWh")
        with c4: st.metric("Heures négatives", f"{neg_h} h", f"{neg_h/len(epex)*100:.1f}%")

        fig = make_subplots(rows=2, cols=2,
            subplot_titles=[
                f"Prix spot EPEX {year_sel} (€/MWh)",
                "Prix mensuel moyen (€/MWh)",
                "Profil journalier moyen par saison",
                "Distribution des prix (histogramme)",
            ])

        fig.add_trace(go.Scatter(x=epex.index, y=epex.values,
            line=dict(color="#42a5f5", width=0.6), name="EPEX"), row=1, col=1)

        monthly = epex.groupby(epex.index.month).mean()
        colors_m = ["#1565c0" if m in (12,1,2,3) else
                    ("#43a047" if m in (6,7,8) else "#78909c")
                    for m in range(1,13)]
        fig.add_trace(go.Bar(x=MONTH_NAMES, y=monthly.values,
            marker_color=colors_m, showlegend=False), row=1, col=2)

        for season, months, color in [("Hiver", [12,1,2,3], "#1565c0"),
                                       ("Printemps", [4,5], "#8d6e63"),
                                       ("Été", [6,7,8], "#43a047"),
                                       ("Automne", [9,10,11], "#e65100")]:
            mask = epex.index.month.isin(months)
            hourly = epex[mask].groupby(epex[mask].index.hour).mean()
            fig.add_trace(go.Scatter(x=list(range(24)), y=hourly.values,
                name=season, line=dict(color=color)), row=2, col=1)

        fig.add_trace(go.Histogram(x=epex.values, nbinsx=80,
            marker_color="#42a5f5", showlegend=False), row=2, col=2)
        if neg_h > 0:
            fig.add_vline(x=0, line_dash="dash", line_color="#ef5350",
                          annotation_text="0 €/MWh", row=2, col=2)

        fig.update_layout(template=PLOTLY_TEMPLATE, height=580,
                          legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 6 : Capture Rates ────────────────────────────────────────────────
    with tabs[5]:
        st.markdown("#### Capture rates — valeur marché relative par site")
        st.markdown(
            '<div class="info-card">Le <b>capture rate systeme</b> mesure à quel point '
            "un parc vend son électricité aux heures où les prix sont élevés.<br>"
            "CR > 1 = parc éolien hivernal (produit quand les prix sont hauts) — "
            "CR < 1 = solaire (cannibalisation aux heures de midi).</div>",
            unsafe_allow_html=True
        )
        cr_df = load_capture_rates(db_dir)
        if cr_df is None:
            st.warning("capture_rates.db non trouvé. Exécuter `download_france_data.py`")
            return

        fig = make_subplots(rows=1, cols=2,
            subplot_titles=[
                "Capture rate par site (tri décroissant)",
                "Capture price vs EPEX moyen",
            ])

        cr_sorted = cr_df.sort_values("capture_rate_systeme", ascending=True)
        colors_cr = [COLORS_TECH.get(t, "#90a4ae") for t in cr_sorted["technology"]]

        fig.add_trace(go.Bar(
            y=cr_sorted["site"], x=cr_sorted["capture_rate_systeme"],
            orientation="h", marker_color=colors_cr,
            text=cr_sorted["capture_rate_systeme"].round(3),
            textposition="outside",
        ), row=1, col=1)
        fig.add_vline(x=1.0, line_dash="dash", line_color="#78909c",
                      annotation_text="CR=1 (neutre)", row=1, col=1)

        for tech, c in COLORS_TECH.items():
            sub = cr_df[cr_df["technology"] == tech]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["cf_mean"]*100, y=sub["capture_price_eur_mwh"],
                    mode="markers+text",
                    text=sub["site"].str[:12],
                    textposition="top center",
                    textfont=dict(size=9),
                    marker=dict(color=c, size=10, symbol="circle"),
                    name=tech,
                ), row=1, col=2)

        fig.update_xaxes(title_text="Capture Rate", row=1, col=1)
        fig.update_xaxes(title_text="CF moyen (%)", row=1, col=2)
        fig.update_yaxes(title_text="Capture Price (€/MWh)", row=1, col=2)
        fig.update_layout(template=PLOTLY_TEMPLATE, height=520,
                          legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(cr_df.sort_values("capture_rate_systeme", ascending=False)
                     .reset_index(drop=True), use_container_width=True)

    # ── Tab 7 : Parcs Éoliens ────────────────────────────────────────────────
    with tabs[6]:
        st.markdown("#### Parcs éoliens — France (offshore + onshore)")
        wg = load_wind_grid(db_dir)
        if wg is None:
            st.error("wind_grid_france.xlsx non trouvé")
            return

        col1, col2 = st.columns([3, 2])
        with col1:
            fig_map = px.scatter_mapbox(
                wg, lat="lat", lon="lon",
                size="capacity",
                color="LCOE",
                hover_name="nom_parc",
                hover_data={"capacity": True, "LCOE": True, "cf_expected": True,
                            "wind_type": True, "project_status": True},
                color_continuous_scale="RdYlGn_r",
                size_max=30,
                zoom=4.8,
                center={"lat": 48.5, "lon": -1.0},
                mapbox_style="open-street-map",
                title="Parcs éoliens — LCOE (€/MWh) et capacité",
            )
            fig_map.update_layout(template=PLOTLY_TEMPLATE, height=480,
                                   margin=dict(t=40, b=10))
            st.plotly_chart(fig_map, use_container_width=True)

        with col2:
            st.markdown("**Comparatif LCOE**")
            fig_lcoe = px.bar(
                wg.sort_values("LCOE"),
                x="nom_parc", y="LCOE",
                color="wind_type",
                text="capacity",
                color_discrete_map={"offshore": "#4fc3f7", "onshore": "#81c784",
                                     "offshore_floating": "#ce93d8"},
                template=PLOTLY_TEMPLATE,
                height=420,
            )
            fig_lcoe.update_traces(texttemplate="%{text} MW", textposition="outside")
            fig_lcoe.update_xaxes(tickangle=45)
            st.plotly_chart(fig_lcoe, use_container_width=True)

        st.dataframe(wg, use_container_width=True)

    # ── Tab 8 : Complémentarité ──────────────────────────────────────────────
    with tabs[7]:
        st.markdown("#### Complémentarité Solaire / Éolien / Charge")
        solar_df = load_solar_db(db_dir)
        wind_df  = load_wind_db(db_dir)
        load_df  = load_load_db(db_dir)

        if solar_df is None or wind_df is None:
            st.warning("Données solaires et/ou éoliennes manquantes")
            return

        c1, c2, c3 = st.columns(3)
        with c1:
            sol_site  = st.selectbox("Site solaire", list(solar_df.columns))
        with c2:
            wind_zone = st.selectbox("Zone éolienne", list(wind_df.columns))
        with c3:
            load_col  = st.selectbox("Profil charge", list(load_df.columns) if load_df is not None else ["—"])

        common = solar_df.index.intersection(wind_df.index)
        sol = solar_df[sol_site].reindex(common).fillna(0)
        wnd = wind_df[wind_zone].reindex(common).fillna(0)

        lod = None
        if load_df is not None and load_col in load_df.columns:
            lod_raw = load_df[load_col]
            if not hasattr(lod_raw.index, 'hour'):
                lod_raw.index = pd.date_range(start="2020-01-01", periods=len(lod_raw), freq="h")
            lod = lod_raw.reindex(common).fillna(lod_raw.mean())

        fig = make_subplots(rows=2, cols=2,
            subplot_titles=[
                "Profil journalier moyen — Hiver vs Été (Solaire / Éolien / Charge)",
                "",
                "CF mensuel moyen — comparaison sources",
                "Matrice de corrélation",
            ])

        # APRÈS — profil journalier moyen par saison (solaire + éolien + charge)
        SEASONS_COMP = [
            ("Hiver",     [12, 1, 2, 3], "solid"),
            ("Été",       [6, 7, 8],     "dot"),
        ]
        for season_name, months, dash in SEASONS_COMP:
            mask_s = common[common.month.isin(months)]
            if len(mask_s) == 0:
                continue
            for source, series, color, r, c_ in [
                ("Solaire", sol, COLORS_TECH["solar"],    1, 1),
                ("Éolien",  wnd, COLORS_TECH["offshore"], 1, 1),
            ] + ([("Charge", lod, COLORS_TECH["load"], 1, 1)] if lod is not None else []):
                hourly = series.reindex(mask_s).groupby(series.reindex(mask_s).index.hour).mean() * 100
                fig.add_trace(go.Scatter(
                    x=list(range(24)),
                    y=hourly.values,
                    name=f"{source} {'H' if season_name == 'Hiver' else 'E'}",
                    line=dict(color=color, dash=dash, width=1.8),
                    showlegend=True,
                    hovertemplate=f"%{{x}}h : %{{y:.1f}}%<extra>{source} {season_name}</extra>"
                ), row=1, col=1)

        # CF mensuel
        for name, s, c_ in [("Solaire", sol, COLORS_TECH["solar"]),
                              ("Éolien",  wnd, COLORS_TECH["offshore"])]:
            monthly = s.groupby(s.index.month).mean() * 100
            fig.add_trace(go.Scatter(x=MONTH_NAMES, y=monthly.values,
                name=name, line=dict(color=c_), showlegend=False), row=2, col=1)
        if lod is not None:
            monthly_l = lod.groupby(lod.index.month).mean() * 100
            fig.add_trace(go.Scatter(x=MONTH_NAMES, y=monthly_l.values,
                name="Charge", line=dict(color=COLORS_TECH["load"], dash="dot"),
                showlegend=False), row=2, col=1)

        # Matrice de corrélation
        data_dict = {"Solaire": sol.values, "Éolien": wnd.values}
        if lod is not None:
            data_dict["Charge"] = lod.values
        corr_mx = pd.DataFrame(data_dict).corr()
        fig.add_trace(go.Heatmap(
            z=corr_mx.values, x=corr_mx.columns.tolist(), y=corr_mx.index.tolist(),
            colorscale="RdBu", zmid=0, showscale=True,
            text=corr_mx.round(3).values,
            texttemplate="%{text}",
        ), row=2, col=2)

        fig.update_layout(template=PLOTLY_TEMPLATE, height=600,
                          legend=dict(orientation="h", y=1.04))
        st.plotly_chart(fig, use_container_width=True)

        # Heures négatives
        epex = load_epex_db(db_dir)
        if epex is not None:
            common2 = sol.index.intersection(epex.index)
            sol2 = sol.reindex(common2).values
            ep   = epex.reindex(common2).values
            neg_sol_hrs = int(((ep < 0) & (sol2 > 0.01)).sum())
            wnd2 = wnd.reindex(common2).values
            neg_wnd_hrs = int(((ep < 0) & (wnd2 > 0.01)).sum())
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Heures production solaire + EPEX négatif",
                           f"{neg_sol_hrs} h/an",
                           help="Heures où le solaire produit pendant que les prix sont négatifs")
            with c2:
                st.metric("Heures production éolienne + EPEX négatif",
                           f"{neg_wnd_hrs} h/an")

    # ── Tab 9 : Qualité des données ──────────────────────────────────────────
    with tabs[8]:
        st.markdown("#### Tableau de qualité des données")
        status_map, labels = _db_status(db_dir)

        rows = [
            ("solar_patterns.db",    "🟢", "PVGIS JRC (ou synthétique)", "Basse", "pvgis.ec.europa.eu"),
            ("wind_patterns.db",     "🟢", "ERA5 Open-Meteo (ou synthétique)", "Basse", "archive-api.open-meteo.com"),
            ("epex_profiles.db",     "🟡", "Synthétique calibré / CSV ODRE", "Haute", "odre.opendatasoft.com"),
            ("grid_france.csv",      "🟡", "Projections ADEME/RTE + eco2mix", "Haute", "odre.opendatasoft.com"),
            ("load_patterns.db",     "🟡", "Profils industriels synthétiques", "Critique", "Client (courbe de charge)"),
            ("TURPE_france.xlsx",    "🟢", "CRE TURPE 6 HTB Nov 2024", "Basse", "cre.fr"),
            ("wind_grid_france.xlsx","🟡", "12 parcs encodés manuellement", "Haute", "Appels d'offres CRE"),
            ("capture_rates.db",     "🟢", "Calculé depuis EPEX × production", "Basse", "Auto-généré"),
        ]

        quality_df = pd.DataFrame(rows, columns=[
            "Fichier", "Statut", "Source actuelle", "Priorité amélioration", "Source recommandée"
        ])
        quality_df["Présent"] = quality_df["Fichier"].map(
            lambda f: "✅" if status_map.get(f, False) else "❌"
        )

        st.dataframe(
            quality_df.set_index("Fichier"),
            use_container_width=True,
            height=320,
        )

        st.markdown("---")
        st.markdown("**Actions prioritaires**")
        actions = [
            ("🔴 Critique", "Collecter la courbe de charge réelle du client",
             "Export ENEDIS depuis espace client ou contrat gestionnaire d'énergie"),
            ("🟠 Haute", "Intégrer le CSV EPEX ODRE réel",
             "`python download_france_data.py --epex epex_2024.csv`"),
            ("🟠 Haute", "Recalculer les capture rates avec EPEX réel",
             "Relancer `download_france_data.py` après intégration EPEX"),
            ("🟡 Moyenne", "Compléter wind_grid_france.xlsx",
             "Ajouter les parcs des AO CRE 2021-2024 (Saint-Nazaire 2, Courseulles...)"),
        ]
        for badge, title, detail in actions:
            color = "badge-red" if "Rouge" in badge or "Critique" in badge else \
                    "badge-orange" if "Haute" in badge else "badge-orange"
            st.markdown(
                f'<div class="info-card">'
                f'<span class="badge-{"red" if "Critique" in badge else "orange"}">{badge}</span> '
                f'<b style="margin-left:8px">{title}</b><br>'
                f'<small style="color:#7a9bb5">{detail}</small></div>',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE : SCÉNARIO & CALCUL
# ─────────────────────────────────────────────────────────────────────────────

def page_scenario(db_dir: str) -> None:
    st.markdown("# ⚙️ Configuration du Scénario")

    xlsx_path = _DB_DIR / "scenario_defaults.xlsx"
    sd = load_scenario_defaults(str(xlsx_path))

    if sd is None:
        st.error(f"scenario_defaults.xlsx non trouvé dans `{_DB_DIR}`")
        st.code("Chemin attendu : " + str(xlsx_path))
        return

    scen_cols = [c for c in sd.columns
                 if c not in ("Scenario Name",) and not c.startswith("Unnamed")]

    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("### Sélection du scénario")
        selected_scenario = st.selectbox("Scénario de base", scen_cols)

        st.markdown("---")
        st.markdown("### Ajustements manuels")

        def get_val(param, default=None):
            row = sd[sd["Scenario Name"] == param]
            if row.empty or selected_scenario not in sd.columns:
                return default
            v = row[selected_scenario].values[0]
            return v if pd.notna(v) else default

        load_mw     = st.number_input("Charge client (MW)",
                                       value=float(get_val("Load ArcelorMittal Dunkerque (MW)", 200)),
                                       min_value=10.0, max_value=2000.0, step=10.0)
        model_year  = st.slider("Année de modélisation", 2025, 2040,
                                 int(get_val("Model Year", 2030)))
        duration    = st.slider("Durée du contrat (ans)", 5, 25,
                                 15)
        wacc        = st.slider("WACC (%)", 3.0, 12.0,
                                 float(get_val("Discount Rate / WACC (%)", 0.07)) * 100,
                                 step=0.5) / 100

        st.markdown("---")
        st.markdown("### Structure contractuelle")
        price_structure = st.selectbox("Structure de prix",
            ["fixed", "fixed_escalating", "collar", "indexed_spot", "floor_only"],
            help="fixed = prix fixe (vPPA CfD) | collar = tunnel floor/cap | indexed = décote marché")

        ppa_type = st.selectbox("Type PPA",
            ["offsite_sleeved", "offsite_direct", "onsite"],
            format_func=lambda x: {
                "offsite_sleeved": "Off-site + sleeving (standard)",
                "offsite_direct": "Off-site direct (>50 MW)",
                "onsite": "On-site (exemption TURPE)"
            }[x])

        tech = st.selectbox("Technologie",
            ["solar", "offshore", "onshore", "hybrid"])

        sleeving = st.slider("Sleeving fee (€/MWh)", 0.0, 10.0,
                              3.0, 0.5)
        go_price = st.slider("Prix GO initial (€/MWh)", 0.0, 25.0,
                              float(get_val("Initial GO Price (EUR/MWh)", 8.0)), 0.5)

    with col2:
        st.markdown("### Aperçu du scénario")

        # Afficher les paramètres clés du scénario sélectionné
        key_params = [
            "Load ArcelorMittal Dunkerque (MW)", "Load H2 Normandie (MW)",
            "Model Year", "Analysis Target Year", "Initial Year",
            "Initial SMP / Prix EPEX (EUR/MWh)", "SMP Annual Increase (%)",
            "Selected Sheet (TURPE domain)",
            "Initial GO Price (EUR/MWh)", "GO Reduction over time",
            "Discount Rate / WACC (%)", "Include Battery Storage",
            "Battery CAPEX per MW (EUR)", "Client Latitude", "Client Longitude",
        ]
        param_df = sd[sd["Scenario Name"].isin(key_params)][
            ["Scenario Name", selected_scenario]
        ].rename(columns={"Scenario Name": "Paramètre", selected_scenario: "Valeur"})
        param_df = param_df.set_index("Paramètre")
        st.dataframe(param_df, use_container_width=True, height=360)

        st.markdown("---")

        # Paramètres avancés AXE 4 et 7
        with st.expander("⚙️ Clauses avancées (AXE 4 — Curtailment / Prix négatifs)"):
            c_a, c_b = st.columns(2)
            with c_a:
                neg_floor = st.number_input("Plancher prix négatifs (€/MWh)",
                                             value=0.0, min_value=-100.0, max_value=10.0,
                                             help="0 = zero floor (standard 2024+)")
                grid_curt = st.slider("Curtailment réseau TSO (%/an)", 0.0, 5.0, 1.5, 0.5)
            with c_b:
                vol_curtail_thr = st.number_input("Seuil curtailment volontaire (€/MWh)",
                                                   value=0.0,
                                                   help="Laisser 0 pour désactiver")
                basis_risk = st.slider("Basis risk (€/MWh)", 0.0, 5.0, 0.5, 0.25)

        with st.expander("⚙️ Clauses avancées (AXE 7 — Risques financiers)"):
            c_a, c_b = st.columns(2)
            with c_a:
                irr_target = st.slider("IRR cible (%)", 4.0, 15.0, 8.0, 0.5)
                hedge_acctg = st.checkbox("Hedge Accounting (IFRS 9)", value=False)
            with c_b:
                credit_support = st.selectbox("Credit support",
                    ["none", "letter_of_credit", "parent_guarantee", "cash_collateral"])
                go_matching = st.selectbox("Granularité GO",
                    ["annual", "monthly", "hourly"],
                    help="Granularité du matching des GOs (hourly = 24/7 CFE premium)")

        st.markdown("---")

        # Bouton de lancement
        run_ready = sum([
            (Path(db_dir) / "solar_patterns.db").exists(),
            (Path(db_dir) / "wind_patterns.db").exists(),
            (Path(db_dir) / "epex_profiles.db").exists(),
        ]) >= 2

        if not run_ready:
            st.warning("⚠️ Données insuffisantes — exécuter d'abord `download_france_data.py --offline`")

        col_btn, col_info = st.columns([1, 2])
        with col_btn:
            run_clicked = st.button("▶ Lancer le calcul", disabled=not run_ready,
                                     use_container_width=True)

        if run_clicked:
            params_override = {
                "scenario_name":    selected_scenario,
                "client_load_mw":   load_mw,
                "model_year":       model_year,
                "contract_duration": duration,
                "wacc":             wacc,
                "price_structure":  price_structure,
                "ppa_type":         ppa_type,
                "ppa_technology":   tech,
                "sleeving_fee":     sleeving,
                "go_price_init":    go_price,
                "neg_price_floor":  neg_floor if neg_floor != 0 else None,
                "grid_curtailment": grid_curt / 100,
                "vol_curtail_thr":  vol_curtail_thr if vol_curtail_thr != 0 else None,
                "basis_risk":       basis_risk,
                "irr_target":       irr_target / 100,
                "credit_support":   credit_support,
                "go_matching":      go_matching,
                "hedge_accounting": hedge_acctg,
                "initial_year":     2024,
                "reference_year":   2020,
                "epex_increase":    0.02,
                "initial_epex":     float(get_val("Initial SMP / Prix EPEX (EUR/MWh)", 65)),
                "turpe_level":      str(get_val("Selected Sheet (TURPE domain)", "HTB2")),
                "go_included":      True,
                "go_reduction":     True,
                "go_increase":      -0.03,
                "carbon_price_init": 65.0,
                "carbon_increase":   0.05,
                "use_ngfs_carbon":   False,
                "rate_increase":     float(get_val("Rate Increase (% per year)", 0.02)),
                "ppa_payment_type":  "levelised",
                "include_battery":   False,
                "solar_cap_max_mw":  200.0,
                "wind_cap_max_mw":   400.0,
                "wind_unit_mw":      15.0,
                "bin_size_mw":       float(get_val("Bin Size (MW)", 5)),
                "min_capacity_mw":   float(get_val("Minimum Capacity (MW)", 10)),
                "max_grid_share":    0.50,
                "output_path":       "output/",
                "price_escalation_rate": 0.015 if price_structure == "fixed_escalating" else 0.0,
                "price_escalation_index": "none",
                "price_floor":       None,
                "price_cap":         None,
                "spot_discount":     5.0,
                "upside_sharing":    0.0,
                "neg_buyer_cap_h":   200,
                "neg_sharing_pct":   0.5,
                "grid_curtail_comp": 0.0,
                "curtail_comp_pct":  0.0,
                "vol_guarantee":     "pay_as_produced",
                "vol_tolerance":     0.10,
                "shortfall_penalty": 0.0,
                "imbalance_resp":    "sleeving",
                "settlement_period": "monthly",
                "contract_duration": duration,
                "start_date_type":   "cod",
                "early_term_fee":    0.15,
                "change_of_law":     "yes_shared",
                "renewal_option":    "none",
                "step_down_pct":     0.0,
                "go_price_type":     "fixed",
                "go_bundled":        True,
                "additionality":     True,
                "cfe_247_target":    False,
                "shape_premium":     0.0,
                "hedge_accounting":  hedge_acctg,
                "termination_method":"replacement_cost",
                "mtm_frequency":     "none",
                "batt_capex_mw":     300_000.0,
                "batt_capex_mwh":    150_000.0,
                "batt_eta":          0.92,
                "batt_rt_eta":       0.92,
                "batt_hours":        4.0,
                "batt_life_years":   15,
                "epex_cap":          999.0,
                "client_lat":        float(get_val("Client Latitude", 50.93)),
                "client_lon":        float(get_val("Client Longitude", 2.38)),
            }

            st.session_state["run_params"] = params_override
            st.session_state["db_dir"]     = db_dir

            with st.spinner("Calcul en cours..."):
                try:
                    sys.path.insert(0, str(_HERE))
                    from ppamodule import (
                        build_temporal_profile, run_site_screening,
                        optimize_ppa_mix, run_npv_cashflow, compute_ppa_settlement,
                        load_scenario
                    )
                    from FranceGridUtils import get_epex_series, load_production_profile

                    p = params_override
                    temporal_df, contract_fee = build_temporal_profile(p)
                    screened = run_site_screening(p, load_col="siderurgie",
                                                   db_dir=Path(db_dir))
                    mix      = optimize_ppa_mix(p, screened, temporal_df, Path(db_dir))
                    cashflow = run_npv_cashflow(p, mix, temporal_df, Path(db_dir))

                    st.session_state["results"] = {
                        "params": p, "temporal_df": temporal_df,
                        "contract_fee": contract_fee, "screened": screened,
                        "mix": mix, "cashflow": cashflow,
                    }
                    st.success("✅ Calcul terminé — consulter la page **📈 Résultats**")
                except ImportError as e:
                    st.error(f"Module non trouvé : {e}")
                    st.info("Vérifier que `ppamodule.py` et `FranceGridUtils.py` sont dans le même dossier.")
                except Exception as e:
                    st.error(f"Erreur calcul : {e}")
                    import traceback
                    st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# PAGE : RÉSULTATS
# ─────────────────────────────────────────────────────────────────────────────

def page_resultats() -> None:
    st.markdown("# 📈 Résultats")

    if "results" not in st.session_state:
        st.info("Aucun résultat disponible — lancer un calcul depuis **⚙️ Scénario & Calcul**")
        return

    res = st.session_state["results"]
    p       = res["params"]
    mix     = res["mix"]
    cf_df   = res["cashflow"]
    screened = res["screened"]

    # ── KPIs principaux ───────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    npv = cf_df.attrs.get("npv_eur", 0) if not cf_df.empty else 0
    with c1: st.metric("VAN économies", f"{npv/1e6:.1f} M€")
    with c2: st.metric("MW PPA optimal", f"{mix.get('total_ppa_mw',0):.0f} MW")
    with c3: st.metric("Couverture charge", f"{mix.get('coverage_pct',0):.1f}%")
    with c4: st.metric("Coût net", f"{mix.get('net_cost_eur_mwh',0):.1f} €/MWh")
    with c5:
        pb = cf_df.attrs.get("payback_years", "—") if not cf_df.empty else "—"
        st.metric("Payback", f"{pb} ans")

    st.markdown("---")

    tabs = st.tabs(["💰 Cashflow & VAN", "🏆 Mix PPA", "📊 Sites screenés", "⚙️ Paramètres"])

    # ── Tab Cashflow ──────────────────────────────────────────────────────────
    with tabs[0]:
        if cf_df.empty:
            st.warning("Cashflow non calculé")
            return

        fig = make_subplots(rows=2, cols=2,
            subplot_titles=[
                "Économies annuelles (€/an)",
                "VAN cumulée (€)",
                "Prix PPA vs EPEX projeté (€/MWh)",
                "CO2 évité (tCO2/an)",
            ])

        fig.add_trace(go.Bar(x=cf_df.index, y=cf_df["savings_eur_yr"]/1e6,
            name="Économies (M€/an)", marker_color="#4fc3f7"), row=1, col=1)
        fig.add_trace(go.Scatter(x=cf_df.index,
            y=cf_df["savings_pv_eur"].cumsum()/1e6,
            name="VAN cumulée (M€)", line=dict(color="#81c784", width=2.5),
            fill="tozeroy"), row=1, col=2)
        fig.add_vline(x=2024 + (cf_df.attrs.get("payback_years", 10) or 10),
                      line_dash="dash", line_color="#ffa726",
                      annotation_text="Payback", row=1, col=2)
        fig.add_trace(go.Scatter(x=cf_df.index, y=cf_df["ppa_price_eur_mwh"],
            name="Prix PPA", line=dict(color="#ffa726")), row=2, col=1)
        fig.add_trace(go.Scatter(x=cf_df.index, y=cf_df["epex_projected_eur_mwh"],
            name="EPEX projeté", line=dict(color="#ef5350", dash="dot")), row=2, col=1)
        fig.add_trace(go.Bar(x=cf_df.index, y=cf_df["co2_avoided_t"]/1000,
            name="CO2 évité (kt)", marker_color="#a5d6a7"), row=2, col=2)

        fig.update_layout(template=PLOTLY_TEMPLATE, height=560,
                          legend=dict(orientation="h", y=1.04))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Tableau cashflow détaillé"):
            st.dataframe(cf_df.reset_index(), use_container_width=True)
            csv = cf_df.reset_index().to_csv(index=False)
            st.download_button("⬇ Télécharger CSV", csv,
                                file_name=f"cashflow_{p['scenario_name']}.csv",
                                mime="text/csv")

    # ── Tab Mix PPA ───────────────────────────────────────────────────────────
    with tabs[1]:
        mix_mw = mix.get("mix_mw", {})
        prices = mix.get("ppa_price", {})
        if not mix_mw:
            st.warning("Aucun mix PPA calculé")
            return

        c1, c2 = st.columns(2)
        with c1:
            fig_bar = px.bar(
                x=list(mix_mw.keys()),
                y=list(mix_mw.values()),
                color=list(mix_mw.keys()),
                labels={"x": "Site", "y": "MW", "color": "Site"},
                title="Capacité allouée par site (MW)",
                template=PLOTLY_TEMPLATE,
                height=350,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with c2:
            fig_pie = px.pie(
                names=list(mix_mw.keys()),
                values=list(mix_mw.values()),
                title="Répartition du mix PPA",
                template=PLOTLY_TEMPLATE,
                height=350,
                hole=0.4,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        mix_summary = pd.DataFrame({
            "Site": list(mix_mw.keys()),
            "Capacité (MW)": list(mix_mw.values()),
            "Prix PPA (€/MWh)": [prices.get(s, 0) for s in mix_mw],
        })
        mix_summary["Coût annuel estimé (M€)"] = (
            mix_summary["Capacité (MW)"] *
            mix_summary["Prix PPA (€/MWh)"] *
            8760 * 0.20 / 1e6  # hypothèse CF 20%
        ).round(2)
        st.dataframe(mix_summary.set_index("Site"), use_container_width=True)

    # ── Tab Sites ─────────────────────────────────────────────────────────────
    with tabs[2]:
        if screened is None or screened.empty:
            st.warning("Aucun site screené")
            return

        fig = make_subplots(rows=1, cols=2,
            subplot_titles=["Score composite par site", "Capture Rate vs Corrélation charge"])

        sorted_s = screened.sort_values("score_composite")
        colors_t = [COLORS_TECH.get(t, "#90a4ae") for t in sorted_s["technology"]]
        fig.add_trace(go.Bar(y=sorted_s["site"], x=sorted_s["score_composite"],
            orientation="h", marker_color=colors_t,
            text=sorted_s["score_composite"].round(3), textposition="outside"), row=1, col=1)

        for tech, c in COLORS_TECH.items():
            sub = screened[screened["technology"] == tech]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["capture_rate_systeme"],
                    y=sub["correlation_load"].fillna(0),
                    mode="markers+text",
                    text=sub["site"].str[:12],
                    textposition="top center",
                    textfont=dict(size=9),
                    marker=dict(color=c, size=12),
                    name=tech,
                ), row=1, col=2)

        fig.add_vline(x=1.0, line_dash="dash", line_color="#78909c", row=1, col=2)
        fig.update_xaxes(title_text="Score composite", row=1, col=1)
        fig.update_xaxes(title_text="Capture Rate", row=1, col=2)
        fig.update_yaxes(title_text="Corrélation charge", row=1, col=2)
        fig.update_layout(template=PLOTLY_TEMPLATE, height=430,
                          legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(screened, use_container_width=True)

    # ── Tab Paramètres ────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("**Paramètres utilisés pour ce calcul**")
        param_df = pd.DataFrame({"Valeur": p}).reset_index()
        param_df.columns = ["Paramètre", "Valeur"]
        st.dataframe(param_df.set_index("Paramètre"), use_container_width=True, height=600)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE : CARTE GIS
# ─────────────────────────────────────────────────────────────────────────────

def page_carte_gis(db_dir: str) -> None:
    st.markdown("# 🗺️ Carte GIS — Parcelles candidates")

    gis_csv = Path(db_dir).parent / "gisdata" / "db_semippa_auto.csv"
    gis_gpkg = Path(db_dir).parent / "gisdata" / "db_semippa_auto.gpkg"
    html_map = Path(db_dir).parent / "gisdata" / "db_semippa_auto_map.html"

    c1, c2 = st.columns([3, 1])
    with c2:
        st.markdown("### Générer les données GIS")
        lat = st.number_input("Latitude site", value=50.93, format="%.4f")
        lon = st.number_input("Longitude site", value=2.38, format="%.4f")
        buf = st.slider("Rayon (km)", 5, 80, 30)
        synthetic = st.checkbox("Mode synthétique (sans internet)", value=True)

        if st.button("🔄 Construire la grille", use_container_width=True):
            with st.spinner("Appel IGN RPG + DVF + INPN..."):
                try:
                    sys.path.insert(0, str(_HERE))
                    from build_land_grid import build_land_grid
                    gis_dir = str(Path(db_dir).parent / "gisdata")
                    build_land_grid(lat, lon, buffer_km=buf,
                                    output_dir=gis_dir,
                                    synthetic_mode=synthetic)
                    st.success("✅ Grille construite")
                    st.rerun()
                except ImportError:
                    st.error("build_land_grid.py non trouvé")
                except Exception as e:
                    st.error(f"Erreur : {e}")

    with c1:
        if html_map.exists():
            st.markdown("**Carte interactive des parcelles candidates**")
            st.markdown(
                '<div class="info-card">La carte est disponible en plein écran — '
                f'<a href="{html_map}" target="_blank" style="color:#4fc3f7">ouvrir dans le navigateur ↗</a></div>',
                unsafe_allow_html=True
            )
            with open(str(html_map), "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=560, scrolling=False)
        else:
            st.info("Carte non disponible — cliquer sur **Construire la grille** →")

    # Tableau des parcelles
    if gis_csv.exists():
        st.markdown("---")
        gdf = pd.read_csv(gis_csv)
        st.markdown(f"### {len(gdf)} parcelles candidates")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Surface totale", f"{gdf['area_m2'].sum()/1e4:,.0f} ha")
        with c2: st.metric("Puissance PV max", f"{gdf['area_photo'].sum()*80/1e6:,.0f} MW")
        with c3: st.metric("Distance moy.", f"{gdf['distance_0'].mean()/1000:.1f} km")
        with c4: st.metric("Prix foncier médian", f"{gdf['wavgprice'].median():.2f} €/m²")

        # Graphique par classe
        if "clc_label" in gdf.columns:
            by_class = gdf.groupby("clc_label")["area_m2"].sum().sort_values(ascending=False)
            fig = px.bar(x=by_class.index, y=by_class.values/1e4,
                labels={"x": "Classe CLC", "y": "Surface (ha)"},
                title="Répartition par type de terrain (ha)",
                template=PLOTLY_TEMPLATE, height=320)
            fig.update_xaxes(tickangle=30)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Données tabulaires complètes"):
            st.dataframe(gdf, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # État initial
    if "db_dir" not in st.session_state:
        st.session_state["db_dir"] = str(_DB_DIR)

    page, db_dir = render_sidebar(st.session_state["db_dir"])
    st.session_state["db_dir"] = db_dir

    if page == "🏠 Accueil":
        page_accueil(db_dir)
    elif page == "📊 Data Explorer":
        page_data_explorer(db_dir)
    elif page == "⚙️ Scénario & Calcul":
        page_scenario(db_dir)
    elif page == "📈 Résultats":
        page_resultats()
    elif page == "🗺️ Carte GIS":
        page_carte_gis(db_dir)


if __name__ == "__main__":
    main()
