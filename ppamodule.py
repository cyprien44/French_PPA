"""
ppamodule.py — French PPA Model v3.0
======================================
Moteur de calcul PPA pour industriels français raccordés HTB.

Ce module reproduit la méthodologie réelle des analystes PPA France 2024-2026 :
  - Settlement CfD horaire sur profil EPEX réel (pas annualisé)
  - LCOE via annuité CAPEX + OPEX (méthode IRENA) par technologie et millésime
  - Optimisation LP scipy : mix MW multi-sites sous contrainte couverture charge
  - Cashflow 20 ans : TURPE escaladé, GOs, dégradation technique, EU ETS
  - 6 structures contractuelles (fixed, collar, floor_only, indexed_spot...)
  - Fin ARENH : CPP EDF comme alternative de référence (jan. 2026)

Différences fondamentales vs modèle coréen :
  | Corée                    | France v3.0                         |
  |--------------------------|-------------------------------------|
  | PyPSA (réseau complet)   | scipy LP (pas de dépendance lourde) |
  | KEPCO (monopole, zones)  | TURPE HTB (péréquation nationale)   |
  | SMP coréen               | EPEX Spot Day-Ahead France          |
  | RECs (~57 euros/MWh)     | GOs (~5-15 euros/MWh selon granul.) |
  | KRW/USD                  | EUR natif (currency_exchange = 1.0) |
  | CO2 ~450 gCO2/kWh        | CO2 ~44 gCO2/kWh (nucléaire 70%)   |
  | Settlement mensuel KEPCO | Settlement CfD financier mensuel    |

Dépendances : pandas, numpy, scipy — standard, pas de PyPSA requis.
"""

from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CHEMINS & CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

_HERE   = Path(__file__).resolve().parent
_DB_DIR = _HERE / "database"

# CAPEX de référence France 2025 (euros/MW) — IRENA 2024 + Aurora Energy Research
_CAPEX_2025 = {
    "solar":    680_000,
    "onshore":  1_350_000,
    "offshore": 2_800_000,
    "hybrid":   900_000,
}

# OPEX en fraction du CAPEX (source : IRENA Renewable Power Generation Costs 2023)
_OPEX_RATE = {
    "solar": 0.018, "onshore": 0.025, "offshore": 0.030, "hybrid": 0.022,
}

# Durée de vie technique (ans)
_LIFETIME = {
    "solar": 30, "onshore": 25, "offshore": 25, "hybrid": 25,
}

# Dégradation technique annuelle (perte de production, source IEA PVPS / Windeurope)
_DEGRADATION = {
    "solar": 0.005, "onshore": 0.004, "offshore": 0.003, "hybrid": 0.004,
}

# CO2 réseau France 2024 (gCO2/kWh) — eco2mix RTE
_CO2_GRID_2024 = 44.0

# Multiplicateur GO selon granularité de matching (marché EEX 2024)
_GO_GRANULARITY_MULT = {"annual": 1.0, "monthly": 1.4, "hourly": 2.2}

# CPP EDF (successeur ARENH depuis jan. 2026) — estimation consensus Q1 2026
_CPP_EDF_EUR_MWH = 67.0


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS SQLITE
# ─────────────────────────────────────────────────────────────────────────────

def _sql_series(db: Path, table: str, idx: str, col: str) -> pd.Series | None:
    if not db.exists():
        return None
    conn = sqlite3.connect(str(db))
    try:
        df = pd.read_sql(f"SELECT {idx},{col} FROM {table}", conn,
                         index_col=idx, parse_dates=[idx])
        conn.close()
        s = df[col]
        s.index = pd.to_datetime(s.index).floor("h")
        return s[~s.index.duplicated(keep="first")]
    except Exception:
        conn.close()
        return None


def _load_epex(db_dir: Path, year: int) -> pd.Series | None:
    db = db_dir / "epex_profiles.db"
    if not db.exists():
        return None
    conn = sqlite3.connect(str(db))
    try:
        df = pd.read_sql(f"SELECT * FROM epex_{year}", conn,
                         index_col="datetime", parse_dates=["datetime"])
        conn.close()
        return df["price_eur_mwh"]
    except Exception:
        conn.close()
        return None


def _load_grid(db_dir: Path) -> pd.DataFrame | None:
    p = db_dir / "grid_france.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(str(p), index_col="year")
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD SCENARIO
# ─────────────────────────────────────────────────────────────────────────────

def load_scenario(xlsx_path: str, scenario_col: str) -> dict:
    """
    Charge un scénario depuis scenario_defaults.xlsx.
    Retourne un dict de paramètres normalisés (unités SI).
    """
    p = Path(xlsx_path)
    if not p.exists():
        raise FileNotFoundError(f"scenario_defaults.xlsx non trouve : {xlsx_path}")

    df = pd.read_excel(str(p), header=3)
    df.columns = [str(c).strip() for c in df.columns]

    def _get(param, default=None):
        row = df[df["Scenario Name"] == param]
        if row.empty or scenario_col not in df.columns:
            return default
        v = row[scenario_col].values[0]
        return v if pd.notna(v) else default

    def _f(param, default):
        v = _get(param, default)
        try:
            return float(v) if v is not None else float(default)
        except (TypeError, ValueError):
            return float(default)

    def _i(param, default):
        return int(_f(param, float(default)))

    def _b(param, default=False):
        v = _get(param, default)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "oui", "1")
        try:
            return bool(float(v))
        except Exception:
            return bool(default)

    params = {
        "scenario_name":         scenario_col,
        # Client
        "client_load_mw":        _f("Load ArcelorMittal Dunkerque (MW)", 200.0),
        "client_lat":            _f("Client Latitude", 50.93),
        "client_lon":            _f("Client Longitude", 2.38),
        "turpe_level":           str(_get("TURPE Level (HTB1/HTB2/HTB3)", "HTB2")),
        # Temporel
        "initial_year":          _i("Initial Year", 2024),
        "model_year":            _i("Model Year", 2030),
        "reference_year":        _i("Reference Year (EPEX + meteo)", 2020),
        "contract_duration":     _i("Contract Duration (years)", 15),
        # Prix marche
        "initial_epex":          _f("Initial SMP / Prix EPEX (EUR/MWh)", 65.0),
        "epex_increase":         _f("SMP Annual Increase (%)", 2.0),
        "rate_increase":         _f("Rate Increase (% per year)", 2.0),
        "max_grid_share":        _f("Max Grid Share (%)", 50.0),
        # GOs
        "go_price_init":         _f("Initial GO Price (EUR/MWh)", 8.0),
        "go_reduction":          _b("GO Reduction over time", False),
        "go_increase":           _f("GO Price Annual Increase (%)", -2.0),
        "go_included":           _b("GO Include in PPA Fees", True),
        "go_matching":           str(_get("GO Matching Granularity", "annual")),
        # Carbone EU ETS
        "carbon_price_init":     _f("Custom Carbon Price Initial (EUR/tCO2)", 65.0),
        "carbon_increase":       _f("Carbon Price Annual Increase (%)", 5.0),
        "use_ngfs_carbon":       _b("Use NGFS Carbon Price Scenario", False),
        # Contrat PPA
        "price_structure":       str(_get("Price Structure", "fixed")),
        "ppa_type":              str(_get("PPA Type", "offsite_sleeved")),
        "ppa_technology":        str(_get("PPA Technology", "solar")),
        "ppa_payment_type":      str(_get("PPA Payment Type", "levelised")),
        "price_escalation_rate": _f("Price Escalation Rate (% /year)", 0.0),
        "price_floor":           _get("Price Floor (euros/MWh)", None),
        "price_cap":             _get("Price Cap (euros/MWh)", None),
        "spot_discount":         _f("Spot Discount (euros/MWh)", 5.0),
        "collar_band":           0.15,
        # Clauses operationnelles
        "neg_price_floor":       _f("Negative Price Floor (euros/MWh)", 0.0),
        "neg_buyer_cap_h":       _i("Negative Price Buyer Cap (h/year)", 200),
        "neg_sharing_pct":       _f("Negative Price Sharing (%)", 50.0),
        "grid_curtailment":      _f("Grid Curtailment Rate (% annual)", 1.5),
        "vol_curtail_thr":       _get("Voluntary Curtailment Threshold (euros/MWh)", None),
        "basis_risk":            _f("Basis Risk (euros/MWh)", 0.5),
        "sleeving_fee":          _f("Sleeving Fee (euros/MWh)", 3.0),
        # Finance
        "wacc":                  _f("Discount Rate / WACC (%)", 7.0),
        "irr_target":            _f("Internal Rate of Return (%)", 8.0),
        "credit_support":        str(_get("Credit Support Type", "none")),
        "hedge_accounting":      _b("Hedge Accounting Eligible (IFRS 9)", False),
        "shape_premium":         _f("Shape Premium (euros/MWh)", 0.0),
        # Dimensionnement
        "solar_cap_max_mw":      _f("Solar Capacity Max (MW)", 200.0),
        "wind_cap_max_mw":       _f("Wind Capacity Max (MW)", 400.0),
        "wind_unit_mw":          _f("Wind Turbine Unit Capacity (MW)", 15.0),
        "bin_size_mw":           _f("Bin Size (MW)", 5.0),
        "min_capacity_mw":       _f("Minimum Capacity (MW)", 10.0),
        # Batterie
        "include_battery":       _b("Include Battery Storage", False),
        "batt_capex_mw":         _f("Battery CAPEX per MW (EUR)", 300_000.0),
        "batt_capex_mwh":        _f("Battery CAPEX per MWh (EUR)", 150_000.0),
        "batt_eta":              _f("Battery Storage Efficiency (%)", 92.0),
        "batt_rt_eta":           _f("Battery Dispatch Efficiency (%)", 92.0),
        "batt_hours":            _f("Battery Max Hours (h)", 4.0),
        "batt_life_years":       _i("Battery Lifespan (years)", 15),
        "output_path":           str(_get("Output File Path", "output/")),
    }

    # Normalisation % -> fraction decimale
    for key in ("epex_increase", "rate_increase", "carbon_increase",
                "go_increase", "price_escalation_rate", "neg_sharing_pct",
                "grid_curtailment"):
        if abs(params[key]) > 1.0:
            params[key] /= 100.0

    for key in ("wacc", "irr_target", "max_grid_share"):
        if params[key] > 1.0:
            params[key] /= 100.0

    for key in ("batt_eta", "batt_rt_eta"):
        if params[key] > 1.0:
            params[key] /= 100.0

    return params


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROFIL TURPE HORAIRE
# ─────────────────────────────────────────────────────────────────────────────

def build_temporal_profile(params: dict,
                            turpe_filepath: str | None = None,
                            db_dir: Path | None = None
                            ) -> tuple[pd.DataFrame, float]:
    """
    Construit le profil tarifaire TURPE horaire pour l'annee modele.

    TURPE HTB CRE TURPE 6 (Nov. 2024) :
        HP Saison Haute  8.924 euros/MWh  |  HP Saison Basse  5.354 euros/MWh
        HC Saison Haute  6.824 euros/MWh  |  HC Saison Basse  3.570 euros/MWh
        + TICFE industrie 0.50 euros/MWh  |  + CTA 0.30 euros/MWh
        Composante puissance : 7.1 euros/kW/an (HTB2)

    Saisons : Haute = nov-mars | Basse = avr-oct
    HP = lun-sam 6h-22h

    Retourne temporal_df (horaire) et contract_fee (euros/kW/an).
    """
    from FranceGridUtils import process_france_grid_data, multiyear_pricing_france

    ddir          = db_dir or _DB_DIR
    model_year    = params.get("model_year", 2030)
    initial_year  = params.get("initial_year", 2024)
    turpe_level   = params.get("turpe_level", "HTB2")
    rate_increase = params.get("rate_increase", 0.02)
    duration      = params.get("contract_duration", 15)

    turpe_path = turpe_filepath or str(ddir / "TURPE_france.xlsx")

    try:
        temporal_df, contract_fee = process_france_grid_data(
            turpe_path, model_year, turpe_level)
    except Exception as e:
        print(f"[WARN] TURPE load failed ({e}) — fallback synthetique {turpe_level}")
        temporal_df, contract_fee = _synthetic_turpe(model_year, turpe_level)

    num_years = (model_year + duration) - initial_year + 1
    try:
        multi_df, _ = multiyear_pricing_france(
            temporal_df, contract_fee, initial_year, num_years, rate_increase)
    except Exception as e:
        print(f"[WARN] multiyear_pricing failed ({e})")
        multi_df = temporal_df.copy()

    if hasattr(multi_df.index, "year"):
        year_df = multi_df[multi_df.index.year == model_year].copy()
    else:
        year_df = temporal_df.copy()

    if year_df.empty:
        year_df = temporal_df.copy()

    if "contract_fee" in year_df.columns:
        year_df["total"] = year_df["rate"].fillna(0) + year_df["contract_fee"].fillna(0)
    else:
        year_df["total"] = year_df["rate"].fillna(0)

    return year_df, float(contract_fee)


def _synthetic_turpe(model_year: int, level: str = "HTB2") -> tuple[pd.DataFrame, float]:
    """TURPE synthetique CRE TURPE 6 HTB — fallback si fichier absent."""
    tarifs = {
        "HTB3": {"HPSH": 6.820, "HCSH": 5.060, "HPSB": 4.180, "HCSB": 2.820, "contract": 4.8},
        "HTB2": {"HPSH": 8.924, "HCSH": 6.824, "HPSB": 5.354, "HCSB": 3.570, "contract": 7.1},
        "HTB1": {"HPSH": 12.64, "HCSH": 9.560, "HPSB": 7.480, "HCSB": 5.120, "contract": 10.2},
    }
    t  = tarifs.get(level, tarifs["HTB2"])
    dr = pd.date_range(f"{model_year}-01-01", f"{model_year}-12-31 23:00", freq="h")

    saison_haute = dr.month.isin([11, 12, 1, 2, 3])
    heure_pleine = (dr.hour >= 6) & (dr.hour < 22) & (dr.dayofweek < 6)

    rate = np.where(
        saison_haute & heure_pleine,  t["HPSH"],
        np.where(saison_haute,        t["HCSH"],
        np.where(heure_pleine,        t["HPSB"],
                                      t["HCSB"]))
    ) + 0.50 + 0.30  # TICFE + CTA

    return pd.DataFrame({"rate": rate}, index=dr), t["contract"]


# ─────────────────────────────────────────────────────────────────────────────
# 3. LCOE PAR SITE (methode annuite IRENA)
# ─────────────────────────────────────────────────────────────────────────────

def compute_lcoe(technology: str, cf_mean: float, model_year: int = 2030,
                  grid_df: pd.DataFrame | None = None, wacc: float = 0.07,
                  connection_fees_eur_mw: float = 50_000.0) -> float:
    """
    LCOE = (CAPEX_annualise + OPEX_annuel) / Production_annuelle

    CAPEX annualise = CAPEX_mw * CRF
    CRF = wacc / (1 - (1+wacc)^-n)   (Capital Recovery Factor)

    CAPEX ajuste selon trajectoire IRENA (grid_france.csv) pour l'annee modele.
    """
    capex  = _CAPEX_2025.get(technology, 700_000)
    opex_r = _OPEX_RATE.get(technology, 0.02)
    n      = _LIFETIME.get(technology, 25)

    # Ajustement CAPEX selon trajectoire IRENA
    if grid_df is not None:
        col_map = {"solar": "solar_capex", "onshore": "wind_onshore_capex",
                   "offshore": "wind_offshore_capex", "hybrid": "solar_capex"}
        col = col_map.get(technology, "solar_capex")
        base_yr   = min(2025, int(grid_df.index.min()))
        target_yr = min(model_year, int(grid_df.index.max()))
        if (col in grid_df.columns and base_yr in grid_df.index
                and target_yr in grid_df.index):
            ratio = (float(grid_df.loc[target_yr, col])
                     / float(grid_df.loc[base_yr, col]))
            capex = capex * ratio

    crf          = wacc / (1 - (1 + wacc) ** (-n)) if wacc > 0 else 1.0 / n
    capex_annual = (capex + connection_fees_eur_mw) * crf
    opex_annual  = capex * opex_r
    prod_annual  = cf_mean * 8760

    if prod_annual <= 0:
        return 999.0
    return round((capex_annual + opex_annual) / prod_annual, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SCREENING DES SITES
# ─────────────────────────────────────────────────────────────────────────────

def run_site_screening(params: dict, load_col: str = "siderurgie",
                        db_dir: Path | None = None) -> pd.DataFrame:
    """
    Screene les sites candidats et calcule pour chacun :
        capture_rate_systeme  — valeur marchande relative (CR > 1 = produit aux heures cheres)
        correlation_load      — matching profil x charge client (Pearson)
        LCOE                  — cout de production annualise (euros/MWh)
        ppa_indicatif         — prix PPA = LCOE x (1+IRR) + sleeving + GO
        score_composite       — 0.5*CR + 0.3*corr + 0.2*(1-canni)
    """
    from FranceGridUtils import compute_all_site_metrics, screen_sites, load_capture_rates_db

    ddir     = db_dir or _DB_DIR
    ref_year = params.get("reference_year", 2020)
    model_yr = params.get("model_year", 2030)
    wacc     = params.get("wacc", 0.07)

    cr_df = load_capture_rates_db(ddir)
    if cr_df is not None and not cr_df.empty and "capture_rate_systeme" in cr_df.columns:
        print(f"[SCREENING] {len(cr_df)} sites depuis capture_rates.db")
        metrics_df = cr_df.copy()
    else:
        print("[SCREENING] Calcul a la volee...")
        metrics_df = compute_all_site_metrics(
            epex_year=ref_year, load_col=load_col, db_dir=ddir)

    if metrics_df.empty:
        return pd.DataFrame()

    grid_df = _load_grid(ddir)

    if "LCOE" not in metrics_df.columns:
        metrics_df["LCOE"] = metrics_df.apply(
            lambda r: compute_lcoe(str(r.get("technology","solar")),
                                    float(r.get("cf_mean", 0.15)),
                                    model_yr, grid_df, wacc), axis=1)

    irr    = params.get("irr_target", 0.08)
    sleev  = params.get("sleeving_fee", 3.0)
    go_p   = params.get("go_price_init", 8.0) * _GO_GRANULARITY_MULT.get(
                params.get("go_matching", "annual"), 1.0)
    go_inc = params.get("go_included", True)

    metrics_df["ppa_indicatif_eur_mwh"] = (
        metrics_df["LCOE"] * (1 + irr) + sleev
        + (go_p if go_inc else 0.0)
        + params.get("shape_premium", 0.0)
    ).round(2)

    if "score_composite" not in metrics_df.columns:
        metrics_df["score_composite"] = (
            0.50 * metrics_df["capture_rate_systeme"].clip(0, 2) / 1.5
            + 0.30 * metrics_df.get("correlation_load",
                     pd.Series(0, index=metrics_df.index)).fillna(0).clip(-1, 1)
            + 0.20 * (1 - metrics_df.get("cannibalization_risk",
                         pd.Series(0, index=metrics_df.index)).fillna(0).clip(0, 1))
        ).round(4)

    # S'assurer que les colonnes attendues par screen_sites() existent
    for col, default in [("correlation_load", 0.0),
                          ("cannibalization_risk", 0.0),
                          ("cf_mean", 0.15)]:
        if col not in metrics_df.columns:
            metrics_df[col] = default

    screened = screen_sites(metrics_df, min_cf=0.08, min_capture_rate=0.75,
                             max_cannibalization=0.40, top_n=20)
    
    print(f"[SCREENING] {len(screened)} sites retenus")
    return screened


# ─────────────────────────────────────────────────────────────────────────────
# 5. OPTIMISATION LP DU MIX PPA
# ─────────────────────────────────────────────────────────────────────────────

def optimize_ppa_mix(params: dict, screened_sites: pd.DataFrame,
                      temporal_df: pd.DataFrame,
                      db_dir: Path | None = None) -> dict:
    """
    Optimise le mix PPA par LP (scipy HiGHS).

    Formulation LP :
        min  sum_i  (ppa_i * p50_i) * x_i        (minimiser cout annuel total)
        s.t.
          sum_i  p50_i * x_i  >= target_mwh      (couverture charge minimale)
          sum_{solar} x_i     <= solar_cap_max    (plafond capacite solaire)
          sum_{wind}  x_i     <= wind_cap_max     (plafond capacite eolien)
          0 <= x_i <= cap_max_i                  (bornes par site)

    Variables x_i = MW installes site i.
    ppa_i = LCOE_i * (1 + IRR) + sleeving + GO (prix PPA = cout producteur + marge).
    p50_i = CF_i * 8760 (MWh/MW/an, production P50).
    """
    ddir = db_dir or _DB_DIR

    if screened_sites is None or screened_sites.empty:
        return _empty_mix()

    load_mw       = params.get("client_load_mw", 200.0)
    load_lf       = 0.80
    solar_max     = params.get("solar_cap_max_mw", 200.0)
    wind_max      = params.get("wind_cap_max_mw", 400.0)
    enr_share     = params.get("max_grid_share", 0.80)
    model_yr      = params.get("model_year", 2030)
    wacc          = params.get("wacc", 0.07)
    irr           = params.get("irr_target", 0.08)
    sleeving      = params.get("sleeving_fee", 3.0)
    go_p          = params.get("go_price_init", 8.0) * _GO_GRANULARITY_MULT.get(
                        params.get("go_matching", "annual"), 1.0)
    go_inc        = params.get("go_included", True)
    shape_p       = params.get("shape_premium", 0.0)
    bin_mw        = params.get("bin_size_mw", 5.0)
    min_cap       = params.get("min_capacity_mw", 10.0)

    sites = screened_sites.copy().reset_index(drop=True)
    n     = len(sites)
    grid_df = _load_grid(ddir)

    # CF et p50 par site
    cf  = np.array([float(sites.loc[i, "cf_mean"]) for i in range(n)])
    p50 = cf * 8760

    # LCOE si absent
    if "LCOE" not in sites.columns:
        sites["LCOE"] = [compute_lcoe(str(sites.loc[i,"technology"]),
                                       float(cf[i]), model_yr, grid_df, wacc)
                         for i in range(n)]

    lcoe_arr = np.array([float(sites.loc[i,"LCOE"]) for i in range(n)])
    ppa_arr  = lcoe_arr * (1 + irr) + sleeving + (go_p if go_inc else 0.0) + shape_p

    load_annual = load_mw * load_lf * 8760
    target_mwh  = load_annual * enr_share

    is_solar = np.array([1.0 if sites.loc[i,"technology"]=="solar" else 0.0
                          for i in range(n)])
    is_wind  = 1.0 - is_solar
    site_max = np.where(is_solar > 0, solar_max, wind_max)

    # LP
    c     = ppa_arr * p50
    A_ub  = np.array([-p50, is_solar, is_wind])
    b_ub  = np.array([-target_mwh, solar_max, wind_max])
    bounds= [(0.0, float(sm)) for sm in site_max]

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    x   = res.x if res.success else _greedy(sites, p50, target_mwh, site_max, solar_max, wind_max)

    # Arrondi + filtre min
    x = np.floor(x / bin_mw) * bin_mw
    x[x < min_cap] = 0.0

    mix_mw = {}; ppa_price = {}; lcoe_mix = {}
    for i in range(n):
        if x[i] > 0:
            s = sites.loc[i,"site"]
            mix_mw[s]    = float(round(x[i], 1))
            ppa_price[s] = float(round(ppa_arr[i], 2))
            lcoe_mix[s]  = float(round(lcoe_arr[i], 2))

    total_mw   = float(sum(mix_mw.values()))
    annual_gen = sum(mix_mw[s] * float(sites.loc[sites["site"]==s].iloc[0]["cf_mean"]) * 8760
                     for s in mix_mw) if mix_mw else 0.0
    coverage   = min(annual_gen / load_annual * 100, 100.0) if load_annual > 0 else 0.0

    epex_proj  = params.get("initial_epex",65.0) * (
                     (1 + params.get("epex_increase",0.02))
                     ** (model_yr - params.get("initial_year",2024)))
    w_ppa      = (sum(mix_mw[s]*ppa_price[s] for s in mix_mw) / total_mw
                  if total_mw > 0 else 0.0)
    net_cost   = w_ppa - epex_proj

    print(f"[LP] {total_mw:.0f} MW | {coverage:.1f}% couv. | "
          f"PPA {w_ppa:.1f} vs EPEX {epex_proj:.1f} euro/MWh ({net_cost:+.1f})")

    return {
        "mix_mw": mix_mw, "ppa_price": ppa_price, "lcoe_by_site": lcoe_mix,
        "total_ppa_mw": round(total_mw,1), "coverage_pct": round(coverage,1),
        "net_cost_eur_mwh": round(net_cost,2), "weighted_ppa_price": round(w_ppa,2),
        "annual_gen_mwh": round(annual_gen,0), "epex_projected": round(epex_proj,2),
        "load_annual_mwh": round(load_annual,0),
        "lp_status": res.status, "lp_success": res.success,
    }


def _greedy(sites, p50, target_mwh, site_max, solar_max, wind_max):
    x = np.zeros(len(sites))
    covered = solar_used = wind_used = 0.0
    for i in range(len(sites)):
        if covered >= target_mwh:
            break
        tech = sites.loc[i,"technology"]
        rem  = (solar_max - solar_used if tech == "solar" else wind_max - wind_used)
        cap  = min(float(site_max[i]), rem)
        if cap <= 0:
            continue
        alloc = min(cap, (target_mwh - covered) / max(p50[i], 1.0))
        x[i]  = alloc
        covered += alloc * p50[i]
        if tech == "solar":
            solar_used += alloc
        else:
            wind_used += alloc
    return x


def _empty_mix():
    return {"mix_mw": {}, "ppa_price": {}, "lcoe_by_site": {},
            "total_ppa_mw": 0.0, "coverage_pct": 0.0,
            "net_cost_eur_mwh": 0.0, "weighted_ppa_price": 0.0,
            "annual_gen_mwh": 0.0, "epex_projected": 0.0,
            "load_annual_mwh": 0.0, "lp_status": 2, "lp_success": False}


# ─────────────────────────────────────────────────────────────────────────────
# 6. SETTLEMENT HORAIRE CfD
# ─────────────────────────────────────────────────────────────────────────────

def compute_ppa_settlement(prod_profile: pd.Series, spot_profile: pd.Series,
                            ppa_strike: float, structure: str = "fixed",
                            params: dict | None = None) -> pd.DataFrame:
    """
    Settlement financier horaire du PPA selon 6 structures contractuelles.

    Structures :
      fixed            CfD classique  : settlement = (strike - spot) * Q
      fixed_escalating CfD prix indexe : strike croit avec inflation
      collar           Tunnel floor/cap : acheteur paie entre floor et cap
      indexed_spot     Decote marche  : settlement = discount * Q (pas de CfD)
      floor_only       Plancher seul  : settlement = max(0, floor-spot) * Q
      pay_as_produced  Volume livre   : idem fixed mais sans obligation volume

    Clauses appliquees dans l'ordre :
      1. Curtailment reseau TSO
      2. Curtailment volontaire (spot < seuil)
      3. Plancher prix negatifs (zero floor standard France post-2022)
      4. Basis risk (ecart prix site vs hub EEX)
      5. Structure tarifaire

    Retourne DataFrame horaire avec Q_eff, spot_settled, settlement_eur_h.
    """
    p = params or {}

    common = prod_profile.index.intersection(spot_profile.index)
    Q_raw  = prod_profile.reindex(common).fillna(0).values.astype(float)
    spot   = spot_profile.reindex(common).values.astype(float)

    # 1. Curtailment reseau
    Q_eff = Q_raw * (1.0 - float(p.get("grid_curtailment", 0.015)))

    # 2. Curtailment volontaire
    curtailed = np.zeros(len(Q_eff), dtype=bool)
    curt_thr  = p.get("vol_curtail_thr", None)
    if curt_thr is not None:
        curtailed = spot < float(curt_thr)
        Q_eff[curtailed] = 0.0

    # 3. Spot de settlement avec plancher prix negatifs
    spot_s = np.maximum(spot, float(p.get("neg_price_floor", 0.0)))

    # 4. Basis risk
    spot_s = spot_s - float(p.get("basis_risk", 0.5))

    # 5. Structure
    if structure == "fixed":
        pay_mwh = ppa_strike - spot_s

    elif structure == "fixed_escalating":
        esc     = float(p.get("price_escalation_rate", 0.015))
        t_yr    = np.arange(len(common)) / 8760.0
        pay_mwh = ppa_strike * (1 + esc) ** t_yr - spot_s

    elif structure == "collar":
        band    = float(p.get("collar_band", 0.15))
        spot_c  = np.clip(spot_s, ppa_strike*(1-band), ppa_strike*(1+band))
        pay_mwh = ppa_strike - spot_c

    elif structure == "indexed_spot":
        pay_mwh = np.full(len(spot_s), float(p.get("spot_discount", 5.0)))

    elif structure == "floor_only":
        floor_p = float(p.get("price_floor") or ppa_strike * 0.85)
        pay_mwh = np.maximum(0.0, floor_p - spot_s)

    else:
        pay_mwh = ppa_strike - spot_s

    return pd.DataFrame({
        "Q_raw_pu":            Q_raw,
        "Q_eff_pu":            Q_eff,
        "spot_raw_eur_mwh":    spot,
        "spot_settled_eur_mwh":spot_s,
        "settlement_eur_mwh":  pay_mwh,
        "settlement_eur_h":    pay_mwh * Q_eff,
        "curtailed":           curtailed.astype(int),
    }, index=common)


# ─────────────────────────────────────────────────────────────────────────────
# 7. CASHFLOW & VAN
# ─────────────────────────────────────────────────────────────────────────────

def run_npv_cashflow(params: dict, mix_result: dict, temporal_df: pd.DataFrame,
                      db_dir: Path | None = None) -> pd.DataFrame:
    """
    Projette le cashflow sur la duree du contrat.

    Methodologie France — ce que font les analystes :

    Cout reference (sans PPA, depuis fin ARENH jan.2026) :
        C_ref(t) = [EPEX_proj(t) + GO_marche(t) + TURPE(t) + TICFE] * Load(t)

    Cout avec PPA :
        C_ppa(t) = PPA_strike(t) * vol_ppa(t)    <- energie couverte
                 + sleeving * vol_ppa(t)          <- fournisseur intermediaire
                 + EPEX_proj(t) * vol_resid(t)    <- volume non couvert
                 + TURPE(t) * Load(t)             <- reseau toujours du (off-site)
                 + TICFE * Load(t)                <- taxe toujours due

    Economie = C_ref - C_ppa
    VAN = sum[ Economie(t) / (1+WACC)^t ]

    CO2 evite (Scope 2 market-based, GHG Protocol) :
        = vol_ppa * intensite_CO2_reseau(t)
    """
    ddir = db_dir or _DB_DIR

    model_yr  = params.get("model_year", 2030)
    init_yr   = params.get("initial_year", 2024)
    duration  = params.get("contract_duration", 15)
    wacc      = params.get("wacc", 0.07)
    load_mw   = params.get("client_load_mw", 200.0)
    load_lf   = 0.80
    load_annual = load_mw * load_lf * 8760

    epex_init   = params.get("initial_epex", 65.0)
    epex_drift  = params.get("epex_increase", 0.02)
    turpe_drift = params.get("rate_increase", 0.02)
    go_init     = params.get("go_price_init", 8.0) * _GO_GRANULARITY_MULT.get(
                      params.get("go_matching","annual"), 1.0)
    go_drift    = params.get("go_increase", -0.02)
    go_inc      = params.get("go_included", True)
    sleeving    = params.get("sleeving_fee", 3.0)
    carbon_init = params.get("carbon_price_init", 65.0)
    carbon_drft = params.get("carbon_increase", 0.05)
    structure   = params.get("price_structure", "fixed")
    ppa_esc     = params.get("price_escalation_rate", 0.0)
    ppa_tech    = params.get("ppa_technology", "solar")
    degrad      = _DEGRADATION.get(ppa_tech, 0.004)

    strike_base = mix_result.get("weighted_ppa_price", 65.0)
    cov_base    = mix_result.get("coverage_pct", 80.0) / 100.0

    turpe_base  = float(temporal_df["total"].mean() if "total" in temporal_df.columns
                        else temporal_df.get("rate", pd.Series([7.5])).mean())
    ticfe       = 0.50  # euros/MWh, taux industrie

    grid_df = _load_grid(ddir)

    rows = []
    for offset, yr in enumerate(range(model_yr, model_yr + duration)):

        epex_yr   = epex_init * ((1 + epex_drift) ** (yr - init_yr))
        turpe_yr  = turpe_base * ((1 + turpe_drift) ** offset)
        go_yr     = max(go_init * ((1 + go_drift) ** offset), 0.0) if go_inc else 0.0
        carbon_yr = carbon_init * ((1 + carbon_drft) ** offset)
        ppa_yr    = strike_base * ((1 + ppa_esc) ** offset)

        # CO2 et ENR reseau
        co2_g_kwh = _CO2_GRID_2024; ren_share = 0.35
        if grid_df is not None:
            yr_c = min(yr, int(grid_df.index.max()))
            if yr_c in grid_df.index:
                co2_g_kwh = float(grid_df.loc[yr_c, "co2_intensity_g_kwh"])
                ren_share  = float(grid_df.loc[yr_c, "ren_share"])

        # Volumes
        eff_cov  = min(cov_base * (1 - degrad) ** offset, 1.0)
        vol_ppa  = load_annual * eff_cov
        vol_grid = load_annual - vol_ppa

        # Cout reference SANS PPA
        cost_ref_mwh   = epex_yr + go_yr + turpe_yr + ticfe
        cost_ref_annual= cost_ref_mwh * load_annual

        # Cout AVEC PPA
        cost_ppa_annual = (
            (ppa_yr + sleeving) * vol_ppa     # energie PPA
            + epex_yr * vol_grid               # energie residuelle spot
            + (turpe_yr + ticfe) * load_annual # reseau (toujours du)
        )

        savings_yr = cost_ref_annual - cost_ppa_annual
        savings_pv = savings_yr * ((1 + wacc) ** (-offset))
        co2_avoided= vol_ppa * (co2_g_kwh / 1000)

        rows.append({
            "year":                   yr,
            "epex_projected_eur_mwh": round(epex_yr, 2),
            "turpe_eur_mwh":          round(turpe_yr, 2),
            "ticfe_eur_mwh":          round(ticfe, 2),
            "go_eur_mwh":             round(go_yr, 2),
            "ppa_price_eur_mwh":      round(ppa_yr, 2),
            "cost_ref_eur_mwh":       round(cost_ref_mwh, 2),
            "cost_ppa_eur_mwh":       round(ppa_yr + sleeving + turpe_yr + ticfe, 2),
            "vol_ppa_mwh":            round(vol_ppa, 0),
            "vol_grid_mwh":           round(vol_grid, 0),
            "savings_eur_yr":         round(savings_yr, 0),
            "savings_pv_eur":         round(savings_pv, 0),
            "co2_avoided_t":          round(co2_avoided, 0),
            "coverage_pct":           round(eff_cov * 100, 1),
            "carbon_price_eur_tco2":  round(carbon_yr, 2),
            "ren_share_grid":         round(ren_share, 3),
        })

    cf = pd.DataFrame(rows).set_index("year")

    npv     = float(cf["savings_pv_eur"].sum())
    cumul   = cf["savings_pv_eur"].cumsum()
    payback = next((int(yr) - model_yr for yr, v in cumul.items() if v > 0), None)
    irr_val = _irr(cf["savings_eur_yr"].values)

    cf.attrs.update({
        "npv_eur":       round(npv, 0),
        "payback_years": payback,
        "irr_pct":       round(irr_val * 100, 2) if irr_val is not None else None,
        "total_co2_t":   round(float(cf["co2_avoided_t"].sum()), 0),
        "avg_coverage":  round(float(cf["coverage_pct"].mean()), 1),
        "model_year":    model_yr,
        "duration":      duration,
    })

    print(f"\n[VAN] {npv/1e6:.1f} M€ | Payback {payback} ans"
          + (f" | IRR {irr_val*100:.1f}%" if irr_val else ""))
    print(f"[CO2] {cf['co2_avoided_t'].sum()/1000:.0f} ktCO2 evites sur {duration} ans")
    return cf


def _irr(cashflows: np.ndarray) -> float | None:
    if len(cashflows) < 2 or cashflows.sum() <= 0:
        return None
    t = np.arange(len(cashflows), dtype=float)
    try:
        lo, hi = -0.50, 5.0
        for _ in range(300):
            mid = (lo + hi) / 2
            if mid <= -1.0:
                return None
            npv = float(np.sum(cashflows / (1 + mid) ** t))
            if npv > 0:
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) < 1e-7:
                break
        irr = (lo + hi) / 2
        return irr if -0.5 < irr < 5.0 else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 8. POINT D'ENTREE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_model(xlsx_path: str = "scenario_defaults.xlsx",
              scenario_col: str = "PPA100_ArcelorMittal_MU",
              load_col: str = "siderurgie",
              db_dir: str | None = None,
              save_output: bool = False) -> dict:
    """
    Execute le modele PPA complet pour un scenario.

    Usage :
        from ppamodule import run_model
        r = run_model('scenario_defaults.xlsx', 'PPA100_ArcelorMittal_MU')
        print(f"VAN : {r['npv_eur']/1e6:.1f} M€  |  Payback : {r['payback_years']} ans")
    """
    ddir = Path(db_dir) if db_dir else _DB_DIR
    print(f"\n{'='*64}\n French PPA Model v3.0  |  {scenario_col}\n{'='*64}\n")

    params = load_scenario(xlsx_path, scenario_col)
    print(f"[1/5] charge={params['client_load_mw']}MW | annee={params['model_year']} | TURPE={params['turpe_level']}")

    temporal_df, contract_fee = build_temporal_profile(params, db_dir=ddir)
    print(f"[2/5] TURPE moy={temporal_df.get('rate',pd.Series([0])).mean():.2f} euro/MWh | CF={contract_fee:.2f} euro/kW/an")

    screened = run_site_screening(params, load_col=load_col, db_dir=ddir)
    print(f"[3/5] {len(screened)} sites retenus")

    mix = optimize_ppa_mix(params, screened, temporal_df, db_dir=ddir)
    print(f"[4/5] Mix: {mix['total_ppa_mw']:.0f}MW | {mix['coverage_pct']:.1f}% | {mix['weighted_ppa_price']:.1f} euro/MWh")

    cashflow = run_npv_cashflow(params, mix, temporal_df, db_dir=ddir)

    results = {
        "scenario_name": scenario_col, "params": params,
        "temporal_df": temporal_df, "contract_fee": contract_fee,
        "screened": screened, "mix": mix, "cashflow": cashflow,
        "npv_eur":         cashflow.attrs.get("npv_eur", 0),
        "payback_years":   cashflow.attrs.get("payback_years"),
        "irr_pct":         cashflow.attrs.get("irr_pct"),
        "total_co2_t":     cashflow.attrs.get("total_co2_t"),
        "coverage_pct":    mix.get("coverage_pct"),
        "net_cost_eur_mwh":mix.get("net_cost_eur_mwh"),
    }

    if save_output:
        out = Path(params.get("output_path", "output/"))
        out.mkdir(parents=True, exist_ok=True)
        cashflow.reset_index().to_csv(out / f"cashflow_{scenario_col}.csv", index=False)
        if not screened.empty:
            screened.to_csv(out / f"screened_{scenario_col}.csv", index=False)
        print(f"[OUTPUT] -> {out}")

    print(f"\n{'='*64}\n")
    return results


def compare_scenarios(xlsx_path: str, scenario_cols: list[str],
                       load_col: str = "siderurgie",
                       db_dir: str | None = None) -> pd.DataFrame:
    """
    Compare plusieurs scenarios, retourne un DataFrame de synthese.

    Usage :
        df = compare_scenarios('scenario_defaults.xlsx',
                               ['PPA100_ArcelorMittal_MU', 'PPA100_H2_Normandie'])
        print(df[['scenario', 'npv_meur', 'coverage_pct', 'irr_pct']])
    """
    rows = []
    for sc in scenario_cols:
        try:
            r = run_model(xlsx_path, sc, load_col, db_dir)
            rows.append({
                "scenario":         sc,
                "npv_meur":         round(r["npv_eur"] / 1e6, 1),
                "payback_years":    r["payback_years"],
                "irr_pct":          r["irr_pct"],
                "coverage_pct":     r["coverage_pct"],
                "net_cost_eur_mwh": r["net_cost_eur_mwh"],
                "total_ppa_mw":     r["mix"].get("total_ppa_mw"),
                "co2_kt":           round(r["total_co2_t"]/1000, 0) if r["total_co2_t"] else None,
                "status":           "OK",
            })
        except Exception as e:
            rows.append({"scenario": sc, "status": f"ERROR: {e}"})
    return pd.DataFrame(rows)
