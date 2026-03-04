"""
download_france_data.py
=======================
Télécharge et génère toutes les données françaises pour le French PPA Model.

Structure créée :
  French_PPA/
  ├── database/
  │   ├── grid_france.csv          ← CAPEX/OPEX/CO2/ENR/EPEX trajectoires 2023-2050
  │   ├── epex_profiles.db         ← Prix spot EPEX horaire (réel ou synthétique)
  │   ├── solar_patterns.db        ← 10 sites solaires multi-colonnes (nommés par site)
  │   ├── wind_patterns.db         ← 10 zones éoliennes (offshore + onshore)
  │   ├── load_patterns.db         ← 4 profils industriels types + load_reel optionnel
  │   ├── capture_rates.db         ← capture_rate_systeme calculé par site
  │   ├── TURPE_france.xlsx        ← TURPE 6 HTB (généré si absent)
  │   └── wind_grid_france.xlsx    ← 12 parcs éoliens (offshore + onshore)
  └── gisdata/

Sources :
  PVGIS     : https://re.jrc.ec.europa.eu/api/v5_2/seriescalc
  Open-Meteo: https://archive-api.open-meteo.com/v1/archive
  EPEX/ODRE : CSV local ou synthétique calibré sur historique RTE

Usage :
  python download_france_data.py                       # tout télécharger
  python download_france_data.py --offline             # tout synthétique
  python download_france_data.py --epex path.csv       # EPEX depuis fichier ODRE
  python download_france_data.py --client-lat 51.03 --client-lon 2.38
  python download_france_data.py --load-csv conso.csv  # profil charge réel client
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).resolve().parent
DB_DIR     = BASE_DIR / "database"
GIS_DIR    = BASE_DIR / "gisdata"

METEO_YEAR = 2020          # Année météo de référence (données ERA5/PVGIS)
EPEX_MEAN_2024 = 65.0      # Prix EPEX moyen 2024 (€/MWh) — RTE bilan 2024


# ─────────────────────────────────────────────────────────────────────────────
# CATALOGUE DES SITES
# ─────────────────────────────────────────────────────────────────────────────

# (nom_colonne, lat, lon, cf_min_attendu, cf_max_attendu, description)
SOLAR_SITES = [
    ("montpellier",    43.61,  3.88, 0.175, 0.225, "Hérault — meilleur gisement France métro"),
    ("bordeaux",       44.84, -0.58, 0.155, 0.195, "Gironde — gisement SW France"),
    ("lyon",           45.75,  4.85, 0.145, 0.185, "Rhône — gisement Centre-Est"),
    ("nantes",         47.22, -1.55, 0.130, 0.165, "Loire-Atlantique — gisement Ouest"),
    ("paris_idf",      48.86,  2.35, 0.115, 0.150, "Île-de-France — gisement moyen"),
    ("lille",          50.63,  3.07, 0.105, 0.135, "Nord — gisement le plus faible"),
    ("marseille",      43.30,  5.40, 0.185, 0.235, "BdR — potentiel agrivoltaïque PACA"),
    ("toulouse",       43.60,  1.44, 0.170, 0.215, "Haute-Garonne — SW France"),
    ("strasbourg",     48.57,  7.75, 0.135, 0.175, "Alsace — gisement Est"),
    ("dunkerque_sol",  51.03,  2.38, 0.100, 0.130, "Nord — gisement côtier faible"),
]

# (nom_colonne, lat, lon, wind_type, cf_min, cf_max, description)
WIND_SITES = [
    ("dunkerque_off",   51.10,  2.20, "offshore", 0.38, 0.48, "EMR Dunkerque — meilleur offshore FR"),
    ("saint_nazaire_off", 47.20, -2.50, "offshore", 0.35, 0.45, "EMR Saint-Nazaire"),
    ("fecamp_off",      49.80,  0.50, "offshore", 0.33, 0.42, "EMR Fécamp — Normandie"),
    ("saint_brieuc_off", 48.60, -2.80, "offshore", 0.32, 0.42, "EMR Saint-Brieuc"),
    ("golfe_lion_off",  43.10,  4.20, "offshore", 0.30, 0.40, "Offshore flottant Méditerranée"),
    ("hdf_onshore",     50.50,  2.50, "onshore",  0.28, 0.38, "Hauts-de-France — zone dense"),
    ("normandie_on",    49.40,  0.80, "onshore",  0.24, 0.33, "Normandie côtière"),
    ("bretagne_on",     48.10, -3.00, "onshore",  0.25, 0.35, "Bretagne centrale"),
    ("pays_loire_on",   47.50, -1.00, "onshore",  0.22, 0.32, "Pays de la Loire"),
    ("occitanie_on",    43.80,  3.00, "onshore",  0.20, 0.30, "Occitanie collines"),
]


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES SQLite
# ─────────────────────────────────────────────────────────────────────────────

def _upsert_column(db: Path, table: str, idx_col: str,
                   idx_vals: list, col: str, vals: np.ndarray) -> None:
    """
    Ajoute/remplace une colonne dans une table SQLite multi-colonnes.
    Crée la table si elle n'existe pas encore.
    """
    conn = sqlite3.connect(str(db))
    try:
        existing = pd.read_sql(f"SELECT * FROM {table}", conn, index_col=idx_col)
    except Exception:
        existing = pd.DataFrame(index=pd.Index(idx_vals, name=idx_col))

    existing[col] = pd.Series(vals, index=idx_vals).reindex(existing.index).values
    existing.reset_index().to_sql(table, conn, if_exists="replace", index=False)
    conn.close()


def _read_column(db: Path, table: str, idx_col: str, col: str) -> pd.Series | None:
    """Lit une colonne depuis SQLite. Retourne None si inexistante."""
    if not db.exists():
        return None
    conn = sqlite3.connect(str(db))
    try:
        df = pd.read_sql(f"SELECT {idx_col}, {col} FROM {table}", conn,
                         index_col=idx_col, parse_dates=[idx_col])
        conn.close()
        return df[col]
    except Exception:
        conn.close()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. GRID FRANCE CSV
# ─────────────────────────────────────────────────────────────────────────────

def create_grid_france_csv(epex_override: dict | None = None) -> pd.DataFrame:
    """
    Génère database/grid_france.csv avec trajectoires 2023-2050 calibrées.

    Colonnes :
      solar_capex, solar_opex,
      wind_onshore_capex, wind_onshore_opex,
      wind_offshore_capex, wind_offshore_opex,
      bess_capex_per_mwh, bess_opex_per_mwh,
      co2_intensity_g_kwh, ren_share, epex_spot_mean,
      solar_cf_ref, wind_onshore_cf_ref, wind_offshore_cf_ref

    Sources : IRENA 2023, RTE Bilan 2024, BloombergNEF BESS 2023, RTE Futurs Énergétiques
    """
    years = list(range(2023, 2051))
    n     = len(years)
    yr    = np.array(years)

    # CAPEX solaire : 700 k€/MW 2023 → plancher 300 k€/MW (IRENA France 2023)
    solar_capex = np.maximum(700_000 * np.exp(-0.028 * (yr - 2023)), 300_000)
    solar_opex  = solar_capex * 0.018

    # CAPEX éolien onshore : 1 500 k€/MW 2023 → 1 100 k€/MW plancher
    wind_on_capex = np.maximum(1_500_000 * np.exp(-0.010 * (yr - 2023)), 1_100_000)
    wind_on_opex  = wind_on_capex * 0.025

    # CAPEX éolien offshore : 3 000 k€/MW 2023 → 1 600 k€/MW (Saint-Nazaire référence)
    wind_off_capex = np.maximum(3_000_000 * np.exp(-0.022 * (yr - 2023)), 1_600_000)
    wind_off_opex  = wind_off_capex * 0.030

    # BESS LFP : 200 k€/MWh 2023 → 60 k€/MWh (BloombergNEF)
    bess_capex = np.maximum(200_000 * np.exp(-0.035 * (yr - 2023)), 60_000)
    bess_opex  = bess_capex * 0.020

    # CO2 réseau : 46 gCO2/kWh 2024 → 8 gCO2/kWh 2050 (RTE eco2mix + PPE)
    co2 = np.maximum(46.0 * np.exp(-0.045 * (yr - 2023)), 8.0)

    # Part ENR : 29% 2023 → 80% 2050 (RTE Futurs Énergétiques scénario central)
    ren_share = 0.29 + (0.80 - 0.29) * (1 - np.exp(-0.060 * (yr - 2023)))

    # EPEX spot moyen annuel (€/MWh) : 96 €/MWh 2023, 65 €/MWh 2024, +2%/an ensuite
    if epex_override:
        epex = np.array([epex_override.get(y, EPEX_MEAN_2024 * 1.02 ** (y - 2024))
                         for y in years])
    else:
        epex = np.where(yr <= 2024,
                        np.interp(yr, [2023, 2024], [96.0, 65.0]),
                        65.0 * (1.02 ** (yr - 2024)))

    df = pd.DataFrame({
        "year":                 years,
        "solar_capex":          solar_capex.round(0),
        "solar_opex":           solar_opex.round(0),
        "wind_onshore_capex":   wind_on_capex.round(0),
        "wind_onshore_opex":    wind_on_opex.round(0),
        "wind_offshore_capex":  wind_off_capex.round(0),
        "wind_offshore_opex":   wind_off_opex.round(0),
        "bess_capex_per_mwh":   bess_capex.round(0),
        "bess_opex_per_mwh":    bess_opex.round(0),
        "co2_intensity_g_kwh":  co2.round(2),
        "ren_share":            ren_share.round(4),
        "epex_spot_mean":       epex.round(2),
        "solar_cf_ref":         np.full(n, 0.148),        # moyenne France PVGIS
        "wind_onshore_cf_ref":  np.full(n, 0.280),        # RTE 2023
        "wind_offshore_cf_ref": np.full(n, 0.400),        # EMR opérationnels 2024
    }).set_index("year")

    out = DB_DIR / "grid_france.csv"
    df.to_csv(out)
    print(f"[OK] grid_france.csv ({n} années) → {out}")
    print(f"     CAPEX PV 2024    : {df.loc[2024,'solar_capex']/1e3:.0f} k€/MW")
    print(f"     CAPEX offshore   : {df.loc[2024,'wind_offshore_capex']/1e3:.0f} k€/MW")
    print(f"     EPEX 2024        : {df.loc[2024,'epex_spot_mean']:.1f} €/MWh")
    print(f"     CO2 réseau 2024  : {df.loc[2024,'co2_intensity_g_kwh']:.1f} gCO2/kWh")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. EPEX SPOT HORAIRE
# ─────────────────────────────────────────────────────────────────────────────

def load_or_generate_epex(year: int = METEO_YEAR,
                           csv_path: Path | None = None,
                           force_synthetic: bool = False) -> pd.Series:
    """
    Charge le profil EPEX Day-Ahead France horaire.

    Priorité :
      1. Colonne déjà en cache dans epex_profiles.db
      2. CSV ODRE fourni explicitement
      3. CSV ODRE auto-détecté dans database/epex_spot_{year}.csv
      4. Profil synthétique calibré sur saisonnalité réelle

    Format CSV ODRE attendu : séparateur ';', colonnes Date/Heures/France (€/MWh)
    """
    db    = DB_DIR / "epex_profiles.db"
    table = f"epex_{year}"

    # Cache
    if db.exists() and not force_synthetic:
        conn = sqlite3.connect(str(db))
        try:
            df = pd.read_sql(f"SELECT * FROM {table}", conn,
                             index_col="datetime", parse_dates=["datetime"])
            conn.close()
            s = df["price_eur_mwh"]
            print(f"[CACHE] EPEX {year} ({len(s)} h, moy={s.mean():.1f} €/MWh)")
            return s
        except Exception:
            conn.close()

    # CSV fourni ou auto-détecté
    for candidate in [csv_path, DB_DIR / f"epex_spot_{year}.csv"]:
        if candidate and Path(candidate).exists():
            s = _parse_epex_csv(Path(candidate), year)
            if s is not None:
                _save_epex(s, db, table)
                return s

    # Synthétique
    print(f"[SYNTH] EPEX {year} — profil synthétique")
    s = _synthetic_epex(year)
    _save_epex(s, db, table)
    return s


def _parse_epex_csv(path: Path, year: int) -> pd.Series | None:
    """Parse CSV ODRE (séparateur ';', décimale ',')."""
    try:
        df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        date_col  = next((c for c in df.columns if "date" in c.lower()), None)
        hour_col  = next((c for c in df.columns if "heure" in c.lower() or "hour" in c.lower()), None)
        price_col = next((c for c in df.columns
                          if "france" in c.lower() or "prix" in c.lower()), None)
        if not all([date_col, hour_col, price_col]):
            return None

        df["dt"] = pd.to_datetime(df[date_col] + " " + df[hour_col],
                                   format="%d/%m/%Y %H:%M", errors="coerce")
        df = df.dropna(subset=["dt"])
        df = df[df["dt"].dt.year == year].copy()
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

        s = df.set_index("dt")[price_col].rename("price_eur_mwh")
        s = s.resample("h").mean()
        dr = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
        s  = s.reindex(dr).interpolate("time").ffill().bfill()
        s.index.name = "datetime"
        print(f"[CSV] EPEX {year} depuis {path.name} : moy={s.mean():.1f} €/MWh")
        return s
    except Exception as e:
        print(f"[WARN] Erreur parsing EPEX : {e}")
        return None


def _synthetic_epex(year: int) -> pd.Series:
    """
    Profil EPEX synthétique calibré sur saisonnalité française.

    Niveaux de base historiques (€/MWh) :
      2020=34, 2021=109, 2022=276, 2023=96, 2024=65
    Saisonnalité : pic hivernal, creux estival (cannibalisation solaire).
    Heures négatives : ~350/an en 2024 (printemps/été, midi + nuit venteuse).
    """
    base_by_year = {2018: 47, 2019: 39, 2020: 34, 2021: 109,
                    2022: 276, 2023: 96, 2024: 65}
    base = base_by_year.get(year, EPEX_MEAN_2024)

    dr  = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    n   = len(dr)
    t   = np.arange(n)
    h   = dr.hour.values
    doy = dr.dayofyear.values

    # Saisonnalité annuelle (pic janvier)
    seasonal = 1.0 + 0.35 * np.cos(2 * np.pi * (t / 8760 - 0.08))

    # Profil journalier double pic (matin + soir)
    daily = (1.0
             + 0.20 * np.exp(-0.5 * ((h - 9) / 2.5) ** 2)
             + 0.25 * np.exp(-0.5 * ((h - 19) / 2.0) ** 2)
             - 0.15 * np.exp(-0.5 * ((h - 3) / 3.0) ** 2))

    # Effet week-end (-12%)
    weekend = 1.0 - 0.12 * (dr.weekday.values >= 5).astype(float)

    # Dépression solaire printemps/été 11h-15h (cannibalisation)
    solar_dep = 1.0 - 0.25 * (
        (dr.month.values >= 4) & (dr.month.values <= 8) &
        (h >= 11) & (h <= 15)
    ).astype(float)

    price = base * seasonal * daily * weekend * solar_dep
    rng   = np.random.default_rng(seed=year)
    price = price * (1 + rng.normal(0, 0.12, n))

    # Heures négatives (printemps/été midi + nuit venteuse)
    n_neg = {2018: 20, 2019: 30, 2020: 90, 2021: 60,
             2022: 40, 2023: 147, 2024: 352}.get(year, 200)
    neg_mask = (
        ((dr.month.values >= 3) & (dr.month.values <= 8)) &
        (((h >= 11) & (h <= 15)) | ((h >= 1) & (h <= 5)))
    )
    neg_idx = np.where(neg_mask)[0]
    if len(neg_idx) > n_neg:
        chosen = rng.choice(neg_idx, size=n_neg, replace=False)
        price[chosen] = rng.uniform(-30, 0, n_neg)

    s = pd.Series(price.round(2), index=dr, name="price_eur_mwh")
    s.index.name = "datetime"
    n_neg_real = (price < 0).sum()
    print(f"     moy={price.mean():.1f} €/MWh  min={price.min():.0f}  "
          f"max={price.max():.0f}  heures négatives={n_neg_real}")
    return s


def _save_epex(s: pd.Series, db: Path, table: str) -> None:
    conn = sqlite3.connect(str(db))
    df   = s.reset_index()
    df["datetime"] = df["datetime"].astype(str)
    df.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3. PROFILS SOLAIRES — solar_patterns.db
# ─────────────────────────────────────────────────────────────────────────────

def _pvgis_fetch(lat: float, lon: float, year: int, name: str) -> pd.Series | None:
    """Appel PVGIS API v5.2. Retourne None si inaccessible."""
    try:
        r = requests.get(
            "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc",
            params={
                "lat": lat, "lon": lon,
                "raddatabase": "PVGIS-SARAH2",
                "startyear": year, "endyear": year,
                "pvcalculation": 1, "peakpower": 1,
                "loss": 14, "pvtechchoice": "crystSi",
                "mountingplace": "free", "angle": 30, "aspect": 0,
                "outputformat": "json", "browser": 0,
            },
            timeout=30,
        )
        r.raise_for_status()
        hourly = pd.DataFrame(r.json()["outputs"]["hourly"])
        hourly["time"] = pd.to_datetime(hourly["time"], format="%Y%m%d:%H%M")
        cf = (hourly.set_index("time")["P"] / 1000.0).clip(0, 1)
        cf.name = name
        cf.index.name = "datetime"
        dr = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
        cf = cf.resample("h").mean().reindex(dr).ffill().bfill()
        print(f"  [PVGIS] {name:<22} CF={cf.mean():.3f}")
        return cf
    except requests.exceptions.ConnectionError:
        print(f"  [WARN] PVGIS inaccessible → synthétique pour {name}")
        return None
    except Exception as e:
        print(f"  [WARN] PVGIS erreur {name} : {e} → synthétique")
        return None


def _synthetic_solar(lat: float, year: int, name: str) -> pd.Series:
    """
    Profil solaire synthétique réaliste calibré sur latitude.
    CF de pointe : ~0.198 à 43°N → ~0.094 à 51°N.
    """
    dr  = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    h   = dr.hour.values
    doy = dr.dayofyear.values

    cf_peak  = float(np.clip(0.198 - 0.0059 * (lat - 43.0), 0.09, 0.22))
    seasonal = 1.0 + 0.55 * np.sin(2 * np.pi * (doy - 80) / 365)

    # Lever/coucher approx
    decl    = 23.45 * np.sin(np.radians(360 / 365 * (doy - 81)))
    ha      = np.degrees(np.arccos(np.clip(-np.tan(np.radians(lat)) * np.tan(np.radians(decl)), -1, 1)))
    sunrise = 12.0 - ha / 15.0
    sunset  = 12.0 + ha / 15.0

    daylight = (h > sunrise) & (h < sunset)
    gaussian = np.exp(-0.5 * ((h - 12.0) / 3.2) ** 2)
    cf       = np.where(daylight, cf_peak * seasonal * gaussian, 0.0)

    rng = np.random.default_rng(seed=int(lat * 100 + year))
    cf  = np.clip(cf * rng.lognormal(0, 0.05, len(dr)), 0, 1)

    s = pd.Series(cf, index=dr, name=name)
    s.index.name = "datetime"
    print(f"  [SYNTH] {name:<22} CF={cf.mean():.3f}")
    return s


def download_solar_site(name: str, lat: float, lon: float,
                         year: int = METEO_YEAR,
                         force_synthetic: bool = False) -> pd.Series:
    """Télécharge ou génère le profil d'un site solaire, stocke dans solar_patterns.db."""
    db = DB_DIR / "solar_patterns.db"
    # Cache check
    if not force_synthetic:
        existing = _read_column(db, "solar_patterns", "datetime", name)
        if existing is not None and len(existing) >= 8700:
            print(f"  [CACHE] {name}")
            return existing

    s = _pvgis_fetch(lat, lon, year, name) if not force_synthetic else None
    if s is None:
        s = _synthetic_solar(lat, year, name)

    dr   = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    vals = s.reindex(dr).fillna(0).values
    _upsert_column(db, "solar_patterns", "datetime", dr.astype(str).tolist(), name, vals)
    return s


def download_all_solar(year: int = METEO_YEAR,
                        client_lat: float | None = None,
                        client_lon: float | None = None,
                        force_synthetic: bool = False) -> None:
    print("\n── PROFILS SOLAIRES (solar_patterns.db) " + "─" * 25)
    for name, lat, lon, *_ in SOLAR_SITES:
        download_solar_site(name, lat, lon, year, force_synthetic)
    if client_lat is not None and client_lon is not None:
        download_solar_site("client_onsite", client_lat, client_lon, year, force_synthetic)
    print(f"[OK] solar_patterns.db → {DB_DIR / 'solar_patterns.db'}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PROFILS ÉOLIENS — wind_patterns.db
# ─────────────────────────────────────────────────────────────────────────────

def _power_curve(ws: np.ndarray, wind_type: str) -> np.ndarray:
    """
    Courbe de puissance normalisée — cubique entre cut-in et rated, plat jusqu'à cut-out.
    offshore (SG 14-222) : cut_in=3, rated=13, cut_out=25
    onshore  (V150)      : cut_in=3, rated=12, cut_out=22
    """
    v_in, v_rated, v_out = (3.0, 13.0, 25.0) if wind_type == "offshore" else (3.0, 12.0, 22.0)
    cf = np.zeros_like(ws, dtype=float)
    cf[(ws >= v_in) & (ws < v_rated)] = ((ws[(ws >= v_in) & (ws < v_rated)] - v_in)
                                          / (v_rated - v_in)) ** 3
    cf[(ws >= v_rated) & (ws <= v_out)] = 1.0
    return np.clip(cf, 0, 1)


def _openmeteo_fetch(lat: float, lon: float, year: int,
                      name: str, wind_type: str) -> pd.Series | None:
    """Appel Open-Meteo ERA5 reanalysis à 100m."""
    try:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat, "longitude": lon,
                "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
                "hourly": "wind_speed_100m",
                "wind_speed_unit": "ms",
                "timezone": "Europe/Paris",
            },
            timeout=45,
        )
        r.raise_for_status()
        data = r.json()
        ws   = np.array(data["hourly"]["wind_speed_100m"], dtype=float)
        cf   = _power_curve(ws, wind_type)
        s    = pd.Series(cf, index=pd.to_datetime(data["hourly"]["time"]), name=name)
        s.index.name = "datetime"
        dr   = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
        s    = s.resample("h").mean().reindex(dr).interpolate("time").ffill().bfill()
        print(f"  [OPEN-METEO] {name:<22} CF={s.mean():.3f}")
        return s
    except requests.exceptions.ConnectionError:
        print(f"  [WARN] Open-Meteo inaccessible → synthétique pour {name}")
        return None
    except Exception as e:
        print(f"  [WARN] Open-Meteo erreur {name} : {e} → synthétique")
        return None


def _synthetic_wind(lat: float, lon: float, year: int,
                     name: str, wind_type: str) -> pd.Series:
    """
    Profil éolien synthétique calibré sur latitude et type.
    CF de base offshore: ~0.43 → 0.30 selon latitude.
    Saisonnalité contra-cyclique (pic hivernal = contra-solaire).
    Simulation par blocs de 6h (persistance météorologique).
    """
    dr  = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    doy = dr.dayofyear.values

    cf_base = (float(np.clip(0.43 - 0.003 * (lat - 47.0), 0.30, 0.48))
               if wind_type == "offshore"
               else float(np.clip(0.32 - 0.004 * (lat - 47.0), 0.18, 0.35)))

    seasonal = 1.0 + 0.30 * np.cos(2 * np.pi * (doy - 15) / 365)

    rng    = np.random.default_rng(seed=int(abs(lon) * 100 + year))
    n_blk  = len(dr) // 6 + 1
    blocks = rng.beta(2, 2.5, n_blk) * cf_base * 2
    cf_raw = np.repeat(blocks, 6)[: len(dr)]
    cf     = np.clip(cf_raw * seasonal + rng.normal(0, 0.03, len(dr)), 0, 1)

    s = pd.Series(cf, index=dr, name=name)
    s.index.name = "datetime"
    print(f"  [SYNTH] {name:<22} CF={cf.mean():.3f}")
    return s


def download_wind_site(name: str, lat: float, lon: float,
                        wind_type: str = "offshore",
                        year: int = METEO_YEAR,
                        force_synthetic: bool = False) -> pd.Series:
    """Télécharge ou génère le profil d'un site éolien, stocke dans wind_patterns.db."""
    db = DB_DIR / "wind_patterns.db"
    if not force_synthetic:
        existing = _read_column(db, "wind_patterns", "datetime", name)
        if existing is not None and len(existing) >= 8700:
            print(f"  [CACHE] {name}")
            return existing

    s = _openmeteo_fetch(lat, lon, year, name, wind_type) if not force_synthetic else None
    if s is None:
        s = _synthetic_wind(lat, lon, year, name, wind_type)

    dr   = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    vals = s.reindex(dr).fillna(0).values
    _upsert_column(db, "wind_patterns", "datetime", dr.astype(str).tolist(), name, vals)
    return s


def download_all_wind(year: int = METEO_YEAR,
                       force_synthetic: bool = False) -> None:
    print("\n── PROFILS ÉOLIENS (wind_patterns.db) " + "─" * 27)
    for name, lat, lon, wtype, *_ in WIND_SITES:
        download_wind_site(name, lat, lon, wtype, year, force_synthetic)
    print(f"[OK] wind_patterns.db → {DB_DIR / 'wind_patterns.db'}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PROFILS DE CHARGE — load_patterns.db
# ─────────────────────────────────────────────────────────────────────────────

def create_load_profiles(year: int = 2030,
                          real_load_csv: Path | None = None) -> None:
    """
    Génère 4 profils de charge industriels synthétiques + load_reel si CSV fourni.

    Profils (normalisés 0..1) :
      siderurgie : process continu H24, LF ~92% (four arc, électrolyseur H2)
      chimie      : continu en semaine, shutdown WE, LF ~78%
      papier      : peak journée + creux nuit + arrêt WE, LF ~72%
      agroalim    : saisonnalité forte été + fêtes, LF ~68%
    """
    print("\n── PROFILS DE CHARGE (load_patterns.db) " + "─" * 24)
    db  = DB_DIR / "load_patterns.db"
    dr  = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    n   = len(dr)
    rng = np.random.default_rng(42)

    h   = dr.hour.values
    dow = dr.dayofweek.values
    doy = dr.dayofyear.values
    is_we = dow >= 5

    # ── Sidérurgie : process continu, très peu de variation ──────────────────
    load_sider = np.clip(np.ones(n) * 0.92 + rng.normal(0, 0.015, n), 0.80, 1.00)

    # ── Chimie : continu semaine, shutdown WE ────────────────────────────────
    load_chimie = np.where(is_we, 0.55, 0.90).astype(float)
    load_chimie[(dow == 4) & (h >= 18)] = 0.70   # vendredi soir : rampe down
    load_chimie[(dow == 0) & (h <= 8)]  = 0.70   # lundi matin : rampe up
    load_chimie = np.clip(load_chimie + rng.normal(0, 0.025, n), 0.40, 1.00)

    # ── Papier : pic journée, creux nuit, WE réduit ──────────────────────────
    load_papier = np.where(is_we, 0.40,
                  np.where((h >= 6) & (h < 22), 0.95, 0.65)).astype(float)
    load_papier = np.clip(load_papier + rng.normal(0, 0.020, n), 0.30, 1.00)

    # ── Agroalimentaire : pic été + pic fêtes de fin d'année ─────────────────
    seasonal = 0.65 + 0.20 * np.sin(2 * np.pi * (doy - 100) / 365)
    seasonal  = np.where(dr.month.values == 12, np.minimum(seasonal + 0.15, 1.0), seasonal)
    daily     = np.where((h >= 7) & (h < 20), 1.15, 0.75)
    daily     = np.where(is_we, daily * 0.80, daily)
    load_agro = np.clip(seasonal * daily + rng.normal(0, 0.025, n), 0.30, 1.00)

    idx = dr.astype(str).tolist()
    for name, vals in [("siderurgie", load_sider), ("chimie", load_chimie),
                        ("papier",     load_papier), ("agroalim",  load_agro)]:
        _upsert_column(db, "load_patterns", "datetime", idx, name, vals)
        print(f"  [OK] {name:<12} LF={vals.mean():.3f}")

    # ── Profil réel client ────────────────────────────────────────────────────
    if real_load_csv and real_load_csv.exists():
        _import_real_load(real_load_csv, db, year)

    print(f"[OK] load_patterns.db → {db}")


def _import_real_load(csv_path: Path, db: Path, year: int) -> None:
    """Importe un CSV de charge réel (colonnes datetime + load_mw), normalise sur max=1."""
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python")
        dt_col   = next((c for c in df.columns if "date" in c.lower() or "time" in c.lower()), None)
        load_col = next((c for c in df.columns
                         if any(k in c.lower() for k in ["load", "mw", "conso", "puiss"])), None)
        if not dt_col or not load_col:
            print(f"  [WARN] Colonnes non reconnues dans {csv_path.name}")
            return
        df["dt"] = pd.to_datetime(df[dt_col], infer_datetime_format=True, errors="coerce")
        df = df.dropna(subset=["dt"])
        df = df[df["dt"].dt.year == year].copy()
        df[load_col] = pd.to_numeric(df[load_col], errors="coerce")
        s  = df.set_index("dt")[load_col].resample("h").mean()
        dr = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
        s  = s.reindex(dr).interpolate("time").ffill().bfill()
        s_norm = (s / s.max()).clip(0, 1)
        _upsert_column(db, "load_patterns", "datetime",
                       dr.astype(str).tolist(), "load_reel", s_norm.values)
        print(f"  [OK] load_reel importé (LF={s_norm.mean():.3f}, max={s.max():.1f} MW)")
    except Exception as e:
        print(f"  [WARN] Import load réel : {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CAPTURE RATES — capture_rates.db
# ─────────────────────────────────────────────────────────────────────────────

def compute_capture_rates(epex_year: int = METEO_YEAR,
                           epex_csv: Path | None = None,
                           force_synthetic: bool = False) -> pd.DataFrame:
    """
    Calcule le capture_rate_systeme pour tous les sites disponibles.

    Formule :
        CR_i = [Σ(Q_i(t) × spot(t)) / Σ Q_i(t)] / spot_moyen

    CR > 1 → parc produit préférentiellement aux heures à prix élevé (éolien hivernal).
    CR < 1 → cannibalisation (solaire aux heures de midi où les prix sont déprimés).

    Stocke dans database/capture_rates.db (table capture_rates).
    """
    print("\n── CAPTURE RATES (capture_rates.db) " + "─" * 28)

    epex      = load_or_generate_epex(epex_year, epex_csv, force_synthetic)
    spot_mean = epex.mean()
    print(f"  EPEX {epex_year} : moy={spot_mean:.2f} €/MWh")

    results   = []
    wind_type_map = {w[0]: w[3] for w in WIND_SITES}

    # APRÈS
    # Charger load_patterns une seule fois (siderurgie = charge sidérurgie type ArcelorMittal)
    load_db = DB_DIR / "load_patterns.db"
    load_df = None
    if load_db.exists():
        conn_l = sqlite3.connect(str(load_db))
        try:
            load_df = pd.read_sql("SELECT * FROM load_patterns", conn_l,
                                   index_col="datetime", parse_dates=["datetime"])
            conn_l.close()
            # ← AJOUTER CES LIGNES : réindexer le profil de charge sur l'année EPEX
            # La forme horaire (H24 × 365) est identique quelle que soit l'année
            if load_df is not None and not load_df.empty:
                dr_epex = pd.date_range(f"{epex_year}-01-01",
                                         f"{epex_year}-12-31 23:00", freq="h")
                load_arr = load_df.values[:len(dr_epex)]  # tronquer si besoin
                load_df  = pd.DataFrame(load_arr, index=dr_epex,
                                         columns=load_df.columns)
        except Exception:
            conn_l.close()

    for db_path, table, tech_default in [
        (DB_DIR / "solar_patterns.db", "solar_patterns", "solar"),
        (DB_DIR / "wind_patterns.db",  "wind_patterns",  "wind"),
    ]:
        if not db_path.exists():
            continue
        conn = sqlite3.connect(str(db_path))
        try:
            df = pd.read_sql(f"SELECT * FROM {table}", conn,
                             index_col="datetime", parse_dates=["datetime"])
            conn.close()
        except Exception:
            conn.close()
            continue

        for col in df.columns:
            cf = df[col].reindex(epex.index).fillna(0)
            total_vol = cf.sum()
            if total_vol <= 0 or spot_mean <= 0:
                continue
            cr   = (cf * epex).sum() / total_vol / spot_mean
            tech = wind_type_map.get(col, tech_default)

            # Corrélation avec la charge client (Pearson)
            corr_load = None
            if load_df is not None and "siderurgie" in load_df.columns:
                common = cf.index.intersection(load_df.index)
                if len(common) >= 8000:
                    q_l = cf.reindex(common).fillna(0).values
                    l_l = load_df["siderurgie"].reindex(common).fillna(0).values
                    if q_l.std() > 0 and l_l.std() > 0:
                        corr_load = round(float(np.corrcoef(q_l, l_l)[0, 1]), 4)

            # Risque de cannibalisation : % d'heures où le site produit avec spot < 0
            q_vals   = cf.values
            s_vals   = epex.reindex(cf.index).fillna(0).values
            neg_hours = int(((s_vals < 0) & (q_vals > 0.01)).sum())
            canni     = round(float(np.clip(neg_hours / max(len(q_vals), 1) * 10, 0, 1)), 4)

            print(f"  {col:<25} CR={cr:.4f}  corr={corr_load or 'N/A':<7}  canni={canni:.3f}  ({tech})")

            results.append({
                "site":                  col,
                "technology":            tech,
                "capture_rate_systeme":  round(float(cr), 4),
                "capture_price_eur_mwh": round(float(cr * spot_mean), 2),
                "cf_mean":               round(float(cf.mean()), 4),
                "epex_year":             epex_year,
                "correlation_load":      corr_load,
                "cannibalization_risk":  canni,
            })

    if not results:
        print("[WARN] Aucun profil dispo pour les capture rates")
        return pd.DataFrame()

    df_cr = pd.DataFrame(results)
    db    = DB_DIR / "capture_rates.db"
    conn  = sqlite3.connect(str(db))
    df_cr.to_sql("capture_rates", conn, if_exists="replace", index=False)
    conn.close()

    sol = df_cr[df_cr.technology == "solar"]["capture_rate_systeme"]
    eol = df_cr[df_cr.technology.isin(["offshore", "onshore"])]["capture_rate_systeme"]
    print(f"\n[OK] {len(df_cr)} sites  |  "
          f"CR solaire moy={sol.mean():.3f}  |  CR éolien moy={eol.mean():.3f}")
    print(f"     → {db}")
    return df_cr


# ─────────────────────────────────────────────────────────────────────────────
# 7. WIND GRID FRANCE — wind_grid_france.xlsx
# ─────────────────────────────────────────────────────────────────────────────

def create_wind_grid_france(output_path: Path | None = None) -> pd.DataFrame:
    """
    Crée wind_grid_france.xlsx avec 12 parcs éoliens (6 offshore + 6 onshore).
    Colonnes attendues par FranceGridUtils.py :
      nom_parc, admin_boundaries, wind_type, LCOE, capacity, lat, lon,
      DT_m, cf_expected, project_status, rec_weight
    """
    print("\n── WIND GRID FRANCE (wind_grid_france.xlsx) " + "─" * 20)
    output_path = output_path or (DB_DIR / "wind_grid_france.xlsx")

    parks = [
        # ── Offshore posé (opérationnels ou en construction) ─────────────────
        dict(nom_parc="Dunkerque",             admin_boundaries="Hauts-de-France",
             wind_type="offshore",    LCOE=58.0,  capacity=600,
             lat=51.10, lon=2.20,  DT_m=10_000, cf_expected=0.43,
             project_status="operating",     rec_weight=1.0),
        dict(nom_parc="Saint-Nazaire",         admin_boundaries="Pays de la Loire",
             wind_type="offshore",    LCOE=63.0,  capacity=480,
             lat=47.20, lon=-2.50, DT_m=12_000, cf_expected=0.39,
             project_status="operating",     rec_weight=1.0),
        dict(nom_parc="Fécamp",                admin_boundaries="Normandie",
             wind_type="offshore",    LCOE=68.0,  capacity=500,
             lat=49.80, lon=0.50,  DT_m=15_000, cf_expected=0.37,
             project_status="operating",     rec_weight=1.0),
        dict(nom_parc="Courseulles-sur-Mer",   admin_boundaries="Normandie",
             wind_type="offshore",    LCOE=70.0,  capacity=450,
             lat=49.40, lon=-0.50, DT_m=10_000, cf_expected=0.36,
             project_status="construction",  rec_weight=1.0),
        dict(nom_parc="Saint-Brieuc",          admin_boundaries="Bretagne",
             wind_type="offshore",    LCOE=72.0,  capacity=496,
             lat=48.60, lon=-2.80, DT_m=16_000, cf_expected=0.35,
             project_status="operating",     rec_weight=1.0),
        # ── Offshore flottant (futur) ─────────────────────────────────────────
        dict(nom_parc="Golfe du Lion Flottant", admin_boundaries="Occitanie",
             wind_type="offshore_floating", LCOE=95.0,  capacity=250,
             lat=43.10, lon=4.20,  DT_m=20_000, cf_expected=0.38,
             project_status="planned",       rec_weight=1.0),
        # ── Onshore ──────────────────────────────────────────────────────────
        dict(nom_parc="Parc HdF Nord",         admin_boundaries="Hauts-de-France",
             wind_type="onshore",     LCOE=55.0,  capacity=120,
             lat=50.50, lon=2.50,  DT_m=5_000,  cf_expected=0.33,
             project_status="operating",     rec_weight=1.0),
        dict(nom_parc="Parc Normandie Côtier", admin_boundaries="Normandie",
             wind_type="onshore",     LCOE=57.0,  capacity=90,
             lat=49.40, lon=0.80,  DT_m=8_000,  cf_expected=0.29,
             project_status="operating",     rec_weight=1.0),
        dict(nom_parc="Parc Bretagne Centrale",admin_boundaries="Bretagne",
             wind_type="onshore",     LCOE=58.0,  capacity=75,
             lat=48.10, lon=-3.00, DT_m=10_000, cf_expected=0.30,
             project_status="operating",     rec_weight=1.0),
        dict(nom_parc="Parc Pays de la Loire", admin_boundaries="Pays de la Loire",
             wind_type="onshore",     LCOE=56.0,  capacity=100,
             lat=47.50, lon=-1.00, DT_m=12_000, cf_expected=0.28,
             project_status="operating",     rec_weight=1.0),
        dict(nom_parc="Parc Occitanie",        admin_boundaries="Occitanie",
             wind_type="onshore",     LCOE=60.0,  capacity=80,
             lat=43.80, lon=3.00,  DT_m=15_000, cf_expected=0.26,
             project_status="operating",     rec_weight=1.0),
        dict(nom_parc="Parc Grand Est",        admin_boundaries="Grand Est",
             wind_type="onshore",     LCOE=54.0,  capacity=110,
             lat=49.00, lon=6.00,  DT_m=7_000,  cf_expected=0.27,
             project_status="operating",     rec_weight=1.0),
    ]

    df = pd.DataFrame(parks)
    df.to_excel(str(output_path), index=False)
    print(f"[OK] {len(df)} parcs (offshore+onshore) → {output_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8. TURPE — vérification / génération fallback
# ─────────────────────────────────────────────────────────────────────────────

def ensure_turpe_file(output_path: Path | None = None) -> Path:
    """
    Vérifie la présence de TURPE_france.xlsx.
    Si absent, génère un fichier de fallback compatible avec FranceGridUtils.py.

    Structure :
      timezone  : heure x mois → HC / HP
      season    : mois → Hiver / Été
      contract  : tarif → fees (€/kW/an)
      HTB1/HTB2/HTB3 : saison x (HP/HC) → énergie (€/MWh)
    """
    candidates = [
        output_path or DB_DIR / "TURPE_france.xlsx",
        BASE_DIR / "TURPE_france.xlsx",
    ]
    for p in candidates:
        if p and p.exists():
            try:
                xl = pd.ExcelFile(str(p))
                print(f"[OK] TURPE_france.xlsx trouvé : {p}  (onglets: {xl.sheet_names})")
                return p
            except Exception:
                pass

    # Génération fallback
    out = DB_DIR / "TURPE_france.xlsx"
    print(f"[WARN] TURPE_france.xlsx absent → génération fallback dans {out}")

    months_fr    = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    winter       = {"Jan","Feb","Mar","Nov","Dec"}
    timezone_df  = pd.DataFrame({"hours": list(range(24)),
                                  **{m: ["HC" if (h<6 or h>=22) else "HP" for h in range(24)]
                                     for m in months_fr}})
    season_df    = pd.DataFrame({"month": months_fr,
                                  "season": ["Hiver" if m in winter else "Ete" for m in months_fr]})
    contract_df  = pd.DataFrame({"tarif": ["HTB1","HTB2","HTB3"],
                                  "fees":  [5.5,   7.5,   9.2]}).set_index("tarif")
    htb3 = pd.DataFrame({"Hiver": [5.2, 1.8], "Ete": [1.0, 0.5]}, index=["HP","HC"])
    htb2 = pd.DataFrame({"Hiver": [4.5, 1.5], "Ete": [0.9, 0.4]}, index=["HP","HC"])
    htb1 = pd.DataFrame({"Hiver": [3.8, 1.2], "Ete": [0.8, 0.3]}, index=["HP","HC"])

    with pd.ExcelWriter(str(out), engine="openpyxl") as w:
        timezone_df.to_excel(w, sheet_name="timezone", index=False)
        season_df.to_excel(w, sheet_name="season",   index=False)
        contract_df.to_excel(w, sheet_name="contract")
        htb3.to_excel(w, sheet_name="HTB3")
        htb2.to_excel(w, sheet_name="HTB2")
        htb1.to_excel(w, sheet_name="HTB1")

    print(f"[OK] TURPE fallback généré → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None) -> None:
    if args is None:
        class _Defaults:
            offline    = False
            epex       = None
            client_lat = None
            client_lon = None
            year       = METEO_YEAR
            load_csv   = None
        args = _Defaults()

    print("=" * 65)
    print("FRENCH PPA MODEL — Initialisation des données")
    print("=" * 65)
    print(f"Mode     : {'OFFLINE' if args.offline else 'ONLINE (PVGIS + Open-Meteo)'}")
    print(f"Année météo : {args.year}")

    DB_DIR.mkdir(parents=True, exist_ok=True)
    GIS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Grid France CSV
    print("\n── 1. GRID FRANCE CSV ─────────────────────────────────────")
    create_grid_france_csv()

    # 2. EPEX spot
    print("\n── 2. EPEX SPOT HORAIRE ───────────────────────────────────")
    load_or_generate_epex(args.year, Path(args.epex) if args.epex else None, args.offline)

    # 3. Profils solaires
    download_all_solar(args.year, getattr(args,"client_lat",None),
                       getattr(args,"client_lon",None), args.offline)

    # 4. Profils éoliens
    download_all_wind(args.year, args.offline)

    # 5. Profils de charge
    create_load_profiles(year=2030,
                         real_load_csv=Path(args.load_csv) if getattr(args,"load_csv",None) else None)

    # 6. Capture rates
    compute_capture_rates(args.year, Path(args.epex) if args.epex else None, args.offline)

    # 7. Wind grid
    create_wind_grid_france()

    # 8. TURPE
    print("\n── 8. TURPE ───────────────────────────────────────────────")
    ensure_turpe_file()

    # Résumé
    print("\n" + "=" * 65)
    print("RÉSUMÉ DES FICHIERS GÉNÉRÉS")
    print("=" * 65)
    files = ["grid_france.csv", "epex_profiles.db", "solar_patterns.db",
             "wind_patterns.db", "load_patterns.db", "capture_rates.db",
             "wind_grid_france.xlsx", "TURPE_france.xlsx"]
    for f in files:
        p    = DB_DIR / f
        size = f"{p.stat().st_size/1024:.0f} KB" if p.exists() else "ABSENT"
        icon = "✓" if p.exists() else "✗"
        print(f"  {icon} {f:<32} {size}")
    print(f"\nRépertoire : {DB_DIR}")
    print("\n→ Prochaine étape : FranceGridUtils.py — compute_site_metrics()")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Télécharge/génère les données French PPA Model"
    )
    parser.add_argument("--offline",     action="store_true",
                        help="Tout synthétique, pas d'appel réseau")
    parser.add_argument("--epex",        type=str, default=None,
                        help="Chemin CSV EPEX ODRE (optionnel)")
    parser.add_argument("--client-lat",  type=float, default=None,
                        help="Latitude site client (profil solaire on-site)")
    parser.add_argument("--client-lon",  type=float, default=None,
                        help="Longitude site client")
    parser.add_argument("--year",        type=int, default=METEO_YEAR,
                        help=f"Année météo de référence (défaut: {METEO_YEAR})")
    parser.add_argument("--load-csv",    type=str, default=None,
                        help="CSV profil de charge réel client")
    main(parser.parse_args())
