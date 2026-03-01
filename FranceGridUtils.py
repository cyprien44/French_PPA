"""
FranceGridUtils.py
==================
Utilitaires de données marché et métriques de sites pour le French PPA Model.

Remplace KEPCOutils.py — adapté au contexte économique français.

Modules :
  1. TURPE HTB          — chargement + calcul composante énergie horaire
  2. Profil de coût réseau — process_france_grid_data() (équivalent process_kepco_data)
  3. Projection multi-annuelle — multiyear_pricing_france()
  4. Garanties d'Origine — create_go_grid()
  5. Métriques de site   — compute_site_metrics() : capture rate, corrélation, LCOE
  6. Capture price effective — avec clauses curtailment/prix négatifs
  7. Screening de sites  — score composite marge × corrélation
  8. Grid info France    — trajectoires CAPEX/CO2/ENR 2023-2050

Contexte économique France :
  - TURPE HTB fixé par la CRE (revue annuelle, ~+4-6%/an depuis 2020)
  - EPEX Spot Day-Ahead : marché de référence (RTE/EPEX SE)
  - TICFE : Taxe Intérieure sur la Consommation Finale d'Électricité
  - CTA : Contribution Tarifaire d'Acheminement
  - GO (Garanties d'Origine) : 5-15 €/MWh en 2024 (EEX)
  - EU ETS : ~60-70 €/tCO2 en 2024

Sources :
  TURPE : https://www.cre.fr/Electricite/Reseaux-d-electricite/Tarifs-d-acces
  EPEX  : https://odre.opendatasoft.com (prix-spot-da-horaires)
  CO2   : https://odre.opendatasoft.com (eco2mix-national-cons-def)
  GO    : https://www.eex.com/en/market-data/environmental-markets
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CHEMINS & CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

_HERE    = Path(__file__).resolve().parent
_DB_DIR  = _HERE / "database"

# TURPE 6 HTB2 MU (Moyenne Utilisation) — valeurs par défaut CRE Nov 2024
# Source : https://www.cre.fr (décision du 18 janvier 2024)
_TURPE_DEFAULTS = {
    "contract_fee_eur_kw_year": 7.1,   # Composante puissance souscrite HTB2 (€/kW/an)
    "HPSH": 8.924,   # HP Saison Haute   (€/MWh)
    "HCSH": 6.824,   # HC Saison Haute
    "HPSB": 5.354,   # HP Saison Basse
    "HCSB": 3.570,   # HC Saison Basse
    "HPTE": 11.444,  # HP Très Haute saison (pointe hiver extrême — rare)
    # Alias HP/HC pour rétro-compatibilité
    "HPH": 8.924, "HCH": 6.824, "HPE": 5.354, "HCE": 3.570,
    "ticfe": 0.5,            # TICFE (€/MWh) — taux industrie 2024
    "cta":   0.3,            # CTA (€/MWh)
    "gestion_eur_an": 9873.3, # Composante gestion annuelle (€/an)
}

_MONTHS_ABR = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# Saisons TURPE : Haute = nov-mars, Basse = avr-oct
_SEASON_DEFAULT = {
    "Jan":"Haute","Feb":"Haute","Mar":"Haute",
    "Apr":"Basse","May":"Basse","Jun":"Basse",
    "Jul":"Basse","Aug":"Basse","Sep":"Basse",
    "Oct":"Basse","Nov":"Haute","Dec":"Haute",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS INTERNES
# ─────────────────────────────────────────────────────────────────────────────

def _find_turpe_file(filepath: str | None = None) -> str | None:
    if filepath and os.path.exists(filepath):
        return filepath
    candidates = [
        str(_DB_DIR / "TURPE_france.xlsx"),
        str(_HERE / "TURPE_france.xlsx"),
        "database/TURPE_france.xlsx",
        "TURPE_france.xlsx",
    ]
    return next((c for c in candidates if os.path.exists(c)), None)


def _read_sqlite_series(db: Path, table: str, idx: str, col: str) -> pd.Series | None:
    """Lit une série depuis SQLite. Retourne None si absente."""
    if not db.exists():
        return None
    conn = sqlite3.connect(str(db))
    try:
        df = pd.read_sql(f"SELECT {idx},{col} FROM {table}", conn,
                         index_col=idx, parse_dates=[idx])
        conn.close()
        return df[col]
    except Exception:
        conn.close()
        return None


def _read_sqlite_all(db: Path, table: str, idx: str) -> pd.DataFrame | None:
    """Lit toutes les colonnes d'une table SQLite."""
    if not db.exists():
        return None
    conn = sqlite3.connect(str(db))
    try:
        df = pd.read_sql(f"SELECT * FROM {table}", conn,
                         index_col=idx, parse_dates=[idx])
        conn.close()
        return df
    except Exception:
        conn.close()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. TURPE HTB
# ─────────────────────────────────────────────────────────────────────────────

def load_turpe_params(turpe_filepath: str | None = None,
                      sheet: str = "HTB2",
                      version: str = "MU") -> dict:
    """
    Charge les paramètres TURPE depuis TURPE_france.xlsx.

    Paramètres :
        turpe_filepath : chemin vers TURPE_france.xlsx (auto-détecté si None)
        sheet          : "HTB1" | "HTB2" | "HTB3" selon tension de raccordement
        version        : "CU" | "MU" | "LU" (courte / moyenne / longue utilisation)

    Retourne un dict avec les clés attendues par process_france_grid_data().
    """
    found = _find_turpe_file(turpe_filepath)
    if not found:
        print("[TURPE] Fichier non trouvé — valeurs TURPE 6 HTB2 MU Nov 2024 par défaut")
        return _TURPE_DEFAULTS.copy()

    try:
        df = pd.read_excel(found, sheet_name=sheet, header=None)

        if sheet == "HTB3":
            for _, row in df.iterrows():
                if "soutirage" in str(row.iloc[0]).lower():
                    val = float(row.iloc[2]) * 10   # c€/kWh → €/MWh
                    d = _TURPE_DEFAULTS.copy()
                    d.update({k: val for k in ["HPSH","HCSH","HPSB","HCSB","HPTE",
                                                "HPH","HCH","HPE","HCE"]})
                    d["contract_fee_eur_kw_year"] = 0.0
                    print(f"[TURPE] {sheet} chargé → {val:.2f} €/MWh")
                    return d

        # HTB1 / HTB2 : trouver la section version (CU/MU/LU) Nov 2024
        ver_cols = {"CU": (2, 3), "MU": (6, 7), "LU": (10, 11)}
        b_col, c_col = ver_cols.get(version, (6, 7))

        for i, row in df.iterrows():
            if "b (€/kW/an)" in str(row.values):
                data = df.iloc[i+1:i+6].reset_index(drop=True)
                classes = ["HPTE","HPSH","HCSH","HPSB","HCSB"]
                params = _TURPE_DEFAULTS.copy()
                for idx, cls in enumerate(classes):
                    if idx < len(data):
                        try:
                            params[cls] = float(data.iloc[idx, c_col]) * 10
                        except (ValueError, IndexError):
                            pass
                # Lire la composante puissance depuis onglet contract
                try:
                    cdf = pd.read_excel(found, sheet_name="contract", index_col=0)
                    params["contract_fee_eur_kw_year"] = float(cdf.loc[sheet, "fees"])
                except Exception:
                    params["contract_fee_eur_kw_year"] = float(data.iloc[1, b_col])
                # Alias HP/HC
                params.update(HPH=params["HPSH"], HCH=params["HCSH"],
                               HPE=params["HPSB"], HCE=params["HCSB"])
                print(f"[TURPE] Chargé {found} — {sheet} {version} "
                      f"| HPSH={params['HPSH']:.2f} HCSH={params['HCSH']:.2f} €/MWh")
                return params

    except Exception as e:
        print(f"[TURPE] Erreur lecture {found} : {e} → valeurs par défaut")

    return _TURPE_DEFAULTS.copy()


def load_turpe_timezone(turpe_filepath: str | None = None) -> pd.DataFrame:
    """
    Charge la grille HP/HC par heure x mois depuis l'onglet 'timezone'.
    Retourne DataFrame (index=heure 0-23, colonnes=Jan..Dec, valeurs='HP'|'HC').
    """
    found = _find_turpe_file(turpe_filepath)
    fallback = pd.DataFrame(
        {m: ["HC" if (h < 6 or h >= 22) else "HP" for h in range(24)]
         for m in _MONTHS_ABR},
        index=range(24)
    )
    if not found:
        return fallback
    try:
        df = pd.read_excel(found, sheet_name="timezone", header=None)
        # Ligne d'en-tête contient "hours" ou "heure"
        for i, row in df.iterrows():
            if str(row.iloc[0]).strip().lower() in ("hours", "heure", "heures"):
                df.columns = [str(v).strip() for v in df.iloc[i]]
                df = df.iloc[i+1:].reset_index(drop=True)
                h_col = df.columns[0]
                df[h_col] = pd.to_numeric(df[h_col], errors="coerce")
                df = df.dropna(subset=[h_col]).astype({h_col: int})
                df = df.set_index(h_col)
                cols = [c for c in _MONTHS_ABR if c in df.columns]
                return df[cols]
    except Exception as e:
        print(f"[TURPE] timezone non chargé : {e}")
    return fallback


def load_turpe_season(turpe_filepath: str | None = None) -> dict:
    """
    Charge le mapping mois → saison depuis l'onglet 'season'.
    Retourne dict {mois_abrégé: 'Haute'|'Basse'}.
    """
    found = _find_turpe_file(turpe_filepath)
    if not found:
        return _SEASON_DEFAULT.copy()
    try:
        df = pd.read_excel(found, sheet_name="season", header=None)
        for i, row in df.iterrows():
            lbl = str(row.iloc[0]).strip().lower()
            if lbl in ("month", "mois"):
                df.columns = [str(v).strip() for v in df.iloc[i]]
                df = df.iloc[i+1:].dropna().reset_index(drop=True)
                c_m = [c for c in df.columns if "mois" in c.lower() or "month" in c.lower()][0]
                c_s = [c for c in df.columns if "saison" in c.lower() or "season" in c.lower()][0]
                return dict(zip(df[c_m].str.strip(), df[c_s].str.strip()))
    except Exception as e:
        print(f"[TURPE] season non chargé : {e}")
    return _SEASON_DEFAULT.copy()


def _compute_turpe_energy_hourly(dr: pd.DatetimeIndex,
                                  params: dict,
                                  turpe_filepath: str | None = None) -> np.ndarray:
    """
    Calcule la composante énergie TURPE HTB pour chaque heure du DatetimeIndex.
    Utilise grille HP/HC + mapping saisonnier depuis TURPE_france.xlsx.
    Week-end → toujours HC (pas de pointe facturée sur les jours non ouvrés).
    """
    tz     = load_turpe_timezone(turpe_filepath)
    season = load_turpe_season(turpe_filepath)
    out    = np.zeros(len(dr))

    for i, ts in enumerate(dr):
        m      = _MONTHS_ABR[ts.month - 1]
        s      = season.get(m, "Basse")
        is_we  = ts.weekday() >= 5
        try:
            hp_hc = tz.loc[ts.hour, m]
        except (KeyError, TypeError):
            hp_hc = "HP" if 8 <= ts.hour < 20 else "HC"
        if is_we:
            hp_hc = "HC"

        if s == "Haute" and hp_hc == "HP":
            out[i] = params.get("HPSH", params.get("HPH", 8.9))
        elif s == "Haute":
            out[i] = params.get("HCSH", params.get("HCH", 6.8))
        elif hp_hc == "HP":
            out[i] = params.get("HPSB", params.get("HPE", 5.4))
        else:
            out[i] = params.get("HCSB", params.get("HCE", 3.6))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROFIL DE COÛT RÉSEAU — process_france_grid_data()
# ─────────────────────────────────────────────────────────────────────────────

def process_france_grid_data(
    year: int,
    epex_filepath: str | None = None,
    turpe_filepath: str | None = None,
    turpe_sheet: str = "HTB2",
    turpe_version: str = "MU",
    turpe_params: dict | None = None,
    epex_db_year: int | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Construit le profil horaire du coût électricité réseau pour un industriel HTB.

    Coût total = EPEX Spot Day-Ahead + TURPE énergie + TICFE + CTA

    Paramètres :
        year          : année de modélisation
        epex_filepath : CSV EPEX ODRE (optionnel — si absent, utilise epex_profiles.db)
        turpe_filepath: TURPE_france.xlsx (auto-détecté si None)
        turpe_sheet   : "HTB1" | "HTB2" | "HTB3"
        turpe_version : "CU" | "MU" | "LU"
        turpe_params  : dict de paramètres TURPE (remplace le chargement fichier si fourni)
        epex_db_year  : année du profil EPEX dans epex_profiles.db (défaut = year)

    Retourne :
        temporal_df   : DataFrame horaire avec colonnes
                        [epex_spot, turpe_energy, ticfe, cta, rate, contract_fee]
        contract_fee  : composante puissance TURPE annualisée (€/kW/an)
    """
    if turpe_params is None:
        turpe_params = load_turpe_params(turpe_filepath, turpe_sheet, turpe_version)
    contract_fee = turpe_params["contract_fee_eur_kw_year"]

    dr = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")

    # ── Chargement du profil EPEX ─────────────────────────────────────────────
    epex = _load_epex_profile(year, epex_filepath, epex_db_year)

    # ── TURPE énergie horaire ─────────────────────────────────────────────────
    turpe_e = _compute_turpe_energy_hourly(dr, turpe_params, turpe_filepath)

    # ── Assemblage ────────────────────────────────────────────────────────────
    df = pd.DataFrame(index=dr)
    df.index.name = "datetime"
    df["epex_spot"]    = epex.reindex(dr).fillna(method="ffill").fillna(method="bfill")
    df["turpe_energy"] = turpe_e
    df["ticfe"]        = turpe_params.get("ticfe", 0.5)
    df["cta"]          = turpe_params.get("cta",   0.3)
    df["rate"]         = df["epex_spot"] + df["turpe_energy"] + df["ticfe"] + df["cta"]
    # Composante puissance ramenée à l'heure (en €/MWh·h pour la cohérence avec le code)
    df["contract_fee"] = (contract_fee * 1000) / len(dr)

    return df, contract_fee


def _load_epex_profile(year: int,
                        csv_path: str | None = None,
                        db_year: int | None = None) -> pd.Series:
    """
    Charge le profil EPEX horaire depuis :
      1. CSV ODRE fourni explicitement
      2. epex_profiles.db (cache SQLite)
      3. Profil synthétique de fallback
    """
    # CSV explicite
    if csv_path and os.path.exists(csv_path):
        s = _parse_epex_odre_csv(csv_path, year)
        if s is not None:
            return s

    # Base de données cache
    db = _DB_DIR / "epex_profiles.db"
    ref_year = db_year or year
    s = _read_sqlite_series(db, f"epex_{ref_year}", "datetime", "price_eur_mwh")
    if s is not None and len(s) >= 8700:
        dr = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
        # Réindexer sur l'année demandée (même profil horaire, autre année)
        if s.index.year[0] != year:
            s.index = dr[:len(s)]
        return s.reindex(dr).fillna(method="ffill").fillna(method="bfill")

    # Fallback synthétique
    print(f"[WARN] EPEX {year} non trouvé en cache — profil synthétique")
    return _synthetic_epex_fallback(year)


def _parse_epex_odre_csv(path: str, year: int) -> pd.Series | None:
    """Parse un CSV EPEX ODRE (sep=';', dec=',')."""
    try:
        df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        date_col  = next((c for c in df.columns if "date" in c.lower()), None)
        hour_col  = next((c for c in df.columns if "heure" in c.lower()), None)
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
        return s.reindex(dr).interpolate("time").ffill().bfill()
    except Exception as e:
        print(f"[WARN] Parsing EPEX CSV : {e}")
        return None


def _synthetic_epex_fallback(year: int, base: float = 65.0) -> pd.Series:
    """Profil EPEX synthétique minimal (pour éviter les crashs)."""
    dr = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    n  = len(dr)
    h  = dr.hour.values
    seasonal = 1.0 + 0.30 * np.cos(2 * np.pi * np.arange(n) / 8760)
    daily    = 1.0 + 0.20 * np.sin(2 * np.pi * (h - 6) / 24)
    rng      = np.random.default_rng(year)
    prices   = base * seasonal * daily * rng.lognormal(0, 0.10, n)
    return pd.Series(prices.round(2), index=dr, name="price_eur_mwh")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PROJECTION MULTI-ANNUELLE
# ─────────────────────────────────────────────────────────────────────────────

def multiyear_pricing_france(
    temporal_df: pd.DataFrame,
    contract_fee: float,
    start_year: int,
    num_years: int,
    rate_increase: float,
    annualised_contract: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Projection multi-annuelle des tarifs réseau français.

    Le taux rate_increase couvre l'escalade combinée :
      - TURPE : historiquement +4-6%/an depuis 2020 (source CRE)
      - EPEX  : hypothèse +2-3%/an réel (très volatile — approximation)
    Recommandation : utiliser rate_increase ∈ [0.02, 0.04] pour les scénarios centraux.

    Retourne :
        long_df      : DataFrame horaire sur num_years années
        contract_df  : DataFrame annuel avec la composante puissance escaladée
    """
    all_dfs        = []
    contract_rows  = []
    preset         = temporal_df.copy()
    preset.index   = preset.index.strftime("%m-%d %H:%M")

    # Ajouter le 29 février pour les années bissextiles
    feb28 = preset.loc[preset.index.str.startswith("02-28")]
    feb29 = feb28.copy()
    feb29.index = feb29.index.str.replace("02-28", "02-29")
    preset = pd.concat([preset, feb29])

    for offset in range(num_years):
        yr = start_year + offset
        dr = pd.date_range(f"{yr}-01-01", f"{yr}-12-31 23:00", freq="h")
        keys = dr.strftime("%m-%d %H:%M")

        esc      = (1 + rate_increase) ** offset
        year_df  = preset.reindex(keys).copy()
        year_df.index = dr
        year_df["rate"] = pd.to_numeric(year_df["rate"], errors="coerce") * esc

        yr_cf = contract_fee * esc
        if annualised_contract:
            year_df["contract_fee"] = (yr_cf * 1000) / len(dr)

        all_dfs.append(year_df)
        contract_rows.append({"year": yr, "rate": yr_cf})

    return pd.concat(all_dfs), pd.DataFrame(contract_rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4. GARANTIES D'ORIGINE
# ─────────────────────────────────────────────────────────────────────────────

def create_go_grid(
    start_year: int,
    end_year: int,
    initial_go_price_eur_mwh: float = 8.0,
    rate_increase: float = 0.0,
    go_type: str = "annual",
) -> pd.DataFrame:
    """
    Génère la trajectoire de prix des Garanties d'Origine (GO) françaises.

    Les GO sont l'équivalent français des RECs coréens, mais beaucoup moins chers
    car le mix électrique français est déjà fortement décarboné (nucléaire ~70%).

    Paramètres :
        initial_go_price_eur_mwh : prix initial (€/MWh). Valeurs 2024 :
            annual matching  → 5-8 €/MWh
            monthly matching → 8-12 €/MWh
            hourly / 24-7 CFE → 12-20 €/MWh
        rate_increase : évolution annuelle du prix GO (% décimal)
        go_type       : "annual" | "monthly" | "hourly" (granularité du matching)

    Sources :
        EEX Environmental Markets : https://www.eex.com
        AIB residual mix report   : https://www.aib-net.org
    """
    # Prime de granularité (plus le matching est fin, plus la GO est premium)
    granularity_mult = {"annual": 1.0, "monthly": 1.4, "hourly": 2.2}
    base = initial_go_price_eur_mwh * granularity_mult.get(go_type, 1.0)

    years = range(start_year, end_year + 1)
    values = {
        y: base * (1 + rate_increase) ** (y - start_year)
        for y in years
    }
    df = pd.DataFrame({"value": values})
    df.index.name = "year"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. MÉTRIQUES DE SITE — compute_site_metrics()
# ─────────────────────────────────────────────────────────────────────────────

def compute_site_metrics(
    site_col: str,
    technology: str,
    epex_year: int = 2020,
    load_col: str = "siderurgie",
    epex_csv: str | None = None,
    db_dir: Path | None = None,
) -> dict:
    """
    Calcule les métriques économiques et de corrélation d'un site candidat PPA.

    Métriques retournées :
        capture_rate_systeme  : CR = Σ(Q×spot) / (ΣQ × spot_moy)
            > 1 → produit aux heures chères (éolien hivernal)
            < 1 → cannibalisation (solaire aux heures creuses)
        capture_price         : CR × spot_moyen (€/MWh)
        cf_mean               : facteur de charge moyen (%)
        correlation_load      : corrélation de Pearson profil×charge client
        p50_annual_prod_mwh   : production P50 annuelle par MW installé (MWh/MW/an)
        best_match_month      : mois avec le meilleur matching profil×charge
        cannibalization_risk  : score 0-1 (1 = forte cannibalisation)

    Paramètres :
        site_col   : nom de la colonne dans solar_patterns.db ou wind_patterns.db
        technology : "solar" | "offshore" | "onshore" | "hybrid"
        epex_year  : année du profil EPEX de référence
        load_col   : profil de charge client ("siderurgie" | "chimie" | "papier" | "agroalim")
        epex_csv   : chemin CSV EPEX ODRE (optionnel)
        db_dir     : répertoire database (défaut: auto-détecté)
    """
    ddir = db_dir or _DB_DIR

    # ── Charger le profil de production ──────────────────────────────────────
    db_name = "solar_patterns.db" if technology == "solar" else "wind_patterns.db"
    prod_series = _read_sqlite_series(ddir / db_name,
                                       "solar_patterns" if technology == "solar" else "wind_patterns",
                                       "datetime", site_col)
    if prod_series is None:
        return {"error": f"Site '{site_col}' introuvable dans {db_name}"}

    # ── Charger le profil EPEX ────────────────────────────────────────────────
    epex = _load_epex_profile(epex_year, epex_csv)
    common_idx = prod_series.index.intersection(epex.index)
    if len(common_idx) < 8000:
        return {"error": f"Index commun trop court ({len(common_idx)} heures)"}

    Q    = prod_series.reindex(common_idx).fillna(0).values.astype(float)
    spot = epex.reindex(common_idx).values.astype(float)

    total_vol  = Q.sum()
    spot_mean  = spot.mean()
    if total_vol <= 0 or spot_mean <= 0:
        return {"error": "Volume ou spot moyen nul"}

    # ── Capture rate ─────────────────────────────────────────────────────────
    capture_price = float((Q * spot).sum() / total_vol)
    cr            = capture_price / spot_mean

    # ── Facteur de charge ─────────────────────────────────────────────────────
    cf_mean = float(Q.mean())

    # ── P50 production annuelle (MWh/MW installé/an) ──────────────────────────
    # On compte sur 8760h — fraction de l'année couverte = len/8760
    p50_mwh_mw = float(Q.sum() / (len(Q) / 8760))

    # ── Corrélation avec la charge client ────────────────────────────────────
    load_db = ddir / "load_patterns.db"
    load_s  = _read_sqlite_series(load_db, "load_patterns", "datetime", load_col)
    corr = np.nan
    if load_s is not None:
        common_load = prod_series.index.intersection(load_s.index)
        if len(common_load) >= 8000:
            q_l = prod_series.reindex(common_load).fillna(0).values.astype(float)
            l_l = load_s.reindex(common_load).fillna(0).values.astype(float)
            if q_l.std() > 0 and l_l.std() > 0:
                corr = float(np.corrcoef(q_l, l_l)[0, 1])

    # ── Risque de cannibalisation ─────────────────────────────────────────────
    # Heures où spot < 0 et Q > 0 (production pendant prix négatifs)
    neg_prod_hours = int(((spot < 0) & (Q > 0.01)).sum())
    neg_revenue    = float((Q * np.minimum(spot, 0)).sum())  # perte potentielle
    cannibalization_risk = float(np.clip(neg_prod_hours / max(len(Q), 1) * 10, 0, 1))

    # ── Meilleur mois de matching ─────────────────────────────────────────────
    dr_common = pd.DatetimeIndex(common_idx)
    monthly_q = pd.Series(Q, index=dr_common).resample("ME").mean()
    best_month = int(monthly_q.idxmax().month) if len(monthly_q) > 0 else 0

    return {
        "site":                   site_col,
        "technology":             technology,
        "cf_mean":                round(cf_mean, 4),
        "p50_mwh_per_mw_year":    round(p50_mwh_mw, 1),
        "capture_rate_systeme":   round(cr, 4),
        "capture_price_eur_mwh":  round(capture_price, 2),
        "epex_spot_mean":         round(spot_mean, 2),
        "correlation_load":       round(corr, 4) if not np.isnan(corr) else None,
        "neg_prod_hours":         neg_prod_hours,
        "neg_revenue_eur_per_mw": round(neg_revenue, 0),
        "cannibalization_risk":   round(cannibalization_risk, 4),
        "best_match_month":       best_month,
        "epex_year":              epex_year,
        "load_profile":           load_col,
    }


def compute_all_site_metrics(
    epex_year: int = 2020,
    load_col: str = "siderurgie",
    epex_csv: str | None = None,
    db_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Calcule les métriques pour tous les sites disponibles dans solar_patterns.db et wind_patterns.db.
    Retourne un DataFrame trié par score composite (capture_rate × corrélation).
    """
    ddir = db_dir or _DB_DIR
    results = []

    # Sites solaires
    sol_df = _read_sqlite_all(ddir / "solar_patterns.db", "solar_patterns", "datetime")
    if sol_df is not None:
        for col in sol_df.columns:
            m = compute_site_metrics(col, "solar", epex_year, load_col, epex_csv, ddir)
            if "error" not in m:
                results.append(m)

    # Sites éoliens
    wind_df = _read_sqlite_all(ddir / "wind_patterns.db", "wind_patterns", "datetime")
    if wind_df is not None:
        # Déduire le type offshore/onshore depuis le nom de colonne
        for col in wind_df.columns:
            tech = "offshore" if "off" in col.lower() else "onshore"
            m = compute_site_metrics(col, tech, epex_year, load_col, epex_csv, ddir)
            if "error" not in m:
                results.append(m)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    # Score composite : capture_rate × max(0, corrélation_load)
    df["score_composite"] = (
        df["capture_rate_systeme"] *
        df["correlation_load"].clip(lower=0).fillna(0)
    ).round(4)
    return df.sort_values("score_composite", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. CAPTURE PRICE EFFECTIVE — avec clauses curtailment / prix négatifs
# ─────────────────────────────────────────────────────────────────────────────

def compute_effective_capture_price(
    prod_profile: pd.Series,
    spot_profile: pd.Series,
    neg_price_floor: float | None = 0.0,
    curtailment_threshold: float | None = None,
    grid_curtailment_rate: float = 0.0,
    sleeving_fee: float = 0.0,
    basis_risk: float = 0.0,
) -> dict:
    """
    Calcule le capture price effectif après application des clauses contractuelles.

    Le capture_price_standard mesure la valeur marchande de la production.
    Le capture_price_effective reflète le revenu net réellement reçu par le producteur
    après prise en compte des clauses de protection et des coûts.

    Paramètres (correspondent aux paramètres de scenario_defaults.xlsx) :
        neg_price_floor      : plancher de prix pour le settlement (€/MWh)
                               0 = zero floor (standard 2024+)
                               None = pas de protection
        curtailment_threshold: seuil de curtailment volontaire (€/MWh)
                               Si spot < seuil → production = 0
                               None = pas de curtailment
        grid_curtailment_rate: taux de curtailment réseau TSO (fraction, ex: 0.015)
        sleeving_fee         : coût du fournisseur intermédiaire (€/MWh)
        basis_risk           : écart structurel site vs hub (€/MWh, négatif = pénalité)

    Retourne un dict avec :
        capture_price_standard  : Σ(Q×spot) / ΣQ  (sans aucune clause)
        capture_price_effective : avec toutes les clauses appliquées
        effective_multiplier    : ratio effective / standard
        neg_hours               : heures de production avec spot < 0
        curtailed_hours         : heures curtailed (volontaire + réseau)
        revenue_impact_pct      : impact total des clauses sur le revenu (%)
    """
    # Aligner les index
    common = prod_profile.index.intersection(spot_profile.index)
    Q     = prod_profile.reindex(common).fillna(0).values.astype(float)
    spot  = spot_profile.reindex(common).values.astype(float)

    total_vol_raw = Q.sum()
    if total_vol_raw <= 0:
        return {"error": "Volume de production nul"}

    # ── 1. Capture price standard (sans clause) ───────────────────────────────
    cp_std = float((Q * spot).sum() / total_vol_raw)

    # ── 2. Curtailment réseau TSO (involontaire) ──────────────────────────────
    Q_eff = Q * (1.0 - grid_curtailment_rate)

    # ── 3. Curtailment volontaire (heures sous seuil) ─────────────────────────
    curtailed_vol = 0.0
    n_curtailed   = 0
    if curtailment_threshold is not None:
        mask_curtail = spot < curtailment_threshold
        curtailed_vol = Q_eff[mask_curtail].sum()
        n_curtailed   = int(mask_curtail.sum())
        Q_eff[mask_curtail] = 0.0

    # ── 4. Settlement avec plancher de prix ──────────────────────────────────
    spot_settlement = spot.copy()
    if neg_price_floor is not None:
        spot_settlement = np.maximum(spot_settlement, neg_price_floor)

    # Ajustement basis risk (le producteur reçoit prix_hub - basis_risk)
    spot_settlement = spot_settlement - basis_risk

    # ── 5. Calcul du revenu effectif ──────────────────────────────────────────
    total_vol_eff = Q_eff.sum()
    if total_vol_eff <= 0:
        cp_eff = 0.0
    else:
        revenue_gross = (Q_eff * spot_settlement).sum()
        # Déduire le sleeving fee sur le volume effectivement livré
        revenue_net   = revenue_gross - sleeving_fee * total_vol_eff
        cp_eff = float(revenue_net / total_vol_eff)

    # ── 6. Métriques annexes ──────────────────────────────────────────────────
    n_neg         = int(((spot < 0) & (Q > 0.01)).sum())
    impact_pct    = (cp_eff - cp_std) / abs(cp_std) * 100 if cp_std != 0 else 0.0

    return {
        "capture_price_standard":  round(cp_std, 2),
        "capture_price_effective": round(cp_eff, 2),
        "effective_multiplier":    round(cp_eff / cp_std, 4) if cp_std != 0 else 0,
        "neg_hours":               n_neg,
        "curtailed_hours_voluntary": n_curtailed,
        "curtailed_vol_mwh_per_mw": round(curtailed_vol, 1),
        "revenue_impact_pct":      round(impact_pct, 2),
        "sleeving_fee":            sleeving_fee,
        "basis_risk":              basis_risk,
        "grid_curtailment_rate":   grid_curtailment_rate,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. SCREENING DE SITES
# ─────────────────────────────────────────────────────────────────────────────

def screen_sites(
    site_metrics_df: pd.DataFrame,
    min_cf: float = 0.10,
    min_capture_rate: float = 0.80,
    min_correlation: float | None = None,
    max_cannibalization: float = 0.3,
    technologies: list[str] | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Filtre et classe les sites candidats selon des critères économiques.

    Logique de screening (cohérente avec le marché PPA France 2024) :
        1. Filtre sur CF minimum (viabilité technique du projet)
        2. Filtre sur capture rate minimum (viabilité économique)
        3. Filtre sur risque de cannibalisation (sécurité pour l'acheteur)
        4. Filtre optionnel sur corrélation avec la charge (matching besoins client)
        5. Filtre optionnel sur la technologie
        6. Classement par score composite et sélection des top_n

    Paramètres :
        min_cf             : CF minimum (défaut 10% — solaire Nord marginalement viable)
        min_capture_rate   : CR minimum (défaut 0.80 — élimine les sites très pénalisés)
        min_correlation    : corrélation load minimum (None = pas de filtre)
        max_cannibalization: risque max de cannibalisation (défaut 0.30)
        technologies       : liste de technologies à garder (None = toutes)
        top_n              : retourner seulement les N meilleurs sites

    Retourne un DataFrame avec les sites passant tous les filtres, triés par score.
    """
    df = site_metrics_df.copy()

    if df.empty:
        return df

    # Appliquer les filtres
    mask = pd.Series(True, index=df.index)
    if min_cf:
        mask &= df["cf_mean"] >= min_cf
    if min_capture_rate:
        mask &= df["capture_rate_systeme"] >= min_capture_rate
    if min_correlation is not None and "correlation_load" in df.columns:
        mask &= df["correlation_load"].fillna(0) >= min_correlation
    if max_cannibalization and "cannibalization_risk" in df.columns:
        mask &= df["cannibalization_risk"] <= max_cannibalization
    if technologies:
        mask &= df["technology"].isin(technologies)

    screened = df[mask].copy()

    # Re-calculer le score composite avec pondération configurable
    # Score = 0.5 × capture_rate + 0.3 × corrélation_load + 0.2 × (1 - cannibalization)
    screened["score_composite"] = (
        0.50 * screened["capture_rate_systeme"].clip(0, 2) / 1.5 +
        0.30 * screened["correlation_load"].fillna(0).clip(-1, 1) / 1.0 +
        0.20 * (1 - screened["cannibalization_risk"].fillna(0).clip(0, 1))
    ).round(4)

    screened = screened.sort_values("score_composite", ascending=False)

    if top_n:
        screened = screened.head(top_n)

    n_filtered = len(site_metrics_df) - len(screened)
    print(f"[SCREENING] {len(screened)} sites retenus ({n_filtered} éliminés)")
    if not screened.empty:
        print(f"  Top 3 : {screened['site'].head(3).tolist()}")
        print(f"  Score max : {screened['score_composite'].max():.4f}")

    return screened.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8. GRID INFO FRANCE — trajectoires 2023-2050
# ─────────────────────────────────────────────────────────────────────────────

def build_france_grid_info(
    start_year: int = 2023,
    end_year: int = 2050,
    eco2mix_filepath: str | None = None,
    grid_csv_path: str | None = None,
) -> pd.DataFrame:
    """
    Charge ou construit les trajectoires CAPEX/CO2/ENR pour la France.

    Priorité :
      1. grid_csv_path fourni explicitement
      2. database/grid_france.csv généré par download_france_data.py
      3. Valeurs calculées (trajectoires de référence)

    Colonnes :
      solar_capex, solar_opex, wind_onshore_capex, wind_onshore_opex,
      wind_offshore_capex, wind_offshore_opex, bess_capex_per_mwh,
      co2_intensity_g_kwh, ren_share, epex_spot_mean,
      solar_cf_ref, wind_onshore_cf_ref, wind_offshore_cf_ref

    Sources :
      CAPEX  : IRENA Renewable Power Generation Costs 2023
      CO2    : RTE eco2mix + trajectoire PPE
      EPEX   : RTE Bilan Électrique 2024 + hypothèse centrale +2%/an
      ENR    : RTE Futurs Énergétiques 2050 — scénario central
    """
    years = list(range(start_year, end_year + 1))
    n     = len(years)
    yr    = np.array(years)

    # ── Charger grid_france.csv si disponible ─────────────────────────────────
    for candidate in [grid_csv_path, str(_DB_DIR / "grid_france.csv")]:
        if candidate and os.path.exists(candidate):
            try:
                df = pd.read_csv(candidate, index_col="year")
                df = df.reindex(years)
                # Interpoler les années manquantes
                df = df.interpolate("index")
                # Extrapoler si nécessaire
                df = df.ffill().bfill()
                print(f"[GRID] Chargé depuis {candidate}")
                return df
            except Exception as e:
                print(f"[WARN] Erreur lecture grid_france.csv : {e}")

    # ── Calcul de référence ───────────────────────────────────────────────────
    print("[GRID] Calcul trajectoires de référence (grid_france.csv absent)")

    solar_capex    = np.maximum(700_000 * np.exp(-0.028 * (yr - 2023)), 300_000)
    wind_on_capex  = np.maximum(1_500_000 * np.exp(-0.010 * (yr - 2023)), 1_100_000)
    wind_off_capex = np.maximum(3_000_000 * np.exp(-0.022 * (yr - 2023)), 1_600_000)
    bess_capex     = np.maximum(200_000 * np.exp(-0.035 * (yr - 2023)), 60_000)
    co2            = np.maximum(46.0 * np.exp(-0.045 * (yr - 2023)), 8.0)
    ren_share      = 0.29 + (0.80 - 0.29) * (1 - np.exp(-0.060 * (yr - 2023)))
    epex           = np.where(yr <= 2024,
                              np.interp(yr, [2023, 2024], [96.0, 65.0]),
                              65.0 * (1.02 ** (yr - 2024)))

    # Charger eco2mix réel si disponible
    if eco2mix_filepath and os.path.exists(eco2mix_filepath):
        co2, ren_share = _load_eco2mix(eco2mix_filepath, years, co2, ren_share)

    df = pd.DataFrame({
        "solar_capex":          solar_capex.round(0),
        "solar_opex":           (solar_capex * 0.018).round(0),
        "wind_onshore_capex":   wind_on_capex.round(0),
        "wind_onshore_opex":    (wind_on_capex * 0.025).round(0),
        "wind_offshore_capex":  wind_off_capex.round(0),
        "wind_offshore_opex":   (wind_off_capex * 0.030).round(0),
        "bess_capex_per_mwh":   bess_capex.round(0),
        "co2_intensity_g_kwh":  co2.round(2),
        "ren_share":            ren_share.round(4),
        "epex_spot_mean":       epex.round(2),
        "solar_cf_ref":         np.full(n, 0.148),
        "wind_onshore_cf_ref":  np.full(n, 0.280),
        "wind_offshore_cf_ref": np.full(n, 0.400),
    }, index=years)
    df.index.name = "year"
    return df


def _load_eco2mix(filepath: str, years: list,
                   co2_default: np.ndarray, ren_default: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Charge les valeurs historiques eco2mix ODRE pour remplacer les estimations."""
    try:
        df = pd.read_csv(filepath, sep=";", parse_dates=["Date - Heure"])
        df["year"] = df["Date - Heure"].dt.year
        ann = df.groupby("year").agg({
            "Taux de CO2 (g/kWh)": "mean",
            "Taux d'EnR (%)":      "mean",
        })
        co2_out, ren_out = [], []
        for i, y in enumerate(years):
            if y in ann.index:
                co2_out.append(ann.loc[y, "Taux de CO2 (g/kWh)"])
                ren_out.append(ann.loc[y, "Taux d'EnR (%)"] / 100)
            else:
                co2_out.append(co2_default[i])
                ren_out.append(ren_default[i])
        print(f"[GRID] eco2mix chargé depuis {filepath}")
        return np.array(co2_out), np.array(ren_out)
    except Exception as e:
        print(f"[WARN] eco2mix non chargé : {e}")
        return co2_default, ren_default


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES PUBLIQUES (compatibilité modules aval)
# ─────────────────────────────────────────────────────────────────────────────

def get_epex_series(year: int, csv_path: str | None = None,
                     db_dir: Path | None = None) -> pd.Series:
    """Accès public au profil EPEX horaire."""
    ddir = db_dir or _DB_DIR
    return _load_epex_profile(year, csv_path)


def load_production_profile(site_col: str, technology: str,
                              db_dir: Path | None = None) -> pd.Series | None:
    """
    Charge le profil de production normalisé [0..1] d'un site depuis SQLite.

    Retourne None si le site est introuvable.
    """
    ddir = db_dir or _DB_DIR
    if technology == "solar":
        return _read_sqlite_series(ddir / "solar_patterns.db",
                                    "solar_patterns", "datetime", site_col)
    else:
        return _read_sqlite_series(ddir / "wind_patterns.db",
                                    "wind_patterns", "datetime", site_col)


def load_capture_rates_db(db_dir: Path | None = None) -> pd.DataFrame | None:
    """Charge le DataFrame de capture rates précalculés."""
    ddir = db_dir or _DB_DIR
    db   = ddir / "capture_rates.db"
    if not db.exists():
        return None
    conn = sqlite3.connect(str(db))
    try:
        df = pd.read_sql("SELECT * FROM capture_rates", conn)
        conn.close()
        return df
    except Exception:
        conn.close()
        return None


def get_turpe_annual_cost_eur_per_mw(
    peak_demand_mw: float,
    turpe_params: dict | None = None,
    turpe_filepath: str | None = None,
    turpe_sheet: str = "HTB2",
    num_years: int = 1,
    rate_increase: float = 0.04,
) -> pd.DataFrame:
    """
    Calcule la facture TURPE annuelle pour une puissance souscrite donnée.

    Utile pour chiffrer l'économie TURPE d'un PPA on-site (exemption totale)
    vs off-site (TURPE toujours dû).

    Paramètres :
        peak_demand_mw : puissance souscrite en MW
        num_years      : nombre d'années de projection
        rate_increase  : escalade annuelle du TURPE (défaut 4%/an historique)

    Retourne un DataFrame par année avec le coût total TURPE (€/an).
    """
    if turpe_params is None:
        turpe_params = load_turpe_params(turpe_filepath, turpe_sheet)

    base_fee    = turpe_params["contract_fee_eur_kw_year"] * peak_demand_mw * 1000
    rows = []
    for offset in range(num_years):
        yr  = 2024 + offset
        esc = (1 + rate_increase) ** offset
        rows.append({
            "year":              yr,
            "turpe_fee_eur_an":  round(base_fee * esc, 0),
            "escalation_factor": round(esc, 4),
        })
    df = pd.DataFrame(rows).set_index("year")
    return df
