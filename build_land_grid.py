"""
build_land_grid.py
==================
Construction automatique de la grille de terrains candidats pour le modèle PPA France.

Remplace le fichier manuel gisdata/db_semippa.gpkg du modèle coréen.

À partir du lat/lon du site client, ce script :
  1. Interroge CORINE Land Cover via IGN Géoplateforme WFS  (data.geopf.fr)
  2. Interroge DVF via API CEREMA open data              (apidf-preprod.cerema.fr)
  3. Filtre les zones protégées via WFS Carmen/INPN      (ws.carmencarto.fr)
  4. Construit le GeoDataFrame compatible avec costutils.py
  5. Sauvegarde en .gpkg + .csv + carte HTML interactive

Usage :
    python build_land_grid.py --lat 50.93 --lon 2.38 --buffer 30
    python build_land_grid.py --test          ← test sans internet (données synthétiques)

Dépendances :
    pip install geopandas shapely requests pandas numpy
"""

import math
import json
import argparse
from pathlib import Path
from typing import Optional
import os as _os_ap

import requests
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import box, shape
    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    print("[WARN] geopandas/shapely non disponibles — mode dégradé (DataFrame sans géométrie)")


# ==============================================================================
# CONSTANTES
# ==============================================================================

# Classes CORINE Land Cover compatibles ENR solaire
# (code: (label, dispo_pv_standard, dispo_agrivol, eligible_agrivol))
CLC_SOLAR_CLASSES = {
    211: ("Terres arables",                         0.80, 0.10, True),
    212: ("Périmètres irrigués",                    0.60, 0.05, True),
    213: ("Rizières",                               0.50, 0.03, True),
    221: ("Vignobles",                              0.00, 0.00, False),  # protégés AOP
    222: ("Vergers",                                0.40, 0.08, True),
    231: ("Prairies",                               0.85, 0.10, True),
    241: ("Cultures annuelles associées",           0.70, 0.08, True),
    242: ("Systèmes culturaux complexes",           0.75, 0.10, True),
    243: ("Surfaces agricoles + espaces naturels",  0.65, 0.08, True),
    321: ("Pelouses et pâturages naturels",         0.80, 0.05, True),
    333: ("Végétation clairsemée",                  0.85, 0.00, False),
    131: ("Décharges / friches",                    0.90, 0.00, False),
    132: ("Chantiers abandonnés",                   0.85, 0.00, False),
    133: ("Espaces en construction",                0.70, 0.00, False),
    121: ("Zones industrielles",                    0.50, 0.00, False),
}

CLC_EXCLUDED = {
    111, 112, 141, 142,     # tissu urbain, espaces verts
    311, 312, 313,          # forêts
    411, 412,               # zones humides
    511, 512, 521, 523,     # eau, mer
    122,                    # routes / voies ferrées
}

# Couleurs HTML par classe CLC pour la carte interactive
CLC_COLORS = {
    211: "#f5d07a", 212: "#f5d07a", 213: "#f5d07a",
    231: "#a8d08d", 241: "#c6e0b4", 242: "#c6e0b4", 243: "#c6e0b4",
    321: "#d9ead3", 333: "#ffe599",
    131: "#b4a7d6", 132: "#b4a7d6", 133: "#b4a7d6",
    121: "#ea9999",
}

# Coût câblage HTB France (€/MW/m) — barème RTE/ENEDIS 2024
CABLE_EUR_PER_MW_PER_M = 35.0
MIN_AREA_M2 = 1_000    # 0.1 ha minimum (RPG peut avoir de petites parcelles)


# ==============================================================================
# 1. GÉOMÉTRIE UTILITAIRES
# ==============================================================================

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def get_bbox(lat, lon, km):
    """Retourne (miny, minx, maxy, maxx) autour d'un point."""
    dlat = km / 111.0
    dlon = km / (111.0 * math.cos(math.radians(lat)))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


# ==============================================================================
# 2. API CORINE LAND COVER — IGN Géoplateforme (nouvelle URL 2024)
# ==============================================================================

# Mapping code_group RPG -> paramètres PV
# (code_group: (label, dispo_pv, dispo_agrivol, eligible_agrivol))
RPG_GROUP_PARAMS = {
    "1":  ("Céréales",                    0.80, 0.10, True),
    "2":  ("Oléagineux",                  0.80, 0.10, True),
    "3":  ("Protéagineux",                0.80, 0.10, True),
    "4":  ("Plantes à fibres",            0.75, 0.08, True),
    "5":  ("Sucre",                       0.75, 0.08, True),
    "6":  ("Fourrage",                    0.85, 0.10, True),
    "7":  ("Légumes/fleurs",              0.50, 0.05, True),
    "8":  ("Vignes",                      0.00, 0.00, False),  # AOP protégés
    "9":  ("Fruits",                      0.40, 0.08, True),
    "10": ("Oliviers",                    0.30, 0.05, True),
    "11": ("Noix",                        0.40, 0.05, True),
    "13": ("Prairies permanentes",        0.85, 0.10, True),
    "14": ("Prairies temporaires",        0.85, 0.10, True),
    "15": ("Estives/landes",              0.70, 0.05, True),
    "16": ("Gel",                         0.90, 0.00, False),
    "17": ("Divers",                      0.70, 0.08, True),
    "18": ("Semences",                    0.75, 0.08, True),
    "19": ("Légumineuses",                0.80, 0.10, True),
    "20": ("Miscanthus/autres énergies",  0.85, 0.00, False),
    "22": ("Vergers",                     0.40, 0.08, True),
    "23": ("Autres cultures industrielles",0.75, 0.08, True),
    "24": ("Surfaces non agricoles",      0.60, 0.00, False),
    "25": ("Cultures permanentes",        0.50, 0.05, True),
    "26": ("Bois/forêts",                 0.00, 0.00, False),
    "27": ("Zones humides",               0.00, 0.00, False),
    "28": ("Surfaces non exploitées",     0.90, 0.00, False),
}


def fetch_rpg(lat, lon, buffer_km):
    """
    IGN Géoplateforme WFS — RPG (Registre Parcellaire Graphique) dernière édition
    URL  : https://data.geopf.fr/wfs/wfs   (noter /wfs/wfs et non /wfs/ows)
    Layer: RPG.LATEST:parcelles_graphiques
    BBOX : ordre miny,minx,maxy,maxx (lat_min,lon_min,lat_max,lon_max) SANS SRSNAME

    Avantages vs CLC :
    - Vraies parcelles agricoles déclarées à la PAC
    - Code culture précis (code_cultu + code_group)
    - Géométries exactes des parcelles
    - Mise à jour annuelle
    """
    miny, minx, maxy, maxx = get_bbox(lat, lon, buffer_km)

    url = "https://data.geopf.fr/wfs/wfs"
    params = {
        "SERVICE":      "WFS",
        "VERSION":      "2.0.0",
        "REQUEST":      "GetFeature",
        "TYPENAMES":    "RPG.LATEST:parcelles_graphiques",
        # CRITIQUE : ordre lat_min,lon_min,lat_max,lon_max SANS SRSNAME
        "BBOX":         f"{miny},{minx},{maxy},{maxx}",
        "outputFormat": "application/json",
        "COUNT":        "5000",
        "SORTBY":       "surf_parc D",  # trier par surface décroissante
    }

    print(f"[RPG] IGN Géoplateforme → RPG.LATEST:parcelles_graphiques...")
    try:
        r = requests.get(url, params=params, timeout=90)
        r.raise_for_status()
        data = r.json()
        features = data.get("features", [])
        total = data.get("totalFeatures", 0)
        print(f"[RPG] OK {len(features)} parcelles recues (total zone: {total:,})")

        if not features:
            print("[RPG] Zone vide -> fallback synthetique")
            return _synth_clc(lat, lon, buffer_km)

        results = []
        for f in features:
            try:
                geom = shape(f["geometry"]) if HAS_GEO else None
                props = f.get("properties", {})
                code_group = str(props.get("code_group") or "17")
                surf_ha = float(props.get("surf_parc") or 0)
                # surf_parc est en hectares dans le RPG IGN
                if surf_ha > 0:
                    area_m2 = surf_ha * 1e4  # ha -> m²
                elif geom is not None:
                    # Convertir degrés² -> m² : 1°lat=111km, 1°lon=111km*cos(lat)
                    import math as _math
                    lat_approx = geom.centroid.y if hasattr(geom, "centroid") else 47.0
                    area_m2 = geom.area * (111_000 ** 2) * _math.cos(_math.radians(lat_approx))
                else:
                    area_m2 = 5_000  # 0.5 ha par défaut
                results.append({
                    "geometry":   geom,
                    "clc_code":   int(code_group) if code_group.isdigit() else 17,
                    "clc_label":  RPG_GROUP_PARAMS.get(code_group, ("Divers", 0.70, 0.08, True))[0],
                    "code_cultu": props.get("code_cultu", ""),
                    "code_group": code_group,
                    "area_m2":    float(area_m2),
                })
            except Exception:
                continue

        print(f"[RPG] {len(results)} parcelles parsees")
        return results

    except Exception as e:
        print(f"[RPG] Erreur : {e} -> fallback synthetique")
        return _synth_clc(lat, lon, buffer_km)


def _synth_clc(lat, lon, buffer_km):
    """Données synthétiques réalistes pour tests sans internet."""
    if not HAS_GEO:
        return _synth_clc_no_geo(lat, lon, buffer_km)

    np.random.seed(int(abs(lat * 100 + lon * 100)) % 9999)
    classes = [211, 231, 242, 312, 121, 133, 111]
    weights = [0.35, 0.25, 0.15, 0.10, 0.05, 0.05, 0.05]
    miny, minx, maxy, maxx = get_bbox(lat, lon, buffer_km)
    results = []

    for _ in range(70):
        lon_c = minx + np.random.random() * (maxx - minx)
        lat_c = miny + np.random.random() * (maxy - miny)
        if haversine_m(lat, lon, lat_c, lon_c) > buffer_km * 1000:
            continue
        size_ha = float(np.clip(np.random.lognormal(3.9, 0.8), 2.5, 500))
        side = math.sqrt(size_ha * 1e4) / 111_000
        geom = box(lon_c - side/2, lat_c - side/2, lon_c + side/2, lat_c + side/2)
        clc_code = int(np.random.choice(classes, p=weights))
        results.append({"geometry": geom, "clc_code": clc_code, "area_m2": size_ha * 1e4})

    print(f"[CLC][SYNTH] {len(results)} parcelles synthétiques générées")
    return results


def _synth_clc_no_geo(lat, lon, buffer_km):
    """Version sans Shapely."""
    np.random.seed(42)
    classes = [211, 231, 242, 312, 121, 133, 111]
    weights = [0.35, 0.25, 0.15, 0.10, 0.05, 0.05, 0.05]
    results = []
    for _ in range(60):
        angle = np.random.uniform(0, 2 * math.pi)
        dist_m = np.random.uniform(500, buffer_km * 900)
        lat_c = lat + (dist_m / 111_000) * math.cos(angle)
        lon_c = lon + (dist_m / (111_000 * math.cos(math.radians(lat)))) * math.sin(angle)
        size_ha = float(np.clip(np.random.lognormal(3.9, 0.8), 2.5, 500))
        clc_code = int(np.random.choice(classes, p=weights))
        results.append({
            "geometry": None, "clc_code": clc_code, "area_m2": size_ha * 1e4,
            "centroid_lat": lat_c, "centroid_lon": lon_c,
        })
    return results


# ==============================================================================
# 3. API DVF — CEREMA Open Data (nouvelle URL 2024)
# ==============================================================================

def fetch_dvf(lat, lon, buffer_km):
    """
    API DVF open data — mutations foncières géolocalisées
    URL de production : https://apidf-preprod.cerema.fr/dvf_opendata/geomutations/

    ⚠️  L'URL preprod retourne 403 depuis l'extérieur.
    Alternative publique : télécharger le CSV DVF annuel depuis data.gouv.fr
    et appeler fetch_dvf_from_csv() à la place.

    URL CSV : https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/departements/{dept}.csv.gz
    Exemple dept 59 (Nord) : https://files.data.gouv.fr/geo-dvf/latest/csv/2023/departements/59.csv.gz
    """
    # Tentative API CEREMA production
    miny, minx, maxy, maxx = get_bbox(lat, lon, buffer_km)
    url = "https://apidf-preprod.cerema.fr/dvf_opendata/geomutations/"
    params = {
        "in_bbox":         f"{minx},{miny},{maxx},{maxy}",
        "nature_mutation": "Vente",
        "page_size":       500,
    }

    print(f"[DVF] API CEREMA → geomutations bbox {minx:.2f},{miny:.2f}...")
    try:
        r = requests.get(url, params=params, timeout=45)
        r.raise_for_status()
        data = r.json()
        # L'API CEREMA retourne soit GeoJSON (features) soit DRF pagination (results)
        features = data.get("features") or data.get("results") or []
        print(f"[DVF] ✓ {len(features)} transactions reçues")

        commune_prices = {}
        for feat in features:
            props = feat.get("properties", feat) if isinstance(feat, dict) else {}
            surface = float(props.get("sterr") or props.get("surface_terrain") or 0)
            valeur  = float(props.get("valeurfonc") or props.get("valeur_fonciere") or 0)
            code    = str(props.get("l_codinsee") or props.get("code_commune") or "")
            if surface > 100 and valeur > 0:
                p = valeur / surface
                if 0.05 <= p <= 100:  # plage plausible €/m² pour terrain nu
                    commune_prices.setdefault(code, []).append(p)

        result = {c: float(np.median(v)) for c, v in commune_prices.items() if v}
        if result:
            med = float(np.median(list(result.values())))
            print(f"[DVF] ✓ Prix médian zone : {med:.2f} €/m²")
        else:
            print("[DVF] Pas de transactions nues — fallback régional")

        result["__default__"] = _dvf_fallback(lat)
        return result

    except Exception as e:
        print(f"[DVF] API CEREMA indisponible ({type(e).__name__}) -> tentative CSV data.gouv.fr...")
        return fetch_dvf_from_csv(lat, lon, buffer_km)


def fetch_dvf_from_csv(lat, lon, buffer_km, year=2023):
    """
    Alternative 100% publique -- telecharge le CSV DVF par departement.
    URL : https://files.data.gouv.fr/geo-dvf/latest/csv/{year}/departements/{dept}.csv.gz
    Pas de cle API, pas d inscription. Mis a jour semestriellement par la DGFiP.
    Utilise automatiquement si l API CEREMA est inaccessible.
    """
    import io
    import gzip as gz_lib

    depts = _guess_depts(lat, lon, buffer_km)
    miny, minx, maxy, maxx = get_bbox(lat, lon, buffer_km)
    all_prices = []
    commune_prices = {}

    for dept in depts:
        url = f"https://files.data.gouv.fr/geo-dvf/latest/csv/{year}/departements/{dept}.csv.gz"
        print(f"[DVF] CSV data.gouv.fr -> departement {dept}...")
        try:
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            buf = io.BytesIO(r.content)
            with gz_lib.open(buf, "rt", encoding="utf-8") as f:
                df = pd.read_csv(f, low_memory=False)

            if "nature_mutation" in df.columns:
                df = df[df["nature_mutation"] == "Vente"]
            # Garder uniquement les terrains nus agricoles (type_local vide)
            if "type_local" in df.columns:
                df = df[df["type_local"].isna() | (df["type_local"] == "")]
            required = ["latitude", "longitude", "valeur_fonciere", "surface_terrain"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Colonnes manquantes : {missing}")

            df = df.dropna(subset=required)
            df["surface_terrain"] = pd.to_numeric(df["surface_terrain"], errors="coerce")
            df["valeur_fonciere"] = pd.to_numeric(df["valeur_fonciere"], errors="coerce")
            df = df[df["surface_terrain"] > 100]
            df = df[(df["latitude"] >= miny) & (df["latitude"] <= maxy) &
                    (df["longitude"] >= minx) & (df["longitude"] <= maxx)]
            df["prix_m2"] = df["valeur_fonciere"] / df["surface_terrain"]
            df = df[(df["prix_m2"] >= 0.05) & (df["prix_m2"] <= 100)]

            print(f"[DVF] OK {len(df)} transactions (dept {dept})")
            for code, grp in df.groupby("code_commune"):
                commune_prices.setdefault(str(code), []).extend(grp["prix_m2"].tolist())
            all_prices.extend(df["prix_m2"].tolist())

        except Exception as e2:
            print(f"[DVF] Dept {dept} : {type(e2).__name__}: {e2}")

    if all_prices:
        med = float(np.median(all_prices))
        print(f"[DVF] OK Prix median : {med:.2f} euro/m2 ({len(all_prices)} transactions)")
        result = {c: float(np.median(v)) for c, v in commune_prices.items()}
        result["__default__"] = med
        return result

    print("[DVF] Aucune donnee -> fallback SAFER")
    return {"__default__": _dvf_fallback(lat)}


def _guess_depts(lat, lon, buffer_km):
    """Retourne les codes departement(s) dans le buffer (approximation par centroides)."""
    dept_centroids = {
        "59": (50.63, 3.06), "62": (50.29, 2.78), "80": (49.89, 2.30),
        "02": (49.56, 3.62), "60": (49.41, 2.08), "76": (49.44, 1.10),
        "27": (49.03, 1.15), "14": (49.18, -0.37), "50": (49.12, -1.08),
        "61": (48.43,  0.09), "75": (48.86, 2.35), "78": (48.77, 1.98),
        "91": (48.63, 2.23), "92": (48.85, 2.23), "93": (48.91, 2.48),
        "94": (48.79, 2.46), "95": (49.05, 2.08), "77": (48.62, 2.71),
        "57": (49.12, 6.18), "67": (48.57, 7.75), "69": (45.75, 4.85),
        "13": (43.30, 5.38), "33": (44.84, -0.58), "31": (43.60, 1.44),
        "06": (43.71, 7.26), "44": (47.22, -1.55), "35": (48.12, -1.68),
        "29": (48.39, -4.49), "56": (47.66, -2.75), "22": (48.51, -2.77),
        "63": (45.78,  3.08), "87": (45.83,  1.26),
    }
    result = [c for c, (dlat, dlon) in dept_centroids.items()
              if haversine_m(lat, lon, dlat, dlon) < buffer_km * 1000 * 2.5]
    if not result:
        closest = min(dept_centroids.items(),
                      key=lambda x: haversine_m(lat, lon, x[1][0], x[1][1]))
        result = [closest[0]]
    return result[:3]


def _dvf_fallback(lat):
    """
    Prix terres agricoles par latitude — Source : SAFER Bilan annuel 2023.
    Unité : €/m²  (= prix_ha / 10 000)
    """
    if lat > 49.5:   return 0.65   # Hauts-de-France  ~6 500 €/ha
    elif lat > 47.5: return 0.55   # Normandie / IdF
    elif lat > 46.0: return 0.45   # Centre / Val de Loire
    elif lat > 43.5: return 0.38   # Nouvelle-Aquitaine / Occitanie
    else:            return 0.42   # PACA


# ==============================================================================
# 4. API INPN — ZONES PROTÉGÉES (nouvelle URL 2024)
# ==============================================================================

def fetch_protected(lat, lon, buffer_km):
    """
    Zones Natura 2000 et ZNIEFF via WFS Carmen/INPN
    URL : https://ws.carmencarto.fr/WFS/119/fxx_natura2000

    ⚠️ L'ancienne URL inpn.mnhn.fr/espece/api/zone-protection retourne 404.
    Le service Carmen est le nouveau point d'accès WFS de l'INPN.
    """
    miny, minx, maxy, maxx = get_bbox(lat, lon, buffer_km)

    # IGN Géoplateforme WFS — couches PatriNat (INPN)
    # Meme regle BBOX que RPG : miny,minx,maxy,maxx SANS SRSNAME
    url = "https://data.geopf.fr/wfs/wfs"
    zones = []

    for layer, zone_type in [
        ("patrinat_znieff1:znieff1", "ZNIEFF1"),
        ("patrinat_sic:sic",         "Natura2000-SIC"),
        ("patrinat_zps:zps",         "Natura2000-ZPS"),
    ]:
        params = {
            "SERVICE":      "WFS",
            "VERSION":      "2.0.0",
            "REQUEST":      "GetFeature",
            "TYPENAMES":    layer,
            "BBOX":         f"{miny},{minx},{maxy},{maxx}",
            "outputFormat": "application/json",
            "COUNT":        "100",
        }
        print(f"[INPN] IGN WFS -> {layer}...")
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            features = r.json().get("features", [])
            for f in features:
                if HAS_GEO:
                    try:
                        geom = shape(f["geometry"])
                        props = f.get("properties", {})
                        name = (props.get("nom_znieff") or props.get("nom")
                                or props.get("sitename") or "")
                        zones.append({"geometry": geom, "type": zone_type, "name": name})
                    except Exception:
                        pass
            print(f"[INPN] OK {len(features)} zones {zone_type}")
        except Exception as e:
            print(f"[INPN] {layer} : {type(e).__name__}")

    print(f"[INPN] Total : {len(zones)} zones protegees")
    return zones


# ==============================================================================
# 5. CONSTRUCTION DU LAND GRID
# ==============================================================================

def build_land_grid(lat, lon, buffer_km=30.0, output_dir=None,
                    synthetic_mode=False, skip_protected=False):
    """
    Fonction principale — construit la grille de terrains candidats.

    Colonnes compatibles costutils.py :
        area_photo    ← surface PV standard (m²)
        area_agrivol  ← surface agrivoltaïque (m²)
        wavgprice     ← prix foncier (€/m²)
        distance_0    ← distance au site (m)

    Sorties :
        gisdata/db_semippa_auto.csv        ← toujours généré
        gisdata/db_semippa_auto.gpkg       ← si geopandas installé
        gisdata/db_semippa_auto_map.html   ← carte interactive (toujours)
    """
    print(f"\n{'='*60}")
    print(f"BUILD LAND GRID  lat={lat}  lon={lon}  buffer={buffer_km} km")
    print(f"{'='*60}")

    # ── 1. Données CLC ────────────────────────────────────────────────────────
    clc = _synth_clc(lat, lon, buffer_km) if synthetic_mode else fetch_rpg(lat, lon, buffer_km)

    # ── 2. Prix fonciers DVF ──────────────────────────────────────────────────
    dvf = {"__default__": _dvf_fallback(lat)} if synthetic_mode else fetch_dvf(lat, lon, buffer_km)
    default_price = dvf["__default__"]

    # ── 3. Zones protégées ────────────────────────────────────────────────────
    protected = [] if (synthetic_mode or skip_protected) else fetch_protected(lat, lon, buffer_km)

    # ── 4. Assembler les lignes ───────────────────────────────────────────────
    rows, excl_class, excl_size = [], 0, 0

    for feat in clc:
        code = feat["clc_code"]

        # RPG: code = code_group (int), use RPG_GROUP_PARAMS
        code_group_str = str(feat.get("code_group") or code)
        if code_group_str not in RPG_GROUP_PARAMS:
            # Groupe inconnu -> traiter comme divers avec dispo partielle
            code_group_str = "17"

        label, avail_pv, avail_agri, elig_agri = RPG_GROUP_PARAMS[code_group_str]

        if avail_pv == 0 and avail_agri == 0:
            excl_class += 1
            continue
        area_m2 = feat["area_m2"]

        # Pas de filtre taille minimum pour RPG — on garde toutes les parcelles
        # (les petites parcelles adjacentes s'agrègeront dans le merit order)
        pass  # excl_size non utilisé pour RPG

        geom = feat.get("geometry")
        if geom is not None and HAS_GEO:
            c = geom.centroid
            clat, clon = c.y, c.x
        else:
            clat = feat.get("centroid_lat", lat)
            clon = feat.get("centroid_lon", lon)

        dist_m = haversine_m(lat, lon, clat, clon)
        if dist_m > buffer_km * 1000:
            continue

        zone = ""
        if protected and geom is not None and HAS_GEO:
            for z in protected:
                if geom.intersects(z["geometry"]):
                    zone = z["type"]
                    break

        row = {
            "clc_code":       code,
            "clc_label":      label,
            "code_cultu":     feat.get("code_cultu", ""),
            "area_m2":        area_m2,
            "area_photo":     area_m2 * avail_pv,                           # ← costutils.py
            "area_agrivol":   area_m2 * avail_agri if elig_agri else 0.0,   # ← costutils.py
            "wavgprice":      default_price,                                 # ← costutils.py
            "distance_0":     dist_m,                                        # ← costutils.py
            "cable_cost_est": dist_m * CABLE_EUR_PER_MW_PER_M,
            "protected_zone": zone,
            "eligible_pv":    avail_pv > 0,
            "eligible_agri":  elig_agri,
            "centroid_lat":   clat,
            "centroid_lon":   clon,
        }
        if geom is not None:
            row["geometry"] = geom
        rows.append(row)

    print(f"\n[GRID] Retenues      : {len(rows)}")
    print(f"[GRID] Excl. classe  : {excl_class}")
    print(f"[GRID] Excl. taille  : {excl_size}")

    if not rows:
        raise ValueError("Aucune parcelle candidate — vérifiez connexion ou augmentez le buffer.")

    # ── 5. DataFrame / GeoDataFrame ───────────────────────────────────────────
    if HAS_GEO and any("geometry" in r for r in rows):
        gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    else:
        gdf = pd.DataFrame(rows)

    _print_stats(gdf)

    # ── 6. Sauvegardes ────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    prefix = str(Path(output_dir) / "db_semippa_auto")

    # CSV — toujours produit (lisible sans QGIS)
    csv_path = prefix + ".csv"
    cols = [c for c in gdf.columns if c != "geometry"]
    gdf[cols].to_csv(csv_path, index=False)
    print(f"[OK] CSV    → {csv_path}")

    # GPKG — si geopandas disponible
    if HAS_GEO and isinstance(gdf, gpd.GeoDataFrame):
        gpkg_path = prefix + ".gpkg"
        gdf.to_file(gpkg_path, driver="GPKG")
        print(f"[OK] GPKG   → {gpkg_path}  (ouvrir avec QGIS)")

    # Carte HTML interactive — toujours produite
    html_path = prefix + "_map.html"
    _build_html_map(gdf, lat, lon, buffer_km, html_path)
    print(f"[OK] Carte  → {html_path}  ← double-cliquer pour ouvrir dans le navigateur")

    return gdf


# ==============================================================================
# 6. CARTE HTML INTERACTIVE (Leaflet.js via CDN)
# ==============================================================================

def _build_html_map(gdf, client_lat, client_lon, buffer_km, output_path):
    """
    Génère une carte HTML interactive avec Leaflet.js.
    Ne nécessite pas folium — fonctionne avec n'importe quel navigateur.
    """
    features_js = []

    for _, row in gdf.iterrows():
        code      = int(row.get("clc_code", 0))
        color     = CLC_COLORS.get(code, "#aaaaaa")
        dist_km   = row.get("distance_0", 0) / 1000
        area_ha   = row.get("area_m2", 0) / 1e4
        pv_ha     = row.get("area_photo", 0) / 1e4
        agri_ha   = row.get("area_agrivol", 0) / 1e4
        price     = row.get("wavgprice", 0)
        label     = row.get("clc_label", f"CLC {code}")
        prot      = str(row.get("protected_zone", "") or "")
        power_mw  = pv_ha * 1e4 * 80 / 1e6
        clat      = row.get("centroid_lat", client_lat)
        clon      = row.get("centroid_lon", client_lon)

        prot_html = (f"<span style='color:orange'>⚠️ Zone protégée : {prot}</span>"
                     if prot else "<span style='color:green'>✓ Pas de zone protégée</span>")

        popup = (
            f"<b>{label}</b> (CLC {code})<br>"
            f"Surface totale : <b>{area_ha:.1f} ha</b><br>"
            f"Disponible PV  : {pv_ha:.1f} ha → <b>{power_mw:.1f} MW</b><br>"
            f"Agrivoltaïque  : {agri_ha:.1f} ha<br>"
            f"Distance site  : {dist_km:.1f} km<br>"
            f"Prix foncier   : {price:.2f} €/m²<br>"
            f"Câblage 1 MW   : {dist_km*1000*CABLE_EUR_PER_MW_PER_M/1000:.0f} k€<br>"
            f"{prot_html}"
        )

        geom = row.get("geometry")
        if geom is not None and HAS_GEO:
            try:
                geojson_str = json.dumps(geom.__geo_interface__)
                features_js.append({
                    "type": "polygon", "geojson": geojson_str,
                    "color": color, "popup": popup,
                })
                continue
            except Exception:
                pass
        # Fallback cercle
        radius = math.sqrt(max(area_ha * 1e4, 1e4) / math.pi)
        features_js.append({
            "type": "circle", "lat": clat, "lon": clon,
            "radius": radius, "color": color, "popup": popup,
        })

    features_json = json.dumps(features_js, ensure_ascii=False)

    # Statistiques pour le panneau
    total_ha  = gdf["area_m2"].sum() / 1e4
    mw_pv     = gdf["area_photo"].sum() * 80 / 1e6
    mw_agri   = gdf["area_agrivol"].sum() * 80 / 1e6
    n         = len(gdf)
    avg_dist  = gdf["distance_0"].mean() / 1000
    prix_med  = gdf["wavgprice"].median()
    n_prot    = int((gdf["protected_zone"].astype(str) != "").sum())

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>PPA Land Grid — {client_lat:.2f}, {client_lon:.2f}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    body {{ margin:0; font-family:Arial,sans-serif; }}
    #map {{ height:100vh; width:100%; }}
    #panel {{
      position:absolute; top:10px; right:10px; z-index:1000;
      background:white; padding:14px 16px; border-radius:8px;
      box-shadow:0 2px 12px rgba(0,0,0,.25); min-width:230px;
      max-width:270px; font-size:13px; line-height:1.5;
    }}
    #panel h3 {{ margin:0 0 8px; font-size:14px; color:#1F4E79; }}
    .stat {{ display:flex; justify-content:space-between; margin:2px 0; }}
    .stat b {{ color:#1F4E79; }}
    .legend-item {{ display:flex; align-items:center; margin:3px 0; font-size:12px; }}
    .dot {{ width:13px; height:13px; border-radius:3px; margin-right:7px;
            border:1px solid #bbb; flex-shrink:0; }}
    hr {{ border:none; border-top:1px solid #eee; margin:8px 0; }}
    .warn {{ color:orange; font-size:12px; }}
  </style>
</head>
<body>
<div id="map"></div>
<div id="panel">
  <h3>🗺️ PPA Land Grid France</h3>
  <div class="stat"><span>Parcelles candidates</span><b>{n}</b></div>
  <div class="stat"><span>Surface totale</span><b>{total_ha:,.0f} ha</b></div>
  <div class="stat"><span>Puissance PV max</span><b>{mw_pv:,.0f} MW</b></div>
  <div class="stat"><span>Puissance agrivolt.</span><b>{mw_agri:,.0f} MW</b></div>
  <div class="stat"><span>Distance moyenne</span><b>{avg_dist:.1f} km</b></div>
  <div class="stat"><span>Prix foncier médian</span><b>{prix_med:.2f} €/m²</b></div>
  {'<div class="warn">⚠️ ' + str(n_prot) + ' parcelles en zone protégée</div>' if n_prot else ''}
  <hr>
  <b style="font-size:12px">Légende (classe CLC)</b><br>
  <div class="legend-item"><div class="dot" style="background:#f5d07a"></div>Terres arables</div>
  <div class="legend-item"><div class="dot" style="background:#a8d08d"></div>Prairies</div>
  <div class="legend-item"><div class="dot" style="background:#c6e0b4"></div>Systèmes complexes</div>
  <div class="legend-item"><div class="dot" style="background:#b4a7d6"></div>Friches / chantiers</div>
  <div class="legend-item"><div class="dot" style="background:#ea9999"></div>Zones industrielles</div>
  <div class="legend-item"><div class="dot" style="background:#ffe599"></div>Végétation clairsemée</div>
  <hr>
  <small style="color:#888">⭐ = site client &nbsp;|&nbsp; --- = rayon buffer<br>
  Cliquer sur une parcelle pour les détails</small>
</div>

<script>
var map = L.map('map').setView([{client_lat}, {client_lon}], 10);

L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  attribution: '© <a href="https://openstreetmap.org">OpenStreetMap</a> contributors',
  maxZoom: 18
}}).addTo(map);

// Marqueur site client
L.marker([{client_lat}, {client_lon}], {{
  icon: L.divIcon({{
    html: '<div style="font-size:26px;margin:-13px 0 0 -13px">⭐</div>',
    className:'', iconSize:[26,26]
  }})
}}).addTo(map)
  .bindPopup('<b>Site client</b><br>lat={client_lat} / lon={client_lon}');

// Cercle buffer
L.circle([{client_lat}, {client_lon}], {{
  radius:{buffer_km * 1000},
  color:'#1F4E79', fill:false, weight:2, dashArray:'8 4'
}}).addTo(map);

// Parcelles
var features = {features_json};

features.forEach(function(f) {{
  var style = {{
    color:'#444', weight:1,
    fillColor: f.color, fillOpacity: 0.65
  }};
  var layer;
  if (f.type === 'polygon') {{
    layer = L.geoJSON(JSON.parse(f.geojson), {{style: style}});
  }} else {{
    layer = L.circle([f.lat, f.lon], {{
      radius: f.radius,
      color:'#444', weight:1,
      fillColor: f.color, fillOpacity: 0.65
    }});
  }}
  layer.bindPopup(f.popup, {{maxWidth:280}});
  layer.addTo(map);
}});
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)


# ==============================================================================
# 7. STATS
# ==============================================================================

def _print_stats(gdf):
    print(f"\n{'─'*55}")
    print("RÉSUMÉ DE LA GRILLE DE TERRAINS")
    print(f"{'─'*55}")
    print(f"Surface totale identifiée       : {gdf['area_m2'].sum()/1e4:,.0f} ha")
    pv_ha = gdf['area_photo'].sum()/1e4
    ag_ha = gdf['area_agrivol'].sum()/1e4
    print(f"Surface disponible PV standard  : {pv_ha:,.0f} ha → {pv_ha*1e4*80/1e6:,.0f} MW")
    print(f"Surface disponible agrivoltaïque: {ag_ha:,.0f} ha → {ag_ha*1e4*80/1e6:,.0f} MW")
    print(f"Distance moyenne au site        : {gdf['distance_0'].mean()/1000:.1f} km")
    print(f"Distance max au site            : {gdf['distance_0'].max()/1000:.1f} km")
    print(f"Prix foncier médian             : {gdf['wavgprice'].median():.2f} €/m²")
    print(f"\nTop 5 classes CLC :")
    totals = gdf.groupby("clc_label")["area_m2"].sum().sort_values(ascending=False)
    s = gdf["area_m2"].sum()
    for lbl, area in totals.head(5).items():
        print(f"  {lbl[:50]:<50} : {area/1e4:6.0f} ha ({100*area/s:.0f}%)")
    n_p = (gdf["protected_zone"].astype(str) != "").sum()
    if n_p:
        print(f"\n⚠️  {n_p} parcelles en zone protégée (étude d'impact requise)")
    print(f"{'─'*55}")


# ==============================================================================
# 8. INTÉGRATION costutils.py
# ==============================================================================

def process_grid_site_data_france(client_lat, client_lon, buffer_km=30.0,
                                   cache_dir=None, force_rebuild=False,
                                   synthetic_mode=False):
    """
    Drop-in replacement de process_grid_site_data() dans costutils.py.

    Usage dans ppamodule.py :
        from build_land_grid import process_grid_site_data_france

        client_lat = scenario_defaults["Client Latitude"]
        client_lon = scenario_defaults["Client Longitude"]
        buffer_km  = scenario_defaults["Buffer distance (m)"] / 1000

        grid_df = process_grid_site_data_france(client_lat, client_lon, buffer_km)
    """
    csv_cache = Path(cache_dir) / "db_semippa_auto.csv"

    if not force_rebuild and csv_cache.exists():
        print(f"[GRID] Cache trouvé → {csv_cache}")
        return pd.read_csv(csv_cache)

    return build_land_grid(
        lat=client_lat, lon=client_lon,
        buffer_km=buffer_km,
        output_dir=cache_dir,
        synthetic_mode=synthetic_mode,
    )


# ==============================================================================
# 9. CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PPA France — Construction automatique grille de terrains candidats"
    )
    parser.add_argument("--lat",       type=float, default=50.93, help="Latitude site client")
    parser.add_argument("--lon",       type=float, default=2.38,  help="Longitude site client")
    parser.add_argument("--buffer",    type=float, default=30.0,  help="Rayon de recherche (km)")
    _default_out = _os_ap.path.join(_os_ap.path.dirname(_os_ap.path.abspath(__file__)), "gisdata")
    parser.add_argument("--output",    type=str,   default=_default_out, help="Dossier de sortie")
    parser.add_argument("--synthetic", action="store_true", help="Données synthétiques (test sans internet)")
    parser.add_argument("--no-inpn",   action="store_true", help="Ne pas interroger INPN (plus rapide)")
    parser.add_argument("--force",     action="store_true", help="Reconstruire même si cache présent")
    parser.add_argument("--test",      action="store_true", help="= --synthetic (alias rapide)")
    args = parser.parse_args()

    if args.test:
        args.synthetic = True

    gdf = build_land_grid(
        lat=args.lat, lon=args.lon,
        buffer_km=args.buffer,
        output_dir=args.output,
        synthetic_mode=args.synthetic,
        skip_protected=args.no_inpn,
    )

    print(f"\n✓ Terminé — {len(gdf)} parcelles candidates")
    print(f"  Carte interactive : {args.output}/db_semippa_auto_map.html")
    print(f"  Données tabulaires: {args.output}/db_semippa_auto.csv")
