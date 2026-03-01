# French PPA Model — README v3.0

> Modèle d'optimisation des contrats PPA pour industriels français raccordés en haute tension (HTB)
> **Version 3.0 · Mars 2026**
> Cas d'usage : ArcelorMittal Dunkerque (685 MW) · H2 Normandie (200 MW)

---

## 1. Présentation du modèle

Le French PPA Model évalue et optimise la stratégie d'approvisionnement en électricité renouvelable d'un industriel français. Il simule heure par heure, sur 20 ans, la production ENR, le prix spot EPEX, le TURPE HTB, les clauses contractuelles et calcule la VAN des économies réalisées.

### Ce que le modèle fait
- **Screener** les meilleurs sites solaires et éoliens disponibles pour un client (capture rate, corrélation charge, LCOE)
- **Optimiser** le mix PPA par programmation linéaire (scipy LP) : MW par site, prix PPA minimum, couverture charge cible
- **Calculer** le settlement financier annuel selon 6 structures contractuelles (fixed, collar, indexed_spot, floor_only...)
- **Projeter** le cashflow sur 20 ans (dégradation technique, escalade TURPE/EPEX/GO) et calculer VAN, payback, IRR
- **Visualiser** toutes les données et résultats dans une interface Streamlit 5 pages

### Ce que le modèle ne fait PAS
- Calcul de dimensionnement de raccordement réseau (RTE/ENEDIS — à traiter séparément)
- Valorisation des flexibilités (effacement, balancing market)
- Modélisation des prix forward au-delà des hypothèses RTE Futurs Énergétiques 2050
- Consultation réglementaire sur les contrats — résultats à valider avec un juriste énergie

> **Note migration Corée → France** : ce modèle est une adaptation complète du modèle PPA coréen original (SK/Samsung). La migration est totale : devise EUR native, TURPE HTB à la place du KEPCO, EPEX Spot à la place du SMP, GOs à la place des RECs, `FranceGridUtils.py` à la place de `KEPCOutils.py`. Les fichiers de compatibilité intermédiaires (`ppamodule_france_adaptations.py`, `KEPCO_france.xlsx`) n'existent plus.

---

## 2. Structure du projet

```
French_PPA/
├── app.py                          ← Interface Streamlit (1 522 lignes)
├── ppamodule.py                    ← Moteur de calcul PPA (1 289 lignes)
├── FranceGridUtils.py              ← Utilitaires réseau & sites (1 023 lignes)
├── download_france_data.py         ← Téléchargement données (944 lignes)
├── build_land_grid.py              ← Données GIS foncières (inchangé)
├── explore_ppa_data_france.ipynb   ← Notebook d'exploration (usage local)
├── scenario_defaults.xlsx          ← Paramètres des 6 scénarios (60+ params)
│
├── database/
│   ├── grid_france.csv             ← Trajectoires CAPEX/CO2/ENR 2023-2050
│   ├── TURPE_france.xlsx           ← Tarifs TURPE 6 HTB (CRE Nov. 2024)
│   ├── wind_grid_france.xlsx       ← 12 parcs éoliens (offshore + onshore)
│   ├── solar_patterns.db           ← Profils PV horaires (10 sites PVGIS)
│   ├── wind_patterns.db            ← Profils éoliens (10 zones ERA5)
│   ├── load_patterns.db            ← Profils charge (4 types industriels)
│   ├── epex_profiles.db            ← Prix EPEX DA horaires (cache SQLite)
│   └── capture_rates.db            ← Capture rates précalculés
│
└── gisdata/                        ← Généré par build_land_grid.py
    ├── db_semippa_auto.gpkg        ← Parcelles candidates (RPG IGN)
    ├── db_semippa_auto.csv         ← Même données en CSV
    └── db_semippa_auto_map.html    ← Carte interactive Leaflet.js
```

---

## 3. Installation et démarrage

### 3.1 Dépendances Python

```bash
# Dépendances obligatoires
pip install pandas numpy scipy plotly streamlit openpyxl requests

# Pour build_land_grid.py uniquement (données GIS foncières)
pip install geopandas shapely

# Pour le notebook explore_ppa_data_france.ipynb
pip install notebook nbformat
```

### 3.2 Démarrage en mode offline (données synthétiques — immédiat)

```bash
# Étape 1 : générer toutes les données synthétiques
python download_france_data.py --offline

# Étape 2 : lancer l'interface Streamlit
streamlit run app.py
# → Accéder à http://localhost:8501
```

> Le mode offline génère des données synthétiques calibrées (PVGIS, ERA5, EPEX, trajectoires ADEME). Suffisant pour tester le modèle et configurer les scénarios. Les résultats absolus ne sont pas à prendre au pied de la lettre — remplacer par des données réelles pour tout usage client.

### 3.3 Démarrage en mode réel (données APIs — ~10-15 min)

```bash
# Télécharger les données réelles depuis PVGIS, ERA5, ODRE
python download_france_data.py

# Options disponibles :
python download_france_data.py --lat 50.93 --lon 2.38   # coordonnées client
python download_france_data.py --year 2021               # année météo
python download_france_data.py --epex epex_2024.csv      # CSV EPEX fourni
```

### 3.4 Calcul depuis Python (sans Streamlit)

```python
from ppamodule import run_model, compare_scenarios

# Scénario unique
results = run_model(
    xlsx_path='scenario_defaults.xlsx',
    scenario_col='PPA100_ArcelorMittal_MU',
    load_col='siderurgie',
    save_output=True,
)
print(f"VAN : {results['npv_eur']/1e6:.1f} M€  |  Payback : {results['payback_years']} ans")

# Comparaison de tous les scénarios
df = compare_scenarios('scenario_defaults.xlsx',
    ['PPA100_ArcelorMittal_MU', 'PPA100_ArcelorMittal_LU', 'PPA100_H2_Normandie'])
print(df[['scenario', 'npv_meur', 'coverage_pct', 'net_cost_eur_mwh']])
```

### 3.5 Données GIS (optionnel — scénarios on-site uniquement)

```bash
# Construire la grille de terrains candidats autour du site client
python build_land_grid.py --lat 50.93 --lon 2.38 --buffer 30

# Mode test sans internet
python build_land_grid.py --lat 50.93 --lon 2.38 --synthetic

# Résultat : gisdata/db_semippa_auto.gpkg + .csv + _map.html
```

> `build_land_grid.py` n'est nécessaire que pour les scénarios on-site. Pour tous les scénarios off-site, cette étape est facultative.

---

## 4. Remplacer les données synthétiques par des données réelles

| Donnée | Priorité | Comment obtenir | Impact si synthétique |
|---|---|---|---|
| Courbe de charge client | 🔴 Critique | Export horodaté depuis ENEDIS (espace client) ou gestionnaire d'énergie. Format CSV datetime/MW. | Résultats VAN et couverture charge non fiables |
| Prix EPEX Spot | 🟠 Haute | odre.opendatasoft.com → dataset `prix-spot-da-horaires` → Export CSV. Puis : `python download_france_data.py --epex epex.csv` | Capture rates et settlement approximatifs |
| eco2mix CO2/ENR | 🟠 Haute | odre.opendatasoft.com → `eco2mix-national-cons-def`. Colonnes `taux_co2` et `taux_enr`. | Indicateurs ESG/Scope 2 approximatifs |
| Parcs éoliens supplémentaires | 🟡 Normale | Compléter `wind_grid_france.xlsx` avec AO CRE 2021-2024 (thewindpower.net). | Mix PPA moins diversifié (12 parcs) |
| TURPE actualisé | 🟡 Normale | Chaque 1er août : services-rte.com → Bibliothèque documentaire → Fiches tarifaires TURPE 6. | Coût réseau sous-estimé si +4-5%/an non appliqué |

> ⚠️ **Priorité absolue** : obtenir la courbe de charge réelle du client industriel. C'est le seul paramètre qui ne peut pas être synthétisé de façon fiable — chaque site a un profil unique (fours à arc, électrolyseurs, process chimique...).

---

## 5. Guide de l'interface Streamlit

| Page | Ce qu'on y fait |
|---|---|
| 🏠 **Accueil** | Vérifier le statut des 8 fichiers données. Voir les scénarios disponibles. Guide de démarrage rapide. |
| 📊 **Data Explorer** | 9 onglets : profils solaires/éoliens/charge, EPEX Spot, capture rates, parcs éoliens (carte mapbox), complémentarité (matrice corrélation), qualité des données. |
| ⚙️ **Scénario & Calcul** | Sélectionner un scénario, ajuster charge/WACC/durée/structure contractuelle. Configurer les clauses avancées (curtailment, prix négatifs, GOs, risques financiers). Cliquer **Lancer le calcul**. |
| 📈 **Résultats** | KPIs : VAN, MW PPA, couverture charge %, coût net €/MWh, payback. Onglets : cashflow 20 ans, mix PPA (bar + camembert), sites screenés (scatter score), paramètres utilisés. |
| 🗺️ **Carte GIS** | Carte Leaflet des parcelles candidates RPG. Stats surface/puissance PV/distance/prix foncier. Bouton pour reconstruire la grille. |

> Le répertoire `database/` est configurable depuis la barre latérale. Le statut des 8 fichiers est mis à jour en temps réel avec un badge coloré.

---

## 6. Concepts clés du marché français

### Capture Rate vs Prix Spot Moyen

Le capture rate mesure la valeur réelle d'un parc sur le marché spot, par rapport au prix moyen annuel.

```
CR = (Σ production(t) × spot(t)) / (Σ production(t)) / spot_moyen_annuel
```

| Technologie | Capture Rate typique France | Interprétation |
|---|---|---|
| Éolien offshore hivernal | 1.05 – 1.15 | Produit aux heures de pointe hiver → CR > 1 |
| Éolien onshore | 0.95 – 1.05 | Production répartie, moins ciblée sur les pointes |
| Solaire PV (latitude 50°N) | 0.82 – 0.92 | Midi solaire = heures de cannibalisation ENR |
| Solaire PV (latitude 44°N) | 0.78 – 0.88 | Cannibalisation plus forte dans le Sud |

### Structure PPA physique sleeved (dominante France 2024)

Le sleeving est la structure standard en France (> 65% des contrats). Le producteur injecte sa production au réseau au prix spot ; un fournisseur intermédiaire (sléeveur) aligne les flux financiers.

| Flux | Qui paie / reçoit quoi |
|---|---|
| Producteur → Réseau | Injecte les MWh au prix spot du marché |
| Sléeveur → Producteur | Verse le complément (PPA_price − spot) si PPA_price > spot |
| Acheteur → Sléeveur | Paie PPA_price × volume + sleeving_fee (2-5 €/MWh) |
| Settlement CfD net | (PPA_price − spot) × volume = transfert acheteur/producteur |

### TURPE péréquation — la règle d'or

En France, le TURPE est **identique partout sur le territoire**. La localisation du site industriel ou du parc ENR n'impacte pas le tarif réseau. Seul le niveau de tension de raccordement (HTB3 / HTB2 / HTB1) différencie les tarifs. C'est fondamentalement différent du modèle coréen où la zone géographique KEPCO déterminait le tarif.

### Fin de l'ARENH (janvier 2026)

> ⚠️ L'ARENH (42 €/MWh) a expiré fin décembre 2025. Son successeur, les contrats de long terme EDF (**CPP — Contrats à Prix Plafond**, estimés 60-70 €/MWh sur 15 ans), constitue désormais la principale alternative aux PPAs renouvelables. À intégrer comme scénario de référence alternatif dans les analyses client.

### Garanties d'Origine (GOs) ≠ RECs coréens

| Paramètre | GOs France | RECs Corée |
|---|---|---|
| Prix 2024 | 3 – 15 €/MWh selon granularité | ~57 €/MWh (80 000 KRW) |
| Impact sur décision PPA | Faible (< 5% du différentiel coût) | Significatif |
| Granularité annual | 5 – 8 €/MWh | Standard |
| Granularité monthly | 7 – 11 €/MWh (+40%) | N/A |
| Granularité hourly / 24-7 CFE | 10 – 20 €/MWh (+120%) | N/A |
| Marché | AIB / EEX Environmental Markets | KPX |

---

## 7. Migration Corée → France — correspondances

| Concept / fichier coréen | Équivalent français v3.0 | Note |
|---|---|---|
| `KEPCOutils.py` | `FranceGridUtils.py` | Entièrement réécrit. Même logique, données françaises. |
| `KEPCO.xlsx` / `KEPCO_france.xlsx` | `TURPE_france.xlsx` | Ancienne appellation supprimée. |
| `ppamodule_france_adaptations.py` | `ppamodule.py` (natif) | Le fichier de patch n'existe plus. |
| `grid.csv` | `database/grid_france.csv` | Nouvelles colonnes : `wind_onshore_capex`, `bess_capex_per_mwh`, `epex_spot_mean`... |
| `wind_grid.xlsx` | `database/wind_grid_france.xlsx` | 12 parcs (vs 6 offshore seulement avant). |
| SMP (System Marginal Price) | EPEX Spot Day-Ahead France | Même mécanique de prix marginal, marché européen couplé. |
| REC (Renewable Energy Cert.) | GO (Garantie d'Origine) | Prix ~10× plus bas en France qu'en Corée. |
| Currency Exchange Rate KRW/USD | Currency = 1.0 (EUR natif) | Paramètre maintenu dans scenario_defaults mais toujours = 1. |
| Load for SK / Load for Samsung | Load ArcelorMittal / Load H2 Normandie | Mêmes paramètres, noms francisés. |
| HV_C_I / HV_C_II / HV_C_III | HTB3 / HTB2 / HTB1 | Correspondance directe par niveau de tension. |
| CO2 ~450 gCO2/kWh | CO2 ~47 gCO2/kWh (2024) | France = nucléaire 70%. Argument PPA = prix, pas CO2. |

---

## 8. Sources officielles

| Source | Données | URL |
|---|---|---|
| ODRE | Prix EPEX DA horaires, eco2mix CO2/ENR | odre.opendatasoft.com |
| CRE | AO solaire/éolien/H2, délibérations TURPE | cre.fr |
| Services-RTE | Fiches tarifaires TURPE 6 HTB | services-rte.com |
| RTE Futurs Énergétiques | Scénarios CO2, ENR, prix long terme | rte-france.com/analyses-tendances-et-prospectives |
| PVGIS JRC | Profils PV horaires par coordonnées GPS | pvgis.ec.europa.eu |
| Open-Meteo ERA5 | Profils vent 100m horaires, réanalyse | archive-api.open-meteo.com |
| IGN Géoplateforme | RPG parcelles agricoles, ZNIEFF, Natura 2000 | data.geopf.fr/wfs/wfs |
| DVF data.gouv.fr | Prix terrain foncier par département | files.data.gouv.fr/geo-dvf/latest/csv |
| IRENA | CAPEX ENR et BESS mondiaux | irena.org/publications |
| AIB / EEX | Prix Garanties d'Origine | aib-net.org · eex.com/environmental-markets |
| Damodaran | WACC par secteur et pays | pages.stern.nyu.edu/~adamodar |
| WindEurope | Statistiques offshore Europe | windeurope.org/intelligence-platform |
| BloombergNEF | BESS CAPEX, prix carbone ETS | bloomberg.com/professional |

---

*French PPA Model v3.0 — Mars 2026 — Usage confidentiel*
