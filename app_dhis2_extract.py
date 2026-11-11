"""
V2
Application Streamlit : extraction de données DHIS2
(un formulaire x une unité d'organisation x N dernières années)
Quatre onglets indépendants : « Extraction », « Complétude », « Qualité des données » et « Tableaux de bord ».
Les données extraites sont agrégées par année : pour chaque élément de données, on cumule
toutes les combinaisons de catégories (categoryOptionCombo).

Installation : pip install streamlit requests pandas python-dotenv plotly
Lancement    : streamlit run app_dhis2_extract.py

Identifiants : fichier .env placé à côté de ce script (voir .env.example)
    DHIS2_URL=https://mon-instance.org/api
    DHIS2_USER=...
    DHIS2_PASSWORD=...
    DHIS2_TOKEN=d2pat_...
"""
import datetime
import math
import os
import re
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# Charge le .env situé dans le même dossier que ce script
load_dotenv(Path(__file__).parent / ".env")

st.set_page_config(page_title="Extraction DHIS2", page_icon="📊", layout="wide")


# ------------------------------------------------------------------
# Utilitaires
# ------------------------------------------------------------------
def secret(cle: str, defaut: str = "") -> str:
    """Lit d'abord le .env / variables d'environnement, puis st.secrets."""
    valeur = os.getenv(cle)
    if valeur:
        return valeur
    try:
        return st.secrets.get(cle, defaut)
    except Exception:
        return defaut


def make_session(mode: str, user: str, password: str, token: str) -> requests.Session:
    s = requests.Session()
    if mode == "Token (PAT)":
        s.headers["Authorization"] = f"ApiToken {token}"
    else:
        s.auth = (user, password)
    return s


@st.cache_data(ttl=3600, show_spinner=False)
def lister(url, mode, user, password, token, ressource) -> pd.DataFrame:
    """Liste (id, nom) d'une ressource : dataSets ou organisationUnits."""
    s = make_session(mode, user, password, token)
    r = s.get(
        f"{url}/{ressource}.json",
        params={"fields": "id,displayName,name", "paging": "false", "order": "displayName:asc"},
        timeout=120,
    )
    r.raise_for_status()
    return pd.DataFrame(r.json()[ressource])


@st.cache_data(ttl=3600, show_spinner=False)
def dictionnaire_noms(url, mode, user, password, token, ressource) -> dict:
    df = lister(url, mode, user, password, token, ressource)
    return dict(zip(df["id"], df["displayName"]))


@st.cache_data(ttl=3600, show_spinner=False)
def unites_avec_niveau(url, mode, user, password, token) -> pd.DataFrame:
    """Unités d'organisation avec leur niveau hiérarchique (id, displayName, level)."""
    s = make_session(mode, user, password, token)
    r = s.get(
        f"{url}/organisationUnits.json",
        params={"fields": "id,displayName,level", "paging": "false"},
        timeout=300,
    )
    r.raise_for_status()
    return pd.DataFrame(r.json().get("organisationUnits", []))


@st.cache_data(ttl=3600, show_spinner=False)
def noms_niveaux(url, mode, user, password, token) -> dict:
    """{niveau: nom} des niveaux hiérarchiques (ressource organisationUnitLevels)."""
    s = make_session(mode, user, password, token)
    r = s.get(
        f"{url}/organisationUnitLevels.json",
        params={"fields": "level,displayName", "paging": "false"},
        timeout=60,
    )
    r.raise_for_status()
    return {
        int(x["level"]): x.get("displayName", f"Niveau {x['level']}")
        for x in r.json().get("organisationUnitLevels", [])
    }


def extraire_annee(s, url, dataset, orgunit, enfants, annee) -> pd.DataFrame:
    params = {
        "dataSet": dataset,
        "orgUnit": orgunit,
        "children": str(enfants).lower(),
        "startDate": f"{annee}-01-01",
        "endDate": f"{annee}-12-31",
    }
    r = s.get(f"{url}/dataValueSets.json", params=params, timeout=300)
    r.raise_for_status()
    df = pd.DataFrame(r.json().get("dataValues", []))
    if not df.empty:
        df["annee_extraction"] = annee
    return df


def agreger(data: pd.DataFrame, cumul_ou: bool) -> pd.DataFrame:
    """Somme par année et par élément de données, toutes combinaisons de
    catégories confondues. Seules les valeurs numériques sont cumulées."""
    cles = ["annee_extraction", "dataElement", "dataElement_nom"]
    if not cumul_ou:
        cles = ["annee_extraction", "orgUnit", "orgUnit_nom", "dataElement", "dataElement_nom"]
    num = data.dropna(subset=["value_num"])
    agg = (
        num.groupby(cles, dropna=False)
        .agg(
            valeur=("value_num", "sum"),
            nb_combinaisons=("categoryOptionCombo", "nunique"),
            nb_valeurs=("value", "size"),
        )
        .reset_index()
        .rename(columns={"annee_extraction": "annee"})
        .sort_values([c if c != "annee_extraction" else "annee" for c in cles])
    )
    return agg


# ---- Complétude : seuils, couleurs, légende ----
LIB_ROUGE, LIB_JAUNE, LIB_VERT, LIB_NA = "0–70 % (rouge)", "70–90 % (jaune)", "90–100 % (vert)", "Pas de données"
COULEURS_CLASSES = {LIB_ROUGE: "#d73027", LIB_JAUNE: "#f4d03f", LIB_VERT: "#1a9850", LIB_NA: "#bdbdbd"}
LEGENDE_COMPLETUDE = "🟥 0–70 % (rouge)   ·   🟨 70–90 % (jaune)   ·   🟩 90–100 % (vert)"
COL_MOY = "Complétude moyenne (%)"
METRIQUES_RAPPORTS = ["EXPECTED_REPORTS", "ACTUAL_REPORTS"]


def classe_completude(x) -> str:
    """0–70 : rouge ; 70–90 : jaune ; 90–100 : vert."""
    if pd.isna(x):
        return LIB_NA
    if x < 70:
        return LIB_ROUGE
    if x < 90:
        return LIB_JAUNE
    return LIB_VERT


def style_completude(valeur) -> str:
    if pd.isna(valeur):
        return ""
    couleur = COULEURS_CLASSES[classe_completude(valeur)]
    texte = "black" if couleur == COULEURS_CLASSES[LIB_JAUNE] else "white"
    return f"background-color: {couleur}; color: {texte}"


def tableau_colore(df: pd.DataFrame, colonnes_taux: list, formats: dict):
    """Styler : formats + couleurs de la légende sur les colonnes de taux."""
    sty = df.style.format(formats, na_rep="–")
    appliquer = getattr(sty, "map", None) or sty.applymap
    return appliquer(style_completude, subset=colonnes_taux)


def _analytics_rows(s, url, dx: str, pe: str, ou: str) -> pd.DataFrame:
    params = {"dimension": [f"dx:{dx}", f"pe:{pe}", f"ou:{ou}"], "skipMeta": "true"}
    r = s.get(f"{url}/analytics.json", params=params, timeout=300)
    r.raise_for_status()
    j = r.json()
    return pd.DataFrame(j.get("rows", []), columns=[h["name"] for h in j["headers"]])


def _periodes(annees, trimestres) -> str:
    return ";".join(f"{a}Q{q}" for a in annees for q in trimestres)


def completude(s, url, dataset, orgunit, annees, trimestres) -> pd.DataFrame:
    """Rapports attendus / saisis et taux de complétude par année, sur les trimestres
    choisis (analytics DHIS2). Couvre l'unité choisie et ses descendantes.
    Taux = rapports saisis / rapports attendus × 100."""
    annees = list(annees)
    dx = ";".join(f"{dataset}.{m}" for m in METRIQUES_RAPPORTS)
    df = _analytics_rows(s, url, dx, _periodes(annees, trimestres), orgunit)
    if df.empty:
        piv = pd.DataFrame(index=pd.Index(annees, name="annee"), columns=METRIQUES_RAPPORTS, dtype=float)
    else:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["metrique"] = df["dx"].str.split(".").str[-1]
        df["annee"] = df["pe"].str[:4].astype(int)
        piv = df.pivot_table(index="annee", columns="metrique", values="value", aggfunc="sum")
        piv = piv.reindex(index=pd.Index(annees, name="annee"), columns=METRIQUES_RAPPORTS)
    out = piv.reset_index().rename(
        columns={"EXPECTED_REPORTS": "rapports_attendus", "ACTUAL_REPORTS": "rapports_saisis"}
    )
    out.columns.name = None
    out["taux_completude"] = (out["rapports_saisis"] / out["rapports_attendus"] * 100).where(
        out["rapports_attendus"] > 0
    )
    return out[["annee", "rapports_attendus", "rapports_saisis", "taux_completude"]]


@st.cache_data(ttl=600, show_spinner=False)
def completude_cache(url, mode, user, password, token, dataset, orgunit, annees, trimestres) -> pd.DataFrame:
    s = make_session(mode, user, password, token)
    return completude(s, url, dataset, orgunit, annees, trimestres)


@st.cache_data(ttl=600, show_spinner=False)
def completude_niveau_cache(
    url, mode, user, password, token, dataset, racine, niveau, annees, trimestres
) -> pd.DataFrame:
    """Rapports attendus / saisis (somme sur les années et trimestres choisis) pour chaque
    unité du niveau demandé sous `racine`."""
    s = make_session(mode, user, password, token)
    dx = ";".join(f"{dataset}.{m}" for m in METRIQUES_RAPPORTS)
    df = _analytics_rows(s, url, dx, _periodes(annees, trimestres), f"LEVEL-{niveau};{racine}")
    cols = ["ou", "attendus", "saisis"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["metrique"] = df["dx"].str.split(".").str[-1]
    piv = df.pivot_table(index="ou", columns="metrique", values="value", aggfunc="sum")
    piv = piv.reindex(columns=METRIQUES_RAPPORTS).reset_index()
    piv.columns.name = None
    return piv.rename(columns={"EXPECTED_REPORTS": "attendus", "ACTUAL_REPORTS": "saisis"})[cols]


@st.cache_data(ttl=3600, show_spinner=False)
def unites_assignees(url, mode, user, password, token) -> list:
    """Unités d'organisation assignées à l'utilisateur connecté (consultation des données
    si définie, sinon saisie). À défaut, les unités de niveau 1."""
    s = make_session(mode, user, password, token)
    r = s.get(
        f"{url}/me.json",
        params={
            "fields": "organisationUnits[id,displayName,level],"
            "dataViewOrganisationUnits[id,displayName,level]"
        },
        timeout=60,
    )
    r.raise_for_status()
    j = r.json()
    unites = j.get("dataViewOrganisationUnits") or j.get("organisationUnits") or []
    if not unites:
        r = s.get(
            f"{url}/organisationUnits.json",
            params={"fields": "id,displayName,level", "filter": "level:eq:1", "paging": "false"},
            timeout=60,
        )
        r.raise_for_status()
        unites = r.json().get("organisationUnits", [])
    return unites


@st.cache_data(ttl=3600, show_spinner=False)
def geometries_niveau(url, mode, user, password, token, racine, niveau) -> list:
    """Unités du niveau demandé sous `racine`, avec leur géométrie."""
    s = make_session(mode, user, password, token)
    r = s.get(
        f"{url}/organisationUnits.json",
        params={
            "fields": "id,displayName,geometry",
            "filter": [f"level:eq:{niveau}", f"path:like:{racine}"],
            "paging": "false",
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json().get("organisationUnits", [])


def _points(coords):
    """Parcourt récursivement des coordonnées GeoJSON et renvoie des (lon, lat)."""
    if isinstance(coords, (list, tuple)):
        if len(coords) >= 2 and all(isinstance(x, (int, float)) for x in coords[:2]):
            yield coords[0], coords[1]
        else:
            for c in coords:
                yield from _points(c)


def _centre_geometrie(geom: dict):
    """Centroïde (lon, lat) du plus grand polygone d'une géométrie GeoJSON."""
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    meilleur, aire_max = None, -1.0
    for poly in polys:
        anneau = poly[0]
        aire = cx = cy = 0.0
        for (x0, y0), (x1, y1) in zip(anneau, anneau[1:] + anneau[:1]):
            croix = x0 * y1 - x1 * y0
            aire += croix
            cx += (x0 + x1) * croix
            cy += (y0 + y1) * croix
        aire /= 2
        if abs(aire) > 1e-12:
            c = (cx / (6 * aire), cy / (6 * aire))
        else:  # polygone dégénéré : moyenne des sommets
            c = (sum(p[0] for p in anneau) / len(anneau), sum(p[1] for p in anneau) / len(anneau))
        if abs(aire) > aire_max:
            meilleur, aire_max = c, abs(aire)
    return meilleur


def construire_carte(df_map: pd.DataFrame, features: list, etiquettes: bool = True):
    """Carte choroplèthe avec les classes de la légende (rouge / jaune / vert)."""
    import plotly.express as px

    pts = [pt for f in features for pt in _points(f["geometry"]["coordinates"])]
    lons, lats = [p[0] for p in pts], [p[1] for p in pts]
    centre = {"lon": (min(lons) + max(lons)) / 2, "lat": (min(lats) + max(lats)) / 2}
    etendue = max(max(lons) - min(lons), (max(lats) - min(lats)) * 1.5, 0.01)
    zoom = max(1.0, min(12.0, math.log2(450 / (etendue * 1.15))))
    args = dict(
        geojson={"type": "FeatureCollection", "features": features},
        locations="id",
        featureidkey="properties.id",
        color="Classe",
        color_discrete_map=COULEURS_CLASSES,
        category_orders={"Classe": list(COULEURS_CLASSES)},
        hover_name="Unité",
        hover_data={"id": False, "Classe": False, COL_MOY: ":.1f"},
        center=centre,
        zoom=zoom,
        opacity=0.8,
    )
    if hasattr(px, "choropleth_map"):
        fig = px.choropleth_map(df_map, map_style="carto-positron", **args)
    else:  # plotly < 5.24
        fig = px.choropleth_mapbox(df_map, mapbox_style="carto-positron", **args)
    if etiquettes:
        import plotly.graph_objects as go

        infos = df_map.set_index("id")
        lon, lat, texte = [], [], []
        for f in features:
            uid = f["properties"]["id"]
            if uid not in infos.index:
                continue
            c = _centre_geometrie(f["geometry"])
            if c is None:
                continue
            valeur = infos.at[uid, COL_MOY]
            val = "–" if pd.isna(valeur) else f"{valeur:.1f}".replace(".", ",") + " %"
            lon.append(c[0])
            lat.append(c[1])
            texte.append(f"{infos.at[uid, 'Unité']}<br><b>{val}</b>")
        trace = dict(
            lon=lon, lat=lat, text=texte, mode="text", hoverinfo="skip", showlegend=False,
            textfont=dict(size=11, color="black"),
        )
        fig.add_trace(
            go.Scattermap(**trace) if hasattr(go, "Scattermap") else go.Scattermapbox(**trace)
        )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=550, legend_title_text="Complétude moyenne"
    )
    return fig


# ---- Qualité des données : règles de validation ----
OPERATEURS = {
    "equal_to": "=",
    "not_equal_to": "≠",
    "greater_than": ">",
    "greater_than_or_equal_to": "≥",
    "less_than": "<",
    "less_than_or_equal_to": "≤",
    "compulsory_pair": "paire obligatoire",
    "exclusive_pair": "paire exclusive",
}
IMPORTANCE = {"HIGH": "Haute", "MEDIUM": "Moyenne", "LOW": "Faible"}
ORDRE_IMPORTANCE = {"Haute": 0, "Moyenne": 1, "Faible": 2}
RE_DE = re.compile(r"#\{([A-Za-z][A-Za-z0-9]{10})(?:[.}])")
RE_DS = re.compile(r"R\{([A-Za-z][A-Za-z0-9]{10})\.")


@st.cache_data(ttl=3600, show_spinner=False)
def groupes_validation(url, mode, user, password, token) -> list:
    s = make_session(mode, user, password, token)
    r = s.get(
        f"{url}/validationRuleGroups.json",
        params={"fields": "id,displayName,validationRules[id,displayName]", "paging": "false"},
        timeout=120,
    )
    r.raise_for_status()
    return sorted(r.json().get("validationRuleGroups", []), key=lambda g: g["displayName"])


@st.cache_data(ttl=3600, show_spinner=False)
def regles_validation(url, mode, user, password, token) -> dict:
    s = make_session(mode, user, password, token)
    r = s.get(
        f"{url}/validationRules.json",
        params={
            "fields": "id,displayName,instruction,importance,operator,"
            "leftSide[expression,description],rightSide[expression,description]",
            "paging": "false",
        },
        timeout=300,
    )
    r.raise_for_status()
    return {x["id"]: x for x in r.json().get("validationRules", [])}


@st.cache_data(ttl=3600, show_spinner=False)
def elements_et_formulaires(url, mode, user, password, token) -> tuple:
    """Renvoie (élément -> [formulaires], élément -> nom, formulaire -> nom)."""
    s = make_session(mode, user, password, token)
    r = s.get(
        f"{url}/dataSets.json",
        params={
            "fields": "id,displayName,dataSetElements[dataElement[id,displayName]]",
            "paging": "false",
        },
        timeout=300,
    )
    r.raise_for_status()
    de_ds, noms_de, noms_formulaires = {}, {}, {}
    for ds in r.json().get("dataSets", []):
        noms_formulaires[ds["id"]] = ds["displayName"]
        for dse in ds.get("dataSetElements", []):
            de = dse.get("dataElement") or {}
            if de.get("id"):
                de_ds.setdefault(de["id"], []).append(ds["id"])
                noms_de[de["id"]] = de.get("displayName", de["id"])
    return de_ds, noms_de, noms_formulaires


@st.cache_data(ttl=600, show_spinner=False)
def analyse_validation(url, mode, user, password, token, groupe, orgunit, debut, fin) -> list:
    """Exécute les règles d'un groupe (sans persister ni notifier) et renvoie les violations."""
    s = make_session(mode, user, password, token)
    r = s.post(
        f"{url}/dataAnalysis/validationRules",
        json={
            "vrg": groupe,
            "ou": orgunit,
            "startDate": debut,
            "endDate": fin,
            "persist": False,
            "notification": False,
        },
        timeout=900,
    )
    r.raise_for_status()
    j = r.json()
    if isinstance(j, dict):  # certaines versions encapsulent la liste
        j = next((v for v in j.values() if isinstance(v, list)), [])
    return j


@st.cache_data(ttl=600, show_spinner=False)
def valeurs_aberrantes(
    url, mode, user, password, token, formulaires, orgunit, debut, fin, algo, seuil, max_resultats
) -> list:
    """Valeurs aberrantes d'un ou plusieurs formulaires (API outlierDetection)."""
    s = make_session(mode, user, password, token)
    params = {
        "ds": list(formulaires),
        "ou": orgunit,
        "startDate": debut,
        "endDate": fin,
        "algorithm": algo,
        "maxResults": max_resultats,
    }
    if algo != "MIN_MAX":
        params["threshold"] = seuil
    r = s.get(f"{url}/outlierDetection.json", params=params, timeout=900)
    r.raise_for_status()
    j = r.json()
    if isinstance(j, dict):
        j = j.get("outlierValues") or next((v for v in j.values() if isinstance(v, list)), [])
    return j


def formulaires_a_ouvrir(regle: dict, de_ds: dict, noms_ds: dict) -> tuple:
    """(noms des formulaires, éléments concernés, ID des formulaires) pour une règle."""
    expr = " ".join(
        ((regle.get(c) or {}).get("expression") or "") for c in ("leftSide", "rightSide")
    )
    de_ids = list(dict.fromkeys(RE_DE.findall(expr)))
    compte = {}
    for d in de_ids:
        for ds in de_ds.get(d, []):
            compte[ds] = compte.get(ds, 0) + 1
    for ds in RE_DS.findall(expr):
        compte[ds] = compte.get(ds, 0) + 1
    if not compte:
        return "", de_ids, []
    maxi = max(compte.values())
    ids = sorted((ds for ds, n in compte.items() if n == maxi), key=lambda d: noms_ds.get(d, d))
    return " | ".join(noms_ds.get(d, d) for d in ids), de_ids, ids


def _id(x):
    """ID d'une référence DHIS2 (chaîne ou objet {"id": ...})."""
    return x.get("id") if isinstance(x, dict) else x


@st.cache_data(ttl=600, show_spinner=False)
def soumissions_formulaire(url, mode, user, password, token, dataset, orgunit, debut, fin) -> list:
    """Formulaires marqués comme complétés (soumis) : unité, période, date, utilisateur."""
    s = make_session(mode, user, password, token)
    r = s.get(
        f"{url}/completeDataSetRegistrations.json",
        params={"dataSet": dataset, "orgUnit": orgunit, "children": "true", "startDate": debut, "endDate": fin},
        timeout=600,
    )
    r.raise_for_status()
    j = r.json()
    regs = j.get("completeDataSetRegistrations", []) if isinstance(j, dict) else j
    sorties, vus = [], set()
    for x in regs:
        if x.get("completed") is False:
            continue
        cle = (_id(x.get("organisationUnit")), _id(x.get("period")))
        if cle in vus or not all(cle):
            continue
        vus.add(cle)
        sorties.append(
            {"ou": cle[0], "pe": cle[1], "date": str(x.get("date") or x.get("lastUpdated") or "")[:10],
             "par": x.get("storedBy") or ""}
        )
    return sorties


@st.cache_data(ttl=600, show_spinner=False)
def valeurs_saisies(url, mode, user, password, token, dataset, orgunit, debut, fin) -> pd.DataFrame:
    """(unité, période, élément de données) pour lesquels au moins une valeur existe."""
    s = make_session(mode, user, password, token)
    r = s.get(
        f"{url}/dataValueSets.json",
        params={"dataSet": dataset, "orgUnit": orgunit, "children": "true", "startDate": debut, "endDate": fin},
        timeout=900,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json().get("dataValues", []))
    if df.empty:
        return pd.DataFrame(columns=["orgUnit", "period", "dataElement"])
    return df[["orgUnit", "period", "dataElement"]].drop_duplicates()


def calculer_manquantes(soumissions, valeurs, elements, noms_de, nom_formulaire, ds_id, noms_unites):
    """Pour chaque formulaire soumis, liste les éléments de données sans aucune valeur.
    Renvoie (lignes par soumission, détail par élément manquant)."""
    presents = {}
    for ou, pe, de in valeurs.itertuples(index=False):
        presents.setdefault((ou, pe), set()).add(de)
    attendus = set(elements)
    lignes, details = [], []
    for sub in soumissions:
        saisis = presents.get((sub["ou"], sub["pe"]), set()) & attendus
        manquants = [d for d in elements if d not in saisis]
        if not manquants:
            continue
        unite = noms_unites.get(sub["ou"], sub["ou"])
        noms = [noms_de.get(d, d) for d in manquants]
        liste = "; ".join(noms[:12]) + (f" … (+{len(noms) - 12})" if len(noms) > 12 else "")
        lignes.append(
            {
                "Formulaire": nom_formulaire,
                "Unité d'organisation": unite,
                "Période": sub["pe"],
                "Soumis le": sub["date"],
                "Soumis par": sub["par"],
                "Éléments attendus": len(elements),
                "Éléments renseignés": len(saisis),
                "Éléments manquants": len(manquants),
                "Taux de renseignement (%)": round(len(saisis) / len(elements) * 100, 1),
                "Formulaire vide": "Oui" if not saisis else "Non",
                "Liste des éléments manquants": liste,
                "Lien de saisie": lien_saisie(ds_id, sub["ou"], sub["pe"]),
            }
        )
        details += [
            {"Formulaire": nom_formulaire, "Unité d'organisation": unite, "Période": sub["pe"],
             "Élément de données manquant": n}
            for n in noms
        ]
    return lignes, details


LIEN_SAISIE_DEFAUT = (
    "{base}/dhis-web-aggregate-data-entry/index.html#/?dataSetId={ds}&orgUnitId={ou}&periodId={pe}"
)


def base_dhis2(api_url: str) -> str:
    """URL de l'instance, sans le suffixe /api (ou /api/NN)."""
    return re.sub(r"/api(/\d+)?/?$", "", api_url)


def lien_saisie(ds, ou, pe):
    """Lien vers l'écran de saisie (application Saisie de données) pour corriger une valeur.
    Modèle modifiable via DHIS2_ENTRY_URL_TEMPLATE ({base}, {ds}, {ou}, {pe})."""
    if not (ds and ou and pe):
        return None
    modele = secret("DHIS2_ENTRY_URL_TEMPLATE", LIEN_SAISIE_DEFAUT)
    return modele.format(base=base_dhis2(url), ds=ds, ou=ou, pe=pe)


def config_lien() -> dict:
    try:
        return {"Lien de saisie": st.column_config.LinkColumn("Saisie", display_text="Ouvrir ↗")}
    except TypeError:  # anciennes versions de Streamlit
        return {"Lien de saisie": st.column_config.LinkColumn("Saisie")}


def filtre_multi(label: str, options, key: str) -> list:
    """Multiselect de filtre ; retire les anciennes sélections devenues invalides."""
    options = [o for o in options if o not in (None, "")]
    st.session_state[key] = [v for v in st.session_state.get(key, []) if v in options]
    return st.multiselect(label, options=options, key=key)


COULEURS_IMPORTANCE = {"Haute": "#d73027", "Moyenne": "#f39c12", "Faible": "#f4d03f"}


def graphique_compte(
    df: pd.DataFrame,
    col: str,
    titre: str,
    top: int = 10,
    ordre: str = "valeur",
    vertical: bool = False,
    ordre_fixe: list = None,
    couleurs: dict = None,
):
    """Graphique en barres : nombre de lignes par modalité de `col`.
    - horizontal (défaut) trié par valeur : classements (top N) ;
    - vertical trié par libellé : évolution dans le temps ;
    - `ordre_fixe` / `couleurs` : ordre et couleurs imposés (ex. importance)."""
    import altair as alt

    compte = df[col].replace("", "(vide)").fillna("(vide)").astype(str).value_counts()
    if ordre_fixe:
        ordre_fixe = list(ordre_fixe) + [i for i in compte.index if i not in ordre_fixe]
        compte = compte.reindex(ordre_fixe, fill_value=0)
    compte = compte.reset_index()
    compte.columns = ["Libellé", "Nombre"]
    if ordre == "valeur" and not ordre_fixe:
        compte = compte.head(top)

    if ordre_fixe:
        tri = ordre_fixe
    elif ordre == "valeur":
        tri = "-y" if vertical else "-x"
    else:
        tri = "ascending"

    if couleurs:
        palette = {**couleurs, "(vide)": "#bdbdbd"}
        modalites = [m for m in compte["Libellé"] if m in palette]
        couleur = alt.Color(
            "Libellé:N",
            scale=alt.Scale(domain=modalites, range=[palette[m] for m in modalites]),
            legend=None,
        )
    else:
        couleur = alt.value("#2E5C8A")

    if vertical:
        axe_x = alt.X(
            "Libellé:N",
            sort=tri,
            title=None,
            axis=alt.Axis(labelAngle=-45 if len(compte) > 6 else 0, labelLimit=160),
        )
        axe_y = alt.Y("Nombre:Q", title=None, axis=alt.Axis(tickMinStep=1))
        hauteur = 280
    else:
        axe_x = alt.X("Nombre:Q", title=None, axis=alt.Axis(tickMinStep=1))
        axe_y = alt.Y("Libellé:N", sort=tri, title=None, axis=alt.Axis(labelLimit=260))
        hauteur = max(70, 26 * len(compte) + 20)

    st.markdown(f"**{titre}**")
    graphe = (
        alt.Chart(compte)
        .mark_bar()
        .encode(x=axe_x, y=axe_y, color=couleur, tooltip=["Libellé", "Nombre"])
        .properties(height=hauteur)
    )
    st.altair_chart(graphe, use_container_width=True)


def vers_nombre(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---- Tableaux de bord : chargement des tableaux croisés ----
CHAMPS_VISUALISATION = (
    "id,displayName,type,"
    "columns[dimension,items[dimensionItem,id]],rows[dimension,items[dimensionItem,id]],"
    "filters[dimension,items[dimensionItem,id]],dataDimensionItems[:all],relativePeriods,"
    "organisationUnitLevels,itemOrganisationUnitGroups[id],userOrganisationUnit,"
    "userOrganisationUnitChildren,userOrganisationUnitGrandChildren"
)
LIBELLES_DIMENSIONS = {
    "dx": "Donnée",
    "pe": "Période",
    "ou": "Unité d'organisation",
    "co": "Combinaison de catégories",
}


def periodes_relatives(viz: dict) -> list:
    """Périodes relatives cochées (ex. last12Months -> LAST_12_MONTHS)."""
    rp = viz.get("relativePeriods") or {}
    return [
        re.sub(r"(?<=[a-z])(?=[A-Z0-9])|(?<=[0-9])(?=[A-Z])", "_", k).upper()
        for k, v in rp.items()
        if v is True
    ]


def elements_dx_depuis_dimensions(viz: dict) -> list:
    """Éléments de la dimension « données » à partir de dataDimensionItems (si items vide)."""
    sortie = []
    for d in viz.get("dataDimensionItems", []) or []:
        t = d.get("dataDimensionItemType")
        try:
            if t == "INDICATOR":
                sortie.append(d["indicator"]["id"])
            elif t == "DATA_ELEMENT":
                sortie.append(d["dataElement"]["id"])
            elif t == "DATA_ELEMENT_OPERAND":
                op = d["dataElementOperand"]
                coc = (op.get("categoryOptionCombo") or {}).get("id")
                sortie.append(op["dataElement"]["id"] + (f".{coc}" if coc else ""))
            elif t == "REPORTING_RATE":
                rr = d["reportingRate"]
                sortie.append(f"{rr['dataSet']['id']}.{rr['metric']}")
            elif t == "PROGRAM_INDICATOR":
                sortie.append(d["programIndicator"]["id"])
        except (KeyError, TypeError):
            continue
    return sortie


def requete_analytics(viz: dict, periodes: list = None, unites_ou: list = None) -> dict:
    """Paramètres analytics reconstruits depuis la définition de la visualisation.
    Si `periodes` est fourni (ex. ["2026", "202611"]), la dimension « période » de la
    visualisation est remplacée par ces périodes explicites (filtrage année/mois)."""

    def dimension(obj: dict) -> str:
        dim = obj.get("dimension")
        items = [i.get("dimensionItem") or i.get("id") for i in obj.get("items", []) or []]
        items = [i for i in items if i]
        if dim == "dx" and not items:
            items = elements_dx_depuis_dimensions(viz)
        if dim == "pe":
            items = list(periodes) if periodes is not None else items + periodes_relatives(viz)
        if dim == "ou":
            if unites_ou is not None:
                # filtre utilisateur : remplace les unités enregistrées dans la visualisation
                items = list(unites_ou)
            else:
                items += [f"LEVEL-{n}" for n in viz.get("organisationUnitLevels", []) or []]
                items += [f"OU_GROUP-{g['id']}" for g in viz.get("itemOrganisationUnitGroups", []) or []]
                if viz.get("userOrganisationUnit"):
                    items.append("USER_ORGUNIT")
                if viz.get("userOrganisationUnitChildren"):
                    items.append("USER_ORGUNIT_CHILDREN")
                if viz.get("userOrganisationUnitGrandChildren"):
                    items.append("USER_ORGUNIT_GRANDCHILDREN")
        items = list(dict.fromkeys(items))
        return dim + (":" + ";".join(items) if items else "")

    lignes = [o["dimension"] for o in viz.get("rows", []) or []]
    colonnes = [o["dimension"] for o in viz.get("columns", []) or []]
    params = {
        "dimension": [dimension(o) for o in (viz.get("columns") or []) + (viz.get("rows") or [])],
        "skipMeta": "false",
        "displayProperty": "NAME",
    }
    filtres = [dimension(o) for o in viz.get("filters", []) or []]
    if filtres:
        params["filter"] = filtres
    if periodes is not None and not any(d.startswith("pe") for d in params["dimension"]):
        # la visualisation ne porte pas de dimension « période » : on filtre quand même
        params.setdefault("filter", []).append("pe:" + ";".join(periodes))
    if unites_ou is not None and not any(d.startswith("ou") for d in params["dimension"]):
        params.setdefault("filter", []).append("ou:" + ";".join(unites_ou))
    return {"params": params, "lignes": lignes, "colonnes": colonnes}


def construire_tableau(rep: dict, lignes: list, colonnes: list) -> pd.DataFrame:
    """Tableau croisé (lignes x colonnes) à partir d'une réponse analytics."""
    df = pd.DataFrame(rep.get("rows", []), columns=[h["name"] for h in rep["headers"]])
    meta = rep.get("metaData", {}) or {}
    items, dims_meta = meta.get("items", {}) or {}, meta.get("dimensions", {}) or {}
    nom = lambda i: (items.get(i) or {}).get("name", i)
    libelle = lambda d: LIBELLES_DIMENSIONS.get(d) or nom(d)
    lignes = [d for d in lignes if d in df.columns]
    colonnes = [d for d in colonnes if d in df.columns]
    if df.empty:
        return pd.DataFrame()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    for dim in lignes + colonnes:
        ordre = [nom(i) for i in dims_meta.get(dim, list(df[dim].unique()))]
        df[dim] = pd.Categorical(df[dim].map(nom), categories=list(dict.fromkeys(ordre)), ordered=True)

    if lignes and colonnes:
        t = df.pivot_table(index=lignes, columns=colonnes, values="value", aggfunc="first", observed=True)
    elif lignes:
        t = df.groupby(lignes, observed=True)["value"].first().to_frame("Valeur")
    elif colonnes:
        t = df.groupby(colonnes, observed=True)["value"].first().to_frame("Valeur").T
    else:
        t = df[["value"]].rename(columns={"value": "Valeur"})
    if isinstance(t.columns, pd.MultiIndex):
        t.columns = [" | ".join(str(x) for x in c) for c in t.columns]
    else:
        t.columns = [str(c) for c in t.columns]
    if lignes:
        t.index = t.index.set_names([libelle(d) for d in lignes])
        t = t.reset_index()
    return t


def charger_tableaux_dashboard(dashboard_id: str, inclure_graphiques: bool, periodes: list = None,
                               unites_ou: list = None, ou_mode: str = None) -> dict:
    """Charge les tableaux d'un tableau de bord : (a) définition des éléments, (b) définition
    de chaque visualisation, (c) données via analytics, reconstruites en tableau croisé."""
    s = make_session(mode, user, password, token)
    r = s.get(
        f"{url}/dashboards/{dashboard_id}.json",
        params={
            "fields": "id,displayName,dashboardItems[id,type,visualization[id,displayName,type],"
            "reportTable[id,displayName],map[id,displayName],eventVisualization[id,displayName,type],"
            "eventChart[id,displayName],eventReport[id,displayName],text]"
        },
        timeout=120,
    )
    r.raise_for_status()
    dash = r.json()

    a_charger, autres = [], []
    for it in dash.get("dashboardItems", []) or []:
        typ = it.get("type")
        viz = it.get("visualization")
        if viz:
            if viz.get("type") == "PIVOT_TABLE" or inclure_graphiques:
                a_charger.append(("visualizations", viz["id"], viz.get("displayName") or viz["id"], viz.get("type")))
            else:
                autres.append(f"{viz.get('displayName', viz['id'])} ({viz.get('type', typ)})")
        elif it.get("reportTable"):
            rt = it["reportTable"]
            a_charger.append(("reportTables", rt["id"], rt.get("displayName") or rt["id"], "REPORT_TABLE"))
        else:
            ref = next(
                (it[k] for k in ("map", "eventVisualization", "eventChart", "eventReport") if it.get(k)), None
            )
            autres.append(f"{(ref or {}).get('displayName', typ)} ({typ})")

    tables, erreurs = [], []
    for ressource, uid, nom_viz, type_viz in a_charger:
        try:
            rv = s.get(f"{url}/{ressource}/{uid}.json", params={"fields": CHAMPS_VISUALISATION}, timeout=120)
            rv.raise_for_status()
            viz = rv.json()
            q = requete_analytics(viz, periodes, unites_ou)
            params_analytics = dict(q["params"])
            if ou_mode:
                params_analytics["ouMode"] = ou_mode
            ra = s.get(f"{url}/analytics.json", params=params_analytics, timeout=300)
            ra.raise_for_status()
            rep = ra.json()
            t = construire_tableau(rep, q["lignes"], q["colonnes"])
            tables.append(
                {
                    "id": uid,
                    "nom": viz.get("displayName") or nom_viz,
                    "type": type_viz,
                    "df": t,
                    "lignes": q["lignes"],
                    "colonnes": q["colonnes"],
                    "requete": q["params"],
                    "nb_lignes_analytics": len(rep.get("rows", [])),
                }
            )
        except Exception as e:
            erreurs.append(f"{nom_viz} : {e}")
    return {
        "id": dashboard_id,
        "nom": dash.get("displayName", dashboard_id),
        "tables": tables,
        "autres": autres,
        "erreurs": erreurs,
    }


# ------------------------------------------------------------------
# Barre latérale : connexion
# ------------------------------------------------------------------
st.sidebar.header("🔌 Connexion DHIS2")
url = st.sidebar.text_input(
    "URL de l'API", value=secret("DHIS2_URL", "https://play.dhis2.org/demo/api")
).rstrip("/")
mode = st.sidebar.radio("Authentification", ["Login / mot de passe", "Token (PAT)"])

user = password = token = ""
if mode == "Login / mot de passe":
    user = st.sidebar.text_input("Utilisateur", value=secret("DHIS2_USER", "admin"))
    password = st.sidebar.text_input(
        "Mot de passe", value=secret("DHIS2_PASSWORD", "district"), type="password"
    )
else:
    token = st.sidebar.text_input("Token", value=secret("DHIS2_TOKEN"), type="password")

connecte = st.sidebar.button("Se connecter / rafraîchir les listes")
if connecte:
    st.cache_data.clear()

# ------------------------------------------------------------------
# Corps de la page
# ------------------------------------------------------------------
st.title("📊 DHIS2 : extraction et complétude")

try:
    df_ds = lister(url, mode, user, password, token, "dataSets")
    df_ou = lister(url, mode, user, password, token, "organisationUnits")
except Exception as e:
    st.error(f"Connexion impossible : {e}")
    st.stop()

st.sidebar.success(f"{len(df_ds)} formulaires, {len(df_ou)} unités d'organisation")

noms_ds = dict(zip(df_ds["id"], df_ds["displayName"]))
noms_ou = dict(zip(df_ou["id"], df_ou["displayName"]))
annee_fin = datetime.date.today().year
options_annees = list(range(annee_fin, annee_fin - 21, -1))
annees_defaut = list(range(annee_fin - 4, annee_fin + 1))


def choisir_unite(prefixe: str) -> str:
    """Sélecteur d'unité d'organisation (liste ou UID manuel), clés propres à chaque onglet."""
    if st.checkbox("Saisir l'UID de l'unité manuellement", key=f"{prefixe}_manuel"):
        return st.text_input("UID de l'unité d'organisation", key=f"{prefixe}_ou_txt")
    return st.selectbox(
        "Unité d'organisation",
        options=df_ou["id"].tolist(),
        format_func=lambda i: noms_ou.get(i, i),
        key=f"{prefixe}_ou",
    )


# ------------------------------------------------------------------
# Onglet 1 : extraction (un formulaire)
# ------------------------------------------------------------------
def onglet_extraction():
    st.caption("Un formulaire × une unité d'organisation × les années choisies")

    col1, col2 = st.columns(2)
    with col1:
        dataset_id = st.selectbox(
            "Formulaire (dataSet)",
            options=df_ds["id"].tolist(),
            format_func=lambda i: noms_ds.get(i, i),
            key="ext_ds",
        )
    with col2:
        orgunit_id = choisir_unite("ext")

    col3, col4 = st.columns(2)
    with col3:
        annees = sorted(
            st.multiselect(
                "Années à extraire",
                options=options_annees,
                default=annees_defaut,
                key="ext_annees",
            )
        )
    with col4:
        enfants = st.checkbox("Inclure les unités enfants", value=False, key="ext_enfants")

    cumul_ou = st.checkbox(
        "Cumuler aussi toutes les unités d'organisation (un seul total par élément et par année)",
        value=False,
        key="ext_cumul",
        help="Si décoché et que les unités enfants sont incluses, une ligne par unité, élément et année.",
    )

    if annees:
        st.info("Années extraites : " + ", ".join(str(a) for a in annees))
    else:
        st.warning("Sélectionnez au moins une année.")

    if st.button(
        "🚀 Lancer l'extraction", type="primary", disabled=not (orgunit_id and annees), key="ext_go"
    ):
        s = make_session(mode, user, password, token)
        morceaux = []
        barre = st.progress(0.0, text="Extraction en cours…")
        try:
            for i, a in enumerate(annees):
                df = extraire_annee(s, url, dataset_id, orgunit_id, enfants, a)
                if not df.empty:
                    morceaux.append(df)
                barre.progress((i + 1) / len(annees), text=f"Année {a} : {len(df)} valeurs")
        except Exception as e:
            st.error(f"Erreur pendant l'extraction : {e}")
            return
        barre.empty()

        if not morceaux:
            st.warning(
                "Aucune donnée trouvée. Vérifiez le formulaire, l'unité d'organisation "
                "et vos droits de lecture."
            )
            st.session_state.pop("resultat", None)
            st.session_state.pop("resultat_agg", None)
        else:
            data = pd.concat(morceaux, ignore_index=True)
            # Noms des éléments de données
            r = s.get(
                f"{url}/dataElements.json",
                params={"fields": "id,displayName", "paging": "false"},
                timeout=120,
            )
            r.raise_for_status()
            noms_de = {x["id"]: x["displayName"] for x in r.json()["dataElements"]}
            data["dataElement_nom"] = data["dataElement"].map(noms_de)
            data["orgUnit_nom"] = data["orgUnit"].map(noms_ou)
            data["value_num"] = pd.to_numeric(data["value"], errors="coerce")
            # Agrégation faite sur les ID (évite de fusionner deux combinaisons homonymes)
            agg = agreger(data, cumul_ou)
            # Remplace l'ID de la combinaison de catégories par son nom
            noms_coc = dictionnaire_noms(url, mode, user, password, token, "categoryOptionCombos")
            data["categoryOptionCombo"] = (
                data["categoryOptionCombo"].map(noms_coc).fillna(data["categoryOptionCombo"])
            )
            st.session_state["resultat"] = data
            st.session_state["resultat_agg"] = agg
            st.session_state["nom_fichier"] = f"dhis2_{dataset_id}_{orgunit_id}"

    if "resultat" not in st.session_state:
        return

    data = st.session_state["resultat"]
    agg = st.session_state["resultat_agg"]
    nom = st.session_state["nom_fichier"]

    st.subheader("✅ Contrôles (à comparer avec DHIS2)")
    ctrl = (
        data.groupby("annee_extraction")
        .agg(nb_lignes_brutes=("value", "size"), somme_valeurs=("value_num", "sum"))
        .reset_index()
    )
    ctrl["somme_agregee"] = ctrl["annee_extraction"].map(agg.groupby("annee")["valeur"].sum())
    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(ctrl, hide_index=True, use_container_width=True)
    with c2:
        st.metric("Total de lignes brutes", f"{len(data):,}".replace(",", " "))
        nb_non_num = int(data["value_num"].isna().sum())
        if nb_non_num:
            st.caption(f"{nb_non_num} valeur(s) non numérique(s) exclue(s) de l'agrégation.")

    st.subheader("Données agrégées par année et par élément de données")
    st.dataframe(agg, use_container_width=True, height=400)
    st.download_button(
        "⬇️ Télécharger les données agrégées (CSV)",
        data=agg.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{nom}_agrege.csv",
        mime="text/csv",
    )

    with st.expander("Données brutes (détail par combinaison de catégories)"):
        st.dataframe(data, use_container_width=True, height=400)
        st.download_button(
            "⬇️ Télécharger les données brutes (CSV)",
            data=data.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{nom}_brut.csv",
            mime="text/csv",
        )


# ------------------------------------------------------------------
# Onglet 2 : complétude (un ou plusieurs formulaires, indépendant de l'extraction)
# ------------------------------------------------------------------
def calculer_carte(choix, annees, trimestres) -> dict:
    """Complétude moyenne des formulaires choisis pour chaque unité du niveau situé
    juste sous l'unité assignée à l'utilisateur (niveau 1 → régions, niveau 2 → districts)."""
    unites = unites_assignees(url, mode, user, password, token)
    erreurs, mesures, geoms, niveaux = [], [], {}, set()
    for u in unites:
        niveau = int(u.get("level", 1)) + 1
        niveaux.add(niveau)
        try:
            for g in geometries_niveau(url, mode, user, password, token, u["id"], niveau):
                geoms[g["id"]] = g
        except Exception as e:
            erreurs.append(f"Géométries ({u['displayName']}) : {e}")
        for ds_id in choix:
            try:
                d = completude_niveau_cache(
                    url, mode, user, password, token, ds_id, u["id"], niveau,
                    tuple(annees), tuple(trimestres),
                )
                mesures.append(d.assign(formulaire=noms_ds.get(ds_id, ds_id)))
            except Exception as e:
                erreurs.append(f"{noms_ds.get(ds_id, ds_id)} ({u['displayName']}) : {e}")

    if mesures:
        allv = pd.concat(mesures, ignore_index=True)
        allv["taux"] = (allv["saisis"] / allv["attendus"] * 100).where(allv["attendus"] > 0)
        piv = allv.pivot_table(index="ou", columns="formulaire", values="taux", aggfunc="mean")
    else:
        piv = pd.DataFrame()
    ids = sorted(set(piv.index) | set(geoms))
    tab = piv.reindex(ids)
    tab[COL_MOY] = tab.mean(axis=1, skipna=True) if len(tab.columns) else float("nan")
    tab.index.name = "id"
    tab = tab.reset_index()
    tab["Unité"] = tab["id"].map(lambda i: (geoms.get(i) or {}).get("displayName") or noms_ou.get(i, i))
    tab["Classe"] = tab[COL_MOY].map(classe_completude)

    features = [
        {"type": "Feature", "properties": {"id": g["id"]}, "geometry": g["geometry"]}
        for g in geoms.values()
        if (g.get("geometry") or {}).get("type") in ("Polygon", "MultiPolygon")
    ]
    libelle = {frozenset({2}): "région", frozenset({3}): "district"}.get(frozenset(niveaux), "unité")
    return {
        "table": tab,
        "features": features,
        "libelle": libelle,
        "racines": ", ".join(u["displayName"] for u in unites),
        "erreurs": erreurs,
    }


def libelle_niveau(niveau: int) -> tuple:
    """(singulier, pluriel) du niveau hiérarchique."""
    return {2: ("DRS", "DRS"), 3: ("district", "districts")}.get(niveau, ("unité", "unités"))


def fmt_pct(x) -> str:
    return "–" if pd.isna(x) else f"{x:.2f}".replace(".", ",") + " %"


def fmt_nb(n: int, niveau: int) -> str:
    sing, plur = libelle_niveau(niveau)
    return f"{n} {sing if n <= 1 else plur}"


def evol_points(a, b) -> tuple:
    """Évolution d'un taux en points. Renvoie (texte, sens) ; sens : +1 mieux, -1 recul, 0 stable."""
    if pd.isna(a) or pd.isna(b):
        return "–", 0
    d = b - a
    if abs(d) < 0.005:
        return "Stable", 0
    mot = "point" if abs(d) < 2 else "points"
    signe = "+" if d > 0 else "-"
    return f"{signe} {abs(d):.2f}".replace(".", ",") + f" {mot}", (1 if d > 0 else -1)


def evol_compte(a: int, b: int, plus_est_mieux: bool) -> tuple:
    d = b - a
    if d == 0:
        return "Stable", 0
    mieux = (d > 0) == plus_est_mieux
    return f"{'Progression' if mieux else 'Recul'} ({d:+d})", (1 if mieux else -1)


def indicateurs_periode(choix, unites, annees, trimestres) -> dict:
    """Complétude moyenne (niveau de l'unité assignée) et complétude moyenne par unité
    aux deux niveaux situés sous elle (ex. : DRS puis districts)."""
    niv_racine = min(int(u.get("level", 1)) for u in unites)
    racines = [u for u in unites if int(u.get("level", 1)) == niv_racine]
    na, nb = niv_racine + 1, niv_racine + 2
    cadres = {na: [], nb: []}
    for u in racines:
        for niv in (na, nb):
            for ds_id in choix:
                d = completude_niveau_cache(
                    url, mode, user, password, token, ds_id, u["id"], niv,
                    tuple(annees), tuple(trimestres),
                )
                cadres[niv].append(d.assign(formulaire=noms_ds.get(ds_id, ds_id)))
    out = {"na": na, "nb": nb, "racines": racines, "unites": {}}
    for niv, frames in cadres.items():
        allv = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["ou", "attendus", "saisis", "formulaire"])
        )
        allv[["attendus", "saisis"]] = allv[["attendus", "saisis"]].astype(float)
        allv["taux"] = (allv["saisis"] / allv["attendus"] * 100).where(allv["attendus"] > 0)
        if allv["taux"].notna().any():
            out["unites"][niv] = (
                allv.pivot_table(index="ou", columns="formulaire", values="taux", aggfunc="mean")
                .mean(axis=1)
            )
        else:
            out["unites"][niv] = pd.Series(dtype=float)
        if niv == na:  # moyenne globale : taux cumulé par formulaire, puis moyenne des formulaires
            g = allv.groupby("formulaire")[["attendus", "saisis"]].sum()
            taux_form = (g["saisis"] / g["attendus"] * 100).where(g["attendus"] > 0)
            out["moyenne"] = taux_form.mean() if len(taux_form) else float("nan")
    return out


def tableau_comparaison(ia: dict, ib: dict, label_a: str, label_b: str) -> tuple:
    """Construit le tableau de comparaison (DataFrame) et la liste des sens d'évolution."""
    na, nb = ia["na"], ia["nb"]
    ua, ub = ia["unites"], ib["unites"]
    maj = lambda t: t[0].upper() + t[1:]
    plur_a, plur_b = maj(libelle_niveau(na)[1]), maj(libelle_niveau(nb)[1])
    noms = ", ".join(u["displayName"] for u in ia["racines"])
    niv_racine = na - 1
    titre_moy = "Complétude moyenne nationale" if niv_racine == 1 else f"Complétude moyenne ({noms})"

    lignes, sens = [], []

    def ajouter(indicateur, va, vb, evol):
        lignes.append({"Indicateur": indicateur, label_a: va, label_b: vb, "Évolution": evol[0]})
        sens.append(evol[1])

    ajouter(titre_moy, fmt_pct(ia["moyenne"]), fmt_pct(ib["moyenne"]), evol_points(ia["moyenne"], ib["moyenne"]))
    comptes = [
        (f"{plur_a} avec complétude > 90 %", na, lambda x: int((x > 90).sum()), True),
        (f"{plur_a} avec complétude < 75 %", na, lambda x: int((x < 75).sum()), False),
        (f"{plur_b} à 100 %", nb, lambda x: int((x >= 100).sum()), True),
        (f"{plur_b} < 50 %", nb, lambda x: int((x < 50).sum()), False),
    ]
    for titre, niv, f, plus_est_mieux in comptes:
        a, b = f(ua[niv]), f(ub[niv])
        ajouter(titre, fmt_nb(a, niv), fmt_nb(b, niv), evol_compte(a, b, plus_est_mieux))
    return pd.DataFrame(lignes), sens


def section_comparaison(res: dict):
    """Comparaison de deux périodes (trimestre ou année) parmi celles du dernier calcul."""
    options = {}
    for a in res["annees"]:
        for q in res["trimestres_liste"]:
            options[f"T{q} {a}"] = ((a,), (q,))
        options[f"Année {a}"] = ((a,), (1, 2, 3, 4))
    labels = list(options)

    st.subheader("Comparaison de deux périodes")
    c1, c2 = st.columns(2)
    with c1:
        la = st.selectbox("Période 1 (référence)", labels, index=0, key="cmp_a")
    with c2:
        lb = st.selectbox("Période 2 (comparée)", labels, index=min(1, len(labels) - 1), key="cmp_b")
    if la == lb:
        st.info("Choisissez deux périodes différentes pour les comparer.")
        return
    try:
        unites = unites_assignees(url, mode, user, password, token)
        with st.spinner("Calcul de la comparaison…"):
            ia = indicateurs_periode(res["choix"], unites, *options[la])
            ib = indicateurs_periode(res["choix"], unites, *options[lb])
    except Exception as e:
        st.warning(f"Comparaison impossible : {e}")
        return

    df, sens = tableau_comparaison(ia, ib, la, lb)
    st.markdown(f"**Comparaison {la} vs {lb}**")
    couleurs = {1: "color: #1a9850; font-weight: 600", -1: "color: #d73027; font-weight: 600", 0: ""}
    st.dataframe(
        df.style.apply(lambda col: [couleurs[x] for x in sens], subset=["Évolution"]),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Évolution = période 2 par rapport à la période 1. Complétude moyenne des formulaires "
        "sélectionnés ; les seuils s'appliquent à la moyenne de chaque unité. "
        "Vert : amélioration, rouge : recul."
    )


def onglet_completude():
    st.caption(
        "Rapports attendus, saisis et taux de complétude (analytics DHIS2) pour un ou "
        "plusieurs formulaires, sur l'unité d'organisation choisie et ses descendantes."
    )
    col1, col2 = st.columns(2)
    with col1:
        choix = st.multiselect(
            "Formulaires",
            options=df_ds["id"].tolist(),
            format_func=lambda i: noms_ds.get(i, i),
            key="comp_ds",
        )
    with col2:
        orgunit_id = choisir_unite("comp")
    col3, col4 = st.columns(2)
    with col3:
        annees = sorted(
            st.multiselect("Années", options=options_annees, default=annees_defaut, key="comp_annees")
        )
    with col4:
        trimestres = sorted(
            st.multiselect(
                "Trimestres",
                options=[1, 2, 3, 4],
                default=[1, 2, 3, 4],
                format_func=lambda q: f"T{q}",
                key="comp_trim",
            )
        )

    if st.button(
        "📈 Calculer la complétude",
        type="primary",
        disabled=not (choix and orgunit_id and annees and trimestres),
        key="comp_go",
    ):
        morceaux, erreurs = [], []
        barre = st.progress(0.0, text="Calcul en cours…")
        for i, ds_id in enumerate(choix):
            try:
                c = completude_cache(
                    url, mode, user, password, token, ds_id, orgunit_id, tuple(annees), tuple(trimestres)
                )
                morceaux.append(c.assign(formulaire=noms_ds.get(ds_id, ds_id)))
            except Exception as e:
                erreurs.append(f"{noms_ds.get(ds_id, ds_id)} : {e}")
            barre.progress((i + 1) / len(choix), text=noms_ds.get(ds_id, ds_id))
        barre.text("Calcul de la carte…")
        try:
            carte = calculer_carte(choix, annees, trimestres)
        except Exception as e:
            carte = None
            erreurs.append(f"Carte : {e}")
        barre.empty()
        st.session_state["comp_result"] = {
            "data": pd.concat(morceaux, ignore_index=True) if morceaux else None,
            "erreurs": erreurs,
            "unite": noms_ou.get(orgunit_id, orgunit_id),
            "trimestres": ", ".join(f"T{q}" for q in trimestres),
            "carte": carte,
            "choix": list(choix),
            "annees": list(annees),
            "trimestres_liste": list(trimestres),
        }

    res = st.session_state.get("comp_result")
    if not res:
        return
    if res["erreurs"]:
        st.warning(
            "Calcul impossible pour certains éléments (analytics non généré, droits "
            "insuffisants ou erreur serveur) :\n\n" + "\n\n".join(res["erreurs"])
        )

    comp = res["data"]
    if comp is not None:
        st.subheader(f"Complétude : {res['unite']}")
        st.caption(f"Trimestres retenus : {res['trimestres']}. Légende : {LEGENDE_COMPLETUDE}")
        affichage = comp.assign(Trimestres=res["trimestres"])[
            ["formulaire", "annee", "Trimestres", "rapports_attendus", "rapports_saisis", "taux_completude"]
        ].rename(
            columns={
                "formulaire": "Formulaire",
                "annee": "Année",
                "rapports_attendus": "Rapports attendus",
                "rapports_saisis": "Rapports saisis",
                "taux_completude": "Complétude (%)",
            }
        )
        st.dataframe(
            tableau_colore(
                affichage,
                ["Complétude (%)"],
                {"Rapports attendus": "{:.0f}", "Rapports saisis": "{:.0f}", "Complétude (%)": "{:.1f}"},
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("**Comparaison des taux de complétude par année**")
        graph = comp.assign(annee=comp["annee"].astype(str)).pivot_table(
            index="annee", columns="formulaire", values="taux_completude", aggfunc="first"
        )
        try:
            st.bar_chart(graph, stack=False, y_label="Complétude (%)", x_label="Année")
        except TypeError:  # anciennes versions de Streamlit
            st.bar_chart(graph)

    section_comparaison(res)

    carte = res.get("carte")
    if carte is None:
        return
    st.subheader(f"Carte : complétude moyenne par {carte['libelle']}")
    st.caption(
        f"Moyenne des taux de complétude des formulaires sélectionnés, par {carte['libelle']} "
        f"sous : {carte['racines']}. Légende : {LEGENDE_COMPLETUDE}"
    )
    tab = carte["table"]
    if tab.empty:
        st.info("Aucune unité trouvée pour ce niveau.")
        return

    ids_geo = {f["properties"]["id"] for f in carte["features"]}
    if ids_geo:
        etiquettes = st.checkbox(
            "Afficher les étiquettes (nom et valeur)", value=True, key="carte_etiquettes"
        )
        try:
            fig = construire_carte(
                tab[tab["id"].isin(ids_geo)],
                [f for f in carte["features"]],
                etiquettes,
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.warning("La carte nécessite plotly : `pip install plotly`.")
    else:
        st.info("Aucune géométrie (polygone) disponible pour ces unités : la carte ne peut pas être tracée.")

    cols_form = [c for c in tab.columns if c not in ("id", "Unité", "Classe", COL_MOY)]
    detail = tab[["Unité", COL_MOY] + cols_form].sort_values("Unité")
    st.dataframe(
        tableau_colore(detail, [COL_MOY] + cols_form, {c: "{:.1f}" for c in [COL_MOY] + cols_form}),
        hide_index=True,
        use_container_width=True,
    )


# ------------------------------------------------------------------
# Onglet 3 : qualité des données (règles de validation, valeurs aberrantes)
# ------------------------------------------------------------------
def synthese_et_tableaux_validation(res: dict):
    """Affiche filtres, synthèse visuelle et tableaux par groupe des erreurs de validation."""
    if res["erreurs"]:
        st.warning(
            "Analyse impossible pour certains groupes (droits insuffisants ou erreur serveur) :\n\n"
            + "\n\n".join(res["erreurs"])
        )
    if not res["resume"]:
        return
    st.subheader(f"Résultats : {res['unite']} ({res['periode']})")
    if not res["lignes"]:
        st.success("Aucune erreur détectée par les règles de validation sélectionnées. ✅")
        return

    df = pd.DataFrame(res["lignes"])
    total = len(df)
    df["_o"] = df["Importance"].map(ORDRE_IMPORTANCE).fillna(3)
    df = df.sort_values(["Groupe", "_o", "Règle", "Unité d'organisation", "Période"]).drop(columns="_o")

    with st.expander("🔎 Filtres", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            sel_imp = filtre_multi(
                "Importance",
                sorted(df["Importance"].unique(), key=lambda x: ORDRE_IMPORTANCE.get(x, 3)),
                "qd_f_imp",
            )
        with f2:
            sel_regle = filtre_multi("Règle", sorted(df["Règle"].dropna().unique()), "qd_f_regle")
        with f3:
            sel_ou = filtre_multi(
                "Unité d'organisation", sorted(df["Unité d'organisation"].dropna().unique()), "qd_f_ou"
            )
    for col, sel in (("Importance", sel_imp), ("Règle", sel_regle), ("Unité d'organisation", sel_ou)):
        if sel:
            df = df[df[col].isin(sel)]
    if df.empty:
        st.info("Aucune erreur ne correspond aux filtres choisis.")
        return

    st.metric("Erreurs à corriger (filtres appliqués)", f"{len(df):,}".replace(",", " "),
              help=f"Total sans filtre : {total}")

    st.markdown("### Synthèse")
    c1, c2 = st.columns(2)
    with c1:
        graphique_compte(df, "Groupe", "Erreurs par groupe de règles")
    with c2:
        graphique_compte(
            df, "Importance", "Erreurs par importance",
            ordre_fixe=["Haute", "Moyenne", "Faible"], couleurs=COULEURS_IMPORTANCE,
        )
    c3, c4 = st.columns(2)
    with c3:
        graphique_compte(df, "Règle", "Règles les plus souvent non respectées (top 10)")
    with c4:
        graphique_compte(df, "Unité d'organisation", "Unités avec le plus d'erreurs (top 10)")

    st.download_button(
        "⬇️ Télécharger les erreurs affichées (CSV)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="dhis2_erreurs_validation.csv",
        mime="text/csv",
    )

    st.markdown("### Détail par groupe de règles")
    for ligne in res["resume"]:
        groupe = ligne["Groupe"]
        sous = df[df["Groupe"] == groupe].drop(columns="Groupe")
        if sous.empty:
            st.markdown(f"#### {groupe}")
            st.caption("✅ Aucune erreur dans ce groupe (avec les filtres actuels).")
            continue
        with st.expander(f"❌ {groupe} : {len(sous)} erreur(s)", expanded=True):
            st.dataframe(
                sous,
                hide_index=True,
                use_container_width=True,
                height=min(500, 40 + 35 * len(sous)),
                column_config=config_lien(),
            )


def sous_onglet_validation(orgunit_id, debut, fin):
    st.caption(
        "Exécute les règles de validation, groupe par groupe, et liste les erreurs à corriger."
    )
    try:
        groupes = groupes_validation(url, mode, user, password, token)
    except Exception as e:
        st.error(f"Impossible de lire les groupes de règles de validation : {e}")
        return
    if not groupes:
        st.info("Aucun groupe de règles de validation n'est défini sur cette instance.")
        return
    noms_g = {g["id"]: g["displayName"] for g in groupes}
    nb_regles = {g["id"]: len(g.get("validationRules", [])) for g in groupes}
    choix = st.multiselect(
        "Groupes de règles de validation",
        options=list(noms_g),
        default=list(noms_g),
        format_func=lambda i: f"{noms_g[i]} ({nb_regles[i]} règle(s))",
        key="qd_groupes",
    )

    if st.button(
        "🔎 Lancer le contrôle des règles",
        type="primary",
        disabled=not (choix and orgunit_id and debut <= fin),
        key="qd_go",
    ):
        try:
            regles = regles_validation(url, mode, user, password, token)
            de_ds, noms_de, noms_formulaires = elements_et_formulaires(
                url, mode, user, password, token
            )
        except Exception as e:
            st.error(f"Impossible de lire les métadonnées (règles, formulaires) : {e}")
            return

        lignes, resume, erreurs = [], [], []
        barre = st.progress(0.0, text="Analyse en cours…")
        for i, g_id in enumerate(choix):
            barre.progress(i / len(choix), text=f"Groupe : {noms_g[g_id]}")
            try:
                violations = analyse_validation(
                    url, mode, user, password, token, g_id, orgunit_id, debut.isoformat(), fin.isoformat()
                )
            except Exception as e:
                erreurs.append(f"{noms_g[g_id]} : {e}")
                continue
            resume.append({"Groupe": noms_g[g_id], "Erreurs détectées": len(violations)})
            for v in violations:
                rid = v.get("validationRuleId") or (v.get("validationRule") or {}).get("id")
                regle = regles.get(rid, {})
                formulaire, de_ids, ds_ids = formulaires_a_ouvrir(regle, de_ds, noms_formulaires)
                imp = IMPORTANCE.get(v.get("importance") or regle.get("importance"), "")
                op = v.get("operator") or regle.get("operator") or ""
                gauche = vers_nombre(v.get("leftsideValue"))
                droite = vers_nombre(v.get("rightsideValue"))
                lignes.append(
                    {
                        "Groupe": noms_g[g_id],
                        "Règle": regle.get("displayName") or v.get("validationRuleDescription") or rid,
                        "Importance": imp,
                        "Unité d'organisation": v.get("organisationUnitDisplayName")
                        or v.get("organisationUnitId"),
                        "Hiérarchie": v.get("organisationUnitAncestorNames", ""),
                        "Période": v.get("periodDisplayName") or v.get("periodId"),
                        "Valeur gauche": gauche,
                        "Condition attendue": OPERATEURS.get(op, op),
                        "Valeur droite": droite,
                        "Écart": None if gauche is None or droite is None else gauche - droite,
                        "Formulaire(s) à ouvrir": formulaire,
                        "Lien de saisie": lien_saisie(
                            ds_ids[0] if ds_ids else None,
                            v.get("organisationUnitId"),
                            v.get("periodId"),
                        ),
                        "Éléments de données à vérifier": ", ".join(noms_de.get(d, d) for d in de_ids),
                        "Instruction de correction": regle.get("instruction", ""),
                    }
                )
        barre.empty()
        st.session_state["qd_result"] = {
            "lignes": lignes,
            "resume": resume,
            "erreurs": erreurs,
            "unite": noms_ou.get(orgunit_id, orgunit_id),
            "periode": f"{debut.isoformat()} → {fin.isoformat()}",
        }

    res = st.session_state.get("qd_result")
    if res:
        synthese_et_tableaux_validation(res)


ALGORITHMES = {
    "Z_SCORE": "Z-score (écart à la moyenne)",
    "MOD_Z_SCORE": "Z-score modifié (écart à la médiane)",
    "MIN_MAX": "Min-max (bornes définies dans DHIS2)",
}


def sous_onglet_aberrantes(orgunit_id, debut, fin):
    st.caption(
        "Repère les valeurs très éloignées de l'historique de l'unité (trop élevées ou trop basses), "
        "souvent des erreurs de saisie (virgule oubliée, zéro en trop…)."
    )
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        choix = st.multiselect(
            "Formulaires à analyser",
            options=df_ds["id"].tolist(),
            format_func=lambda i: noms_ds.get(i, i),
            key="ab_ds",
        )
    with col2:
        algo = st.selectbox(
            "Méthode", list(ALGORITHMES), format_func=lambda a: ALGORITHMES[a], key="ab_algo"
        )
    with col3:
        seuil = st.number_input(
            "Seuil", min_value=0.5, max_value=20.0, value=3.0, step=0.5,
            disabled=(algo == "MIN_MAX"), key="ab_seuil",
            help="Plus le seuil est bas, plus il y a de valeurs signalées. 3 est la valeur usuelle.",
        )
    max_res = st.number_input(
        "Nombre maximal de valeurs renvoyées", min_value=50, max_value=5000, value=500, step=50, key="ab_max"
    )

    if st.button(
        "📈 Rechercher les valeurs aberrantes",
        type="primary",
        disabled=not (choix and orgunit_id and debut <= fin),
        key="ab_go",
    ):
        try:
            de_ds, _noms_de, noms_formulaires = elements_et_formulaires(url, mode, user, password, token)
            with st.spinner("Analyse en cours…"):
                valeurs = valeurs_aberrantes(
                    url, mode, user, password, token, tuple(choix), orgunit_id,
                    debut.isoformat(), fin.isoformat(), algo, float(seuil), int(max_res),
                )
        except Exception as e:
            st.error(f"Analyse impossible (droits insuffisants, version de DHIS2 trop ancienne ou erreur serveur) : {e}")
            return
        lignes = []
        for v in valeurs:
            de = v.get("de")
            ds_ids = [d for d in de_ds.get(de, []) if d in choix] or de_ds.get(de, [])
            val, moy = vers_nombre(v.get("value")), vers_nombre(v.get("mean"))
            bas, haut = vers_nombre(v.get("lowerBound")), vers_nombre(v.get("upperBound"))
            if val is None:
                sens = ""
            elif haut is not None and val > haut:
                sens = "Trop élevée"
            elif bas is not None and val < bas:
                sens = "Trop basse"
            elif moy is not None:
                sens = "Trop élevée" if val > moy else "Trop basse"
            else:
                sens = ""
            arrondi = lambda x: None if x is None else round(x, 2)
            lignes.append(
                {
                    "Élément de données": v.get("deName") or de,
                    "Combinaison": v.get("cocName", ""),
                    "Unité d'organisation": v.get("ouName") or v.get("ou"),
                    "Période": v.get("pe"),
                    "Valeur": val,
                    "Moyenne": arrondi(moy),
                    "Écart-type": arrondi(vers_nombre(v.get("stdDev"))),
                    "Z-score": arrondi(vers_nombre(v.get("zScore"))),
                    "Borne basse": arrondi(bas),
                    "Borne haute": arrondi(haut),
                    "Anomalie": sens,
                    "Formulaire(s) à ouvrir": " | ".join(sorted(noms_formulaires.get(d, d) for d in ds_ids)),
                    "Lien de saisie": lien_saisie(ds_ids[0] if ds_ids else None, v.get("ou"), v.get("pe")),
                }
            )
        st.session_state["ab_result"] = {
            "lignes": lignes,
            "unite": noms_ou.get(orgunit_id, orgunit_id),
            "periode": f"{debut.isoformat()} → {fin.isoformat()}",
            "limite": int(max_res),
        }

    res = st.session_state.get("ab_result")
    if not res:
        return
    st.subheader(f"Valeurs aberrantes : {res['unite']} ({res['periode']})")
    if not res["lignes"]:
        st.success("Aucune valeur aberrante détectée avec ces paramètres. ✅")
        return
    if len(res["lignes"]) >= res["limite"]:
        st.warning(
            "La limite de résultats est atteinte : il peut y avoir d'autres valeurs aberrantes. "
            "Augmentez la limite ou réduisez l'unité, la période ou les formulaires."
        )
    df = pd.DataFrame(res["lignes"])
    total = len(df)

    with st.expander("🔎 Filtres", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            sel_an = filtre_multi("Anomalie", sorted(df["Anomalie"].unique()), "ab_f_an")
        with f2:
            sel_de = filtre_multi("Élément de données", sorted(df["Élément de données"].dropna().unique()), "ab_f_de")
        with f3:
            sel_ou = filtre_multi("Unité d'organisation", sorted(df["Unité d'organisation"].dropna().unique()), "ab_f_ou")
        with f4:
            sel_pe = filtre_multi("Période", sorted(df["Période"].dropna().unique()), "ab_f_pe")
    for col, sel in (
        ("Anomalie", sel_an), ("Élément de données", sel_de),
        ("Unité d'organisation", sel_ou), ("Période", sel_pe),
    ):
        if sel:
            df = df[df[col].isin(sel)]
    if df.empty:
        st.info("Aucune valeur ne correspond aux filtres choisis.")
        return

    st.metric("Valeurs à vérifier (filtres appliqués)", f"{len(df):,}".replace(",", " "),
              help=f"Total sans filtre : {total}")
    st.markdown("### Synthèse")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sens de l'anomalie**")
        k1, k2 = st.columns(2)
        k1.metric("Trop élevées", int((df["Anomalie"] == "Trop élevée").sum()))
        k2.metric("Trop basses", int((df["Anomalie"] == "Trop basse").sum()))
    with c2:
        graphique_compte(df, "Période", "Valeurs aberrantes par période", ordre="libelle", vertical=True)
    c3, c4 = st.columns(2)
    with c3:
        graphique_compte(df, "Élément de données", "Éléments de données les plus concernés (top 10)")
    with c4:
        graphique_compte(df, "Unité d'organisation", "Unités avec le plus de valeurs aberrantes (top 10)")

    st.download_button(
        "⬇️ Télécharger les valeurs affichées (CSV)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="dhis2_valeurs_aberrantes.csv",
        mime="text/csv",
    )
    st.markdown("### Détail")
    st.dataframe(df, hide_index=True, use_container_width=True, height=500, column_config=config_lien())


def sous_onglet_manquantes(orgunit_id, debut, fin):
    st.caption(
        "Repère les formulaires marqués comme complétés (soumis) dans lesquels certains éléments "
        "de données n'ont aucune valeur. Un élément est considéré comme renseigné dès qu'une valeur "
        "existe (zéro compris), quelle que soit sa combinaison de catégories."
    )
    choix = st.multiselect(
        "Formulaires à analyser",
        options=df_ds["id"].tolist(),
        format_func=lambda i: noms_ds.get(i, i),
        key="vm_ds",
    )

    if st.button(
        "🕳️ Rechercher les valeurs manquantes",
        type="primary",
        disabled=not (choix and orgunit_id and debut <= fin),
        key="vm_go",
    ):
        try:
            de_ds, noms_de, noms_formulaires = elements_et_formulaires(url, mode, user, password, token)
        except Exception as e:
            st.error(f"Impossible de lire les métadonnées (formulaires, éléments) : {e}")
            return
        ds_de = {}
        for de, dss in de_ds.items():
            for d in dss:
                ds_de.setdefault(d, []).append(de)

        lignes, details, erreurs, analysees = [], [], [], 0
        barre = st.progress(0.0, text="Analyse en cours…")
        for i, ds_id in enumerate(choix):
            nom = noms_ds.get(ds_id, ds_id)
            barre.progress(i / len(choix), text=f"Formulaire : {nom}")
            elements = sorted(ds_de.get(ds_id, []), key=lambda d: noms_de.get(d, d))
            if not elements:
                erreurs.append(f"{nom} : aucun élément de données dans ce formulaire.")
                continue
            try:
                subs = soumissions_formulaire(
                    url, mode, user, password, token, ds_id, orgunit_id, debut.isoformat(), fin.isoformat()
                )
                vals = valeurs_saisies(
                    url, mode, user, password, token, ds_id, orgunit_id, debut.isoformat(), fin.isoformat()
                )
            except Exception as e:
                erreurs.append(f"{nom} : {e}")
                continue
            analysees += len(subs)
            l, d = calculer_manquantes(subs, vals, elements, noms_de, nom, ds_id, noms_ou)
            lignes += l
            details += d
        barre.empty()
        st.session_state["vm_result"] = {
            "lignes": lignes,
            "details": details,
            "erreurs": erreurs,
            "analysees": analysees,
            "unite": noms_ou.get(orgunit_id, orgunit_id),
            "periode": f"{debut.isoformat()} → {fin.isoformat()}",
        }

    res = st.session_state.get("vm_result")
    if not res:
        return
    if res["erreurs"]:
        st.warning(
            "Analyse impossible pour certains formulaires (droits insuffisants ou erreur serveur) :\n\n"
            + "\n\n".join(res["erreurs"])
        )
    st.subheader(f"Valeurs manquantes : {res['unite']} ({res['periode']})")
    if res["analysees"] == 0:
        st.info("Aucun formulaire soumis (complété) trouvé pour ces paramètres.")
        return
    if not res["lignes"]:
        st.success(
            f"Les {res['analysees']} formulaire(s) soumis sont entièrement renseignés. ✅"
        )
        return

    df = pd.DataFrame(res["lignes"])
    det = pd.DataFrame(res["details"])
    with st.expander("🔎 Filtres", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            sel_fo = filtre_multi("Formulaire", sorted(df["Formulaire"].unique()), "vm_f_fo")
        with f2:
            sel_ou = filtre_multi("Unité d'organisation", sorted(df["Unité d'organisation"].unique()), "vm_f_ou")
        with f3:
            sel_pe = filtre_multi("Période", sorted(df["Période"].unique()), "vm_f_pe")
        with f4:
            sel_vide = filtre_multi("Formulaire vide", ["Oui", "Non"], "vm_f_vide")
    for col, sel in (("Formulaire", sel_fo), ("Unité d'organisation", sel_ou), ("Période", sel_pe),
                     ("Formulaire vide", sel_vide)):
        if sel:
            df = df[df[col].isin(sel)]
    if df.empty:
        st.info("Aucun résultat ne correspond aux filtres choisis.")
        return
    cles = ["Formulaire", "Unité d'organisation", "Période"]
    det = det.merge(df[cles].drop_duplicates(), on=cles)

    m1, m2, m3 = st.columns(3)
    m1.metric("Formulaires soumis analysés", f"{res['analysees']:,}".replace(",", " "))
    m2.metric(
        "Soumis avec éléments manquants",
        f"{len(df):,}".replace(",", " "),
        help="Compte les formulaires correspondant aux filtres.",
    )
    m3.metric("Soumis mais vides", int((df["Formulaire vide"] == "Oui").sum()))

    st.markdown("### Synthèse")
    c1, c2 = st.columns(2)
    with c1:
        graphique_compte(det, "Élément de données manquant", "Éléments de données les plus souvent manquants (top 10)")
    with c2:
        graphique_compte(det, "Unité d'organisation", "Unités avec le plus d'éléments manquants (top 10)")
    c3, c4 = st.columns(2)
    with c3:
        graphique_compte(
            df, "Période", "Formulaires incomplets par période", ordre="libelle", vertical=True
        )
    with c4:
        graphique_compte(df, "Formulaire", "Formulaires incomplets par formulaire")

    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇️ Télécharger la liste des formulaires (CSV)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="dhis2_valeurs_manquantes.csv",
        mime="text/csv",
    )
    d2.download_button(
        "⬇️ Télécharger le détail par élément (CSV)",
        data=det.to_csv(index=False).encode("utf-8-sig"),
        file_name="dhis2_valeurs_manquantes_detail.csv",
        mime="text/csv",
    )
    st.markdown("### Détail : formulaires soumis avec éléments manquants")
    st.dataframe(
        df.sort_values(["Taux de renseignement (%)", "Unité d'organisation", "Période"]),
        hide_index=True,
        use_container_width=True,
        height=500,
        column_config=config_lien(),
    )


def onglet_qualite():
    st.caption(
        "Contrôle la qualité des données sur l'unité d'organisation et la période choisies "
        "(unités descendantes comprises)."
    )
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        orgunit_id = choisir_unite("qd")
    with col2:
        debut = st.date_input("Début de la période", value=datetime.date(annee_fin, 1, 1), key="qd_debut")
    with col3:
        fin = st.date_input("Fin de la période", value=datetime.date.today(), key="qd_fin")
    if debut > fin:
        st.warning("La date de début doit précéder la date de fin.")
    st.caption("⚠️ Sur une grande unité (pays, région) et une longue période, l'analyse peut être longue.")

    sous_val, sous_ab, sous_vm = st.tabs(
        ["📋 Règles de validation", "📈 Valeurs aberrantes", "🕳️ Valeurs manquantes"]
    )
    with sous_val:
        sous_onglet_validation(orgunit_id, debut, fin)
    with sous_ab:
        sous_onglet_aberrantes(orgunit_id, debut, fin)
    with sous_vm:
        sous_onglet_manquantes(orgunit_id, debut, fin)


# ------------------------------------------------------------------
# Onglet 4 : tableaux de bord
# ------------------------------------------------------------------
def onglet_dashboard():
    st.caption(
        "Charge les tableaux croisés d'un tableau de bord DHIS2 : la requête enregistrée de chaque "
        "tableau est relancée avec les données actuelles."
    )
    c1, c2 = st.columns([2, 2])
    with c1:
        dashboard_id = st.text_input("ID du tableau de bord", value="GU65zFQh5WH", key="db_id").strip()
    with c2:
        inclure = st.checkbox(
            "Inclure aussi les graphiques (leurs données sous forme de tableau)", value=False, key="db_graphiques"
        )
    if st.button("📥 Charger les tableaux", type="primary", disabled=not dashboard_id, key="db_go"):
        try:
            with st.spinner("Chargement du tableau de bord…"):
                st.session_state["db_result"] = charger_tableaux_dashboard(dashboard_id, inclure)
        except Exception as e:
            st.session_state.pop("db_result", None)
            st.error(
                f"Impossible de charger le tableau de bord (ID incorrect, droits insuffisants ou erreur serveur) : {e}"
            )

    res = st.session_state.get("db_result")
    if not res:
        return
    st.subheader(f"{res['nom']}")
    st.caption(f"{len(res['tables'])} tableau(x) chargé(s).")
    if res["erreurs"]:
        st.warning("Tableaux non chargés :\n\n" + "\n\n".join(res["erreurs"]))
    if res["autres"]:
        with st.expander(f"Autres éléments du tableau de bord, non chargés ({len(res['autres'])})"):
            st.write("\n".join(f"- {a}" for a in res["autres"]))
    for t in res["tables"]:
        with st.expander(f"📋 {t['nom']}", expanded=len(res["tables"]) <= 3):
            if t["df"].empty:
                st.info("Aucune donnée renvoyée pour ce tableau.")
            else:
                st.dataframe(t["df"], hide_index=True, use_container_width=True)
            st.caption(
                f"Type : {t['type']} · {t['nb_lignes_analytics']} ligne(s) renvoyée(s) par analytics · "
                f"lignes : {', '.join(t['lignes']) or '–'} · colonnes : {', '.join(t['colonnes']) or '–'}"
            )
            with st.expander("Requête analytics utilisée"):
                st.json(t["requete"])


# ------------------------------------------------------------------
# Onglet 5 : validation (tableaux du tableau de bord GU65zFQh5WH)
# ------------------------------------------------------------------
DASHBOARD_VALIDATION = "GU65zFQh5WH"
MOIS = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
        "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]


def onglet_validation():
    st.caption(
        "Charge les tableaux du tableau de bord de validation : la requête enregistrée de "
        "chaque tableau est relancée avec les données actuelles, sur la période choisie."
    )
    try:
        df_un = unites_avec_niveau(url, mode, user, password, token)
        noms_niv = noms_niveaux(url, mode, user, password, token)
    except Exception as e:
        st.error(f"Impossible de lire les unités d'organisation : {e}")
        return

    # L'onglet Validation travaille exclusivement au niveau district (niveau 3).
    districts = df_un[pd.to_numeric(df_un["level"], errors="coerce") == 3].sort_values("displayName")
    if districts.empty:
        st.warning("Aucun district (niveau 3) n'a été trouvé dans la hiérarchie DHIS2.")
        return
    noms_districts = dict(zip(districts["id"], districts["displayName"]))

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown(f"**Tableau de bord** : `{DASHBOARD_VALIDATION}`")
    with c2:
        unite_val = st.selectbox(
            "District",
            options=districts["id"].tolist(),
            format_func=lambda i: noms_districts[i],
            key="val_unite",
        )

    c3, c4, c5 = st.columns([1, 1, 1])
    with c3:
        annees_val = sorted(st.multiselect(
            "Année(s)", options=options_annees, default=[annee_fin], key="val_annees"
        ))
    with c4:
        mois_val = sorted(st.multiselect(
            "Mois", options=list(range(1, 13)), default=list(range(1, 13)),
            format_func=lambda m: MOIS[m - 1], key="val_mois"
        ))
    with c5:
        rafraichir = st.button("🔄 Rafraîchir", key="val_refresh", use_container_width=True)

    if not annees_val:
        st.warning("Sélectionnez au moins une année.")
        return
    # Périodes DHIS2 explicites : mensuelles (YYYYMM) si des mois sont choisis, annuelles sinon
    periodes = (
        [f"{a}{m:02d}" for a in annees_val for m in mois_val]
        if mois_val
        else [str(a) for a in annees_val]
    )
    signature = (tuple(annees_val), tuple(mois_val), unite_val)
    if rafraichir or st.session_state.get("val_signature") != signature:
        try:
            with st.spinner("Chargement du tableau de bord de validation…"):
                st.session_state["val_result"] = charger_tableaux_dashboard(
                    DASHBOARD_VALIDATION, False, periodes=periodes,
                    unites_ou=[unite_val], ou_mode="DESCENDANTS",
                )
            st.session_state["val_signature"] = signature
        except Exception as e:
            st.session_state.pop("val_result", None)
            st.error(
                "Impossible de charger le tableau de bord de validation "
                f"(ID incorrect, droits insuffisants ou erreur serveur) : {e}"
            )

    res = st.session_state.get("val_result")
    if not res:
        return
    st.subheader(f"{res['nom']}")
    nom_unite = noms_districts.get(unite_val, unite_val)
    libelle_mois = ", ".join(MOIS[m - 1] for m in mois_val) if mois_val else "année(s) complète(s)"
    st.caption(
        f"{len(res['tables'])} tableau(x) chargé(s) · district : {nom_unite} (structures enfants incluses) · période : "
        f"{', '.join(map(str, annees_val))} · {libelle_mois}"
    )
    if res["erreurs"]:
        st.warning("Tableaux non chargés :\n\n" + "\n\n".join(res["erreurs"]))
    if res["autres"]:
        with st.expander(f"Autres éléments du tableau de bord, non chargés ({len(res['autres'])})"):
            st.write("\n".join(f"- {a}" for a in res["autres"]))
    for t in res["tables"]:
        with st.expander(f"📋 {t['nom']}", expanded=len(res["tables"]) <= 3):
            if t["df"].empty:
                st.info("Aucune donnée renvoyée pour ce tableau.")
            else:
                st.dataframe(t["df"], hide_index=True, use_container_width=True)
            st.caption(
                f"Type : {t['type']} · {t['nb_lignes_analytics']} ligne(s) renvoyée(s) par analytics · "
                f"lignes : {', '.join(t['lignes']) or '–'} · colonnes : {', '.join(t['colonnes']) or '–'}"
            )
            with st.expander("Requête analytics utilisée"):
                st.json(t["requete"])


tab_ext, tab_comp, tab_qd, tab_db, tab_val = st.tabs(
    ["📥 Extraction", "✅ Complétude", "🔍 Qualité des données", "📊 Tableaux de bord", "🧪 Validation"]
)
with tab_ext:
    onglet_extraction()
with tab_comp:
    onglet_completude()
with tab_qd:
    onglet_qualite()
with tab_db:
    onglet_dashboard()
with tab_val:
    onglet_validation()
