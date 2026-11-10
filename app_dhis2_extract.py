"""
Application Streamlit : extraction de données DHIS2
(un formulaire x une unité d'organisation x N dernières années)
Trois onglets indépendants : « Extraction », « Complétude » et « Validation ».
L'onglet « Validation » regroupe : règles de validation, valeurs aberrantes (z-score),
variations dans le temps et données manquantes (unités silencieuses, retards, zéros), avec lien direct vers la saisie (modèle réglable : DHIS2_LIEN_SAISIE).
Les données extraites sont agrégées par année : pour chaque élément de données, on cumule
toutes les combinaisons de catégories (categoryOptionCombo).

Installation : pip install streamlit requests pandas python-dotenv
Lancement    : streamlit run app_dhis2_extract.py

Identifiants : fichier .env placé à côté de ce script (voir .env.example)
    DHIS2_URL=https://mon-instance.org/api
    DHIS2_USER=...
    DHIS2_PASSWORD=...
    DHIS2_TOKEN=d2pat_...
"""
import datetime
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


def completude(s, url, dataset, orgunit, annees) -> pd.DataFrame:
    """Taux de complétude, rapports attendus et saisis par année (analytics DHIS2).
    Les valeurs couvrent l'unité choisie et ses descendants."""
    metriques = ["REPORTING_RATE", "EXPECTED_REPORTS", "ACTUAL_REPORTS"]
    params = {
        "dimension": [
            "dx:" + ";".join(f"{dataset}.{m}" for m in metriques),
            "pe:" + ";".join(str(a) for a in annees),
            f"ou:{orgunit}",
        ],
        "skipMeta": "true",
    }
    r = s.get(f"{url}/analytics.json", params=params, timeout=300)
    r.raise_for_status()
    j = r.json()
    df = pd.DataFrame(j.get("rows", []), columns=[h["name"] for h in j["headers"]])
    if df.empty:
        piv = pd.DataFrame(index=annees, columns=metriques, dtype=float)
    else:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["metrique"] = df["dx"].str.split(".").str[-1]
        piv = df.pivot_table(index="pe", columns="metrique", values="value", aggfunc="sum")
        piv.index = piv.index.astype(int)
        piv = piv.reindex(index=annees, columns=metriques)
    out = piv.reset_index().rename(
        columns={
            "index": "annee",
            "pe": "annee",
            "EXPECTED_REPORTS": "rapports_attendus",
            "ACTUAL_REPORTS": "rapports_saisis",
            "REPORTING_RATE": "taux_completude",
        }
    )
    out.columns.name = None
    return out[["annee", "rapports_attendus", "rapports_saisis", "taux_completude"]]


@st.cache_data(ttl=600, show_spinner=False)
def completude_cache(url, mode, user, password, token, dataset, orgunit, annees) -> pd.DataFrame:
    s = make_session(mode, user, password, token)
    return completude(s, url, dataset, orgunit, list(annees))


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


def formulaires_a_ouvrir(regle: dict, de_ds: dict, noms_ds: dict) -> tuple:
    """Formulaire(s) contenant les éléments de la règle : (noms, éléments concernés, UID des formulaires)."""
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


def vers_nombre(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---- Validation : liens de saisie, valeurs aberrantes, variations dans le temps ----
# Modèle du lien vers l'application de saisie (modifiable via DHIS2_LIEN_SAISIE dans le .env)
MODELE_LIEN = secret(
    "DHIS2_LIEN_SAISIE",
    "{base}/dhis-web-aggregate-data-entry/index.html#/?dataSetId={ds}&orgUnitId={ou}&periodId={pe}",
)


def lien_saisie(ds_id, ou_id, pe_id) -> str:
    """Lien vers la saisie de données (formulaire, unité et période préremplis)."""
    if not (ds_id and ou_id and pe_id):
        return ""
    base = re.sub(r"/api(/\d+)?/?$", "", url)
    return MODELE_LIEN.format(base=base, ds=ds_id, ou=ou_id, pe=pe_id)


def config_liens() -> dict:
    try:
        try:
            col = st.column_config.LinkColumn("Lien de saisie", display_text="Ouvrir ↗")
        except TypeError:  # anciennes versions de Streamlit
            col = st.column_config.LinkColumn("Lien de saisie")
        return {"Lien de saisie": col}
    except Exception:
        return {}


def premiere_colonne(df: pd.DataFrame, *noms) -> pd.Series:
    for n in noms:
        if n in df.columns:
            return df[n]
    return pd.Series([None] * len(df), index=df.index, dtype="object")


@st.cache_data(ttl=600, show_spinner=False)
def valeurs_aberrantes(url, mode, user, password, token, formulaires, orgunit, debut, fin, algo, seuil, maxi):
    """Détection statistique DHIS2 (/outlierDetection) sur l'unité et ses descendantes."""
    s = make_session(mode, user, password, token)
    params = [("ds", d) for d in formulaires] + [
        ("ou", orgunit),
        ("startDate", debut),
        ("endDate", fin),
        ("algorithm", algo),
        ("maxResults", maxi),
    ]
    if algo != "MIN_MAX":
        params.append(("threshold", seuil))
    r = s.get(f"{url}/outlierDetection.json", params=params, timeout=900)
    r.raise_for_status()
    j = r.json()
    if isinstance(j, dict) and isinstance(j.get("outlierValues"), list):
        return pd.DataFrame(j["outlierValues"])
    if isinstance(j, dict) and "rows" in j:
        cols = [h.get("name") or h.get("column") for h in j.get("headers", [])]
        return pd.DataFrame(j["rows"], columns=cols)
    if isinstance(j, list):
        return pd.DataFrame(j)
    return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def valeurs_formulaire(url, mode, user, password, token, dataset, orgunit, debut, fin) -> pd.DataFrame:
    """Valeurs brutes d'un formulaire (unité + enfants) sur une période, avec leur période DHIS2."""
    s = make_session(mode, user, password, token)
    r = s.get(
        f"{url}/dataValueSets.json",
        params={
            "dataSet": dataset,
            "orgUnit": orgunit,
            "children": "true",
            "startDate": debut,
            "endDate": fin,
        },
        timeout=900,
    )
    r.raise_for_status()
    return pd.DataFrame(r.json().get("dataValues", []))


def anomalies_temporelles(data, controles, hausse, baisse, base_min, nb_constant) -> pd.DataFrame:
    """Valeurs négatives, variations brutales et séries constantes.
    Les valeurs sont cumulées sur les combinaisons de catégories, par unité, élément et période.
    Les périodes comparées sont les périodes consécutives *saisies* (les trous ne sont pas comblés)."""
    colonnes = ["Anomalie", "orgUnit", "dataElement", "Période", "Valeur",
                "Valeur précédente", "Variation (%)", "Détail"]
    if data.empty:
        return pd.DataFrame(columns=colonnes)
    d = data.assign(valeur=pd.to_numeric(data["value"], errors="coerce")).dropna(subset=["valeur"])
    morceaux = []

    if "Valeurs négatives" in controles:
        neg = d[d["valeur"] < 0]
        morceaux.append(
            pd.DataFrame(
                {
                    "Anomalie": "Valeur négative",
                    "orgUnit": neg["orgUnit"],
                    "dataElement": neg["dataElement"],
                    "Période": neg["period"],
                    "Valeur": neg["valeur"],
                    "Valeur précédente": None,
                    "Variation (%)": None,
                    "Détail": "Valeur négative saisie",
                }
            )
        )

    agg = (
        d.groupby(["orgUnit", "dataElement", "period"], as_index=False)["valeur"]
        .sum()
        .sort_values(["orgUnit", "dataElement", "period"])
    )
    cle = ["orgUnit", "dataElement"]
    agg["precedente"] = agg.groupby(cle)["valeur"].shift(1)
    agg["pe_prec"] = agg.groupby(cle)["period"].shift(1)

    if "Variations brutales" in controles:
        base = agg["precedente"] >= base_min
        hausses = agg[base & (agg["valeur"] >= agg["precedente"] * (1 + hausse / 100))]
        baisses = agg[base & (agg["valeur"] >= 0) & (agg["valeur"] <= agg["precedente"] * (1 - baisse / 100))]
        for lot, nom in ((hausses, "Hausse brutale"), (baisses, "Baisse brutale")):
            morceaux.append(
                pd.DataFrame(
                    {
                        "Anomalie": nom,
                        "orgUnit": lot["orgUnit"],
                        "dataElement": lot["dataElement"],
                        "Période": lot["period"],
                        "Valeur": lot["valeur"],
                        "Valeur précédente": lot["precedente"],
                        "Variation (%)": (lot["valeur"] - lot["precedente"]) / lot["precedente"] * 100,
                        "Détail": "Par rapport à la période " + lot["pe_prec"].astype(str),
                    }
                )
            )

    if "Séries constantes" in controles:
        agg["rupture"] = ~((agg["valeur"] == agg["precedente"]) & (agg["valeur"] > 0))
        agg["serie"] = agg.groupby(cle)["rupture"].cumsum()
        agg["taille"] = agg.groupby(cle + ["serie"])["valeur"].transform("size")
        longues = agg[agg["taille"] >= nb_constant]
        if not longues.empty:
            res = longues.groupby(cle + ["serie"], as_index=False).agg(
                debut=("period", "first"),
                fin=("period", "last"),
                n=("period", "size"),
                valeur=("valeur", "first"),
            )
            morceaux.append(
                pd.DataFrame(
                    {
                        "Anomalie": "Série constante",
                        "orgUnit": res["orgUnit"],
                        "dataElement": res["dataElement"],
                        "Période": res["fin"],
                        "Valeur": res["valeur"],
                        "Valeur précédente": None,
                        "Variation (%)": None,
                        "Détail": res["n"].astype(str)
                        + " périodes consécutives identiques ("
                        + res["debut"].astype(str)
                        + " → "
                        + res["fin"].astype(str)
                        + ")",
                    }
                )
            )

    morceaux = [m for m in morceaux if not m.empty]
    if not morceaux:
        return pd.DataFrame(columns=colonnes)
    res = pd.concat(morceaux, ignore_index=True)[colonnes]
    for c in ("Valeur", "Valeur précédente", "Variation (%)"):
        res[c] = pd.to_numeric(res[c], errors="coerce")
    return res



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
st.title("📊 DHIS2 : extraction, complétude et validation")

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
    annees = sorted(
        st.multiselect(
            "Années", options=options_annees, default=annees_defaut, key="comp_annees"
        )
    )

    if st.button(
        "📈 Calculer la complétude",
        type="primary",
        disabled=not (choix and orgunit_id and annees),
        key="comp_go",
    ):
        morceaux, erreurs = [], []
        barre = st.progress(0.0, text="Calcul en cours…")
        for i, ds_id in enumerate(choix):
            try:
                c = completude_cache(
                    url, mode, user, password, token, ds_id, orgunit_id, tuple(annees)
                )
                morceaux.append(c.assign(formulaire=noms_ds.get(ds_id, ds_id)))
            except Exception as e:
                erreurs.append(f"{noms_ds.get(ds_id, ds_id)} : {e}")
            barre.progress((i + 1) / len(choix), text=noms_ds.get(ds_id, ds_id))
        barre.empty()
        st.session_state["comp_result"] = {
            "data": pd.concat(morceaux, ignore_index=True) if morceaux else None,
            "erreurs": erreurs,
            "unite": noms_ou.get(orgunit_id, orgunit_id),
        }

    res = st.session_state.get("comp_result")
    if not res:
        return
    if res["erreurs"]:
        st.warning(
            "Complétude indisponible pour certains formulaires (analytics non généré "
            "ou droits insuffisants) :\n\n" + "\n\n".join(res["erreurs"])
        )
    comp = res["data"]
    if comp is None:
        return

    st.subheader(f"Complétude : {res['unite']}")
    affichage = comp[
        ["formulaire", "annee", "rapports_attendus", "rapports_saisis", "taux_completude"]
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
        affichage.style.format(
            {"Rapports attendus": "{:.0f}", "Rapports saisis": "{:.0f}", "Complétude (%)": "{:.1f}"},
            na_rep="–",
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


# ------------------------------------------------------------------
# Onglet 3 : validation (groupes de règles de validation)
# ------------------------------------------------------------------
def section_regles():
    st.caption(
        "Exécute les règles de validation, groupe par groupe, sur l'unité d'organisation "
        "choisie (et ses descendantes) et liste les erreurs à corriger."
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

    col1, col2 = st.columns(2)
    with col1:
        choix = st.multiselect(
            "Groupes de règles de validation",
            options=list(noms_g),
            default=list(noms_g),
            format_func=lambda i: f"{noms_g[i]} ({nb_regles[i]} règle(s))",
            key="qd_groupes",
        )
    with col2:
        orgunit_id = choisir_unite("qd")
    col3, col4 = st.columns(2)
    with col3:
        debut = st.date_input("Début de la période", value=datetime.date(annee_fin, 1, 1), key="qd_debut")
    with col4:
        fin = st.date_input("Fin de la période", value=datetime.date.today(), key="qd_fin")
    if debut > fin:
        st.warning("La date de début doit précéder la date de fin.")
    st.caption(
        "⚠️ Sur une grande unité (pays, région) et une longue période, l'analyse peut être longue."
    )

    if st.button(
        "🔎 Lancer la validation",
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
                        "Éléments de données à vérifier": ", ".join(noms_de.get(d, d) for d in de_ids),
                        "Instruction de correction": regle.get("instruction", ""),
                        "Lien de saisie": lien_saisie(
                            ds_ids[0] if ds_ids else "",
                            v.get("organisationUnitId"),
                            v.get("periodId"),
                        ),
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
    if not res:
        return
    if res["erreurs"]:
        st.warning(
            "Analyse impossible pour certains groupes (droits insuffisants ou erreur serveur) :\n\n"
            + "\n\n".join(res["erreurs"])
        )
    if not res["resume"]:
        return

    st.subheader(f"Résultats : {res['unite']} ({res['periode']})")
    df_resume = pd.DataFrame(res["resume"])
    total = int(df_resume["Erreurs détectées"].sum())
    st.metric("Erreurs à corriger (tous groupes)", f"{total:,}".replace(",", " "))
    st.dataframe(df_resume, hide_index=True, use_container_width=True)

    if not res["lignes"]:
        st.success("Aucune erreur détectée par les règles de validation sélectionnées. ✅")
        return

    df = pd.DataFrame(res["lignes"])
    df["_o"] = df["Importance"].map(ORDRE_IMPORTANCE).fillna(3)
    df = df.sort_values(["Groupe", "_o", "Règle", "Unité d'organisation", "Période"]).drop(columns="_o")

    st.download_button(
        "⬇️ Télécharger toutes les erreurs (CSV)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="dhis2_erreurs_validation.csv",
        mime="text/csv",
    )

    # Un tableau par groupe de règles de validation
    for ligne in res["resume"]:
        groupe = ligne["Groupe"]
        sous = df[df["Groupe"] == groupe].drop(columns="Groupe")
        if sous.empty:
            st.markdown(f"#### {groupe}")
            st.caption("✅ Aucune erreur dans ce groupe.")
            continue
        with st.expander(f"❌ {groupe} : {len(sous)} erreur(s)", expanded=True):
            st.dataframe(
                sous,
                hide_index=True,
                use_container_width=True,
                height=min(500, 40 + 35 * len(sous)),
                column_config=config_liens(),
            )


# ------------------------------------------------------------------
# Validation : valeurs aberrantes (algorithmes statistiques DHIS2)
# ------------------------------------------------------------------
def section_aberrantes():
    st.caption(
        "Détection statistique des valeurs aberrantes par DHIS2 (z-score, z-score modifié ou "
        "min-max) : repère des valeurs absurdes qu'aucune règle de validation ne couvre."
    )
    col1, col2 = st.columns(2)
    with col1:
        choix = st.multiselect(
            "Formulaires",
            options=df_ds["id"].tolist(),
            format_func=lambda i: noms_ds.get(i, i),
            key="ab_ds",
        )
    with col2:
        orgunit_id = choisir_unite("ab")
    col3, col4 = st.columns(2)
    aujourdhui = datetime.date.today()
    with col3:
        debut = st.date_input(
            "Début de la période contrôlée",
            value=aujourdhui - datetime.timedelta(days=365),
            key="ab_debut",
        )
    with col4:
        fin = st.date_input("Fin de la période contrôlée", value=aujourdhui, key="ab_fin")
    col5, col6, col7 = st.columns(3)
    with col5:
        algo = st.selectbox(
            "Algorithme",
            ["Z_SCORE", "MODIFIED_Z_SCORE", "MIN_MAX"],
            format_func={
                "Z_SCORE": "Z-score",
                "MODIFIED_Z_SCORE": "Z-score modifié (robuste)",
                "MIN_MAX": "Min-max (bornes définies dans DHIS2)",
            }.get,
            key="ab_algo",
        )
    with col6:
        seuil = st.number_input(
            "Seuil",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            disabled=algo == "MIN_MAX",
            help="Plus le seuil est bas, plus on détecte de valeurs (et de faux positifs).",
            key="ab_seuil",
        )
    with col7:
        maxi = st.number_input(
            "Nombre maximum de résultats", min_value=10, max_value=10000, value=500, step=100, key="ab_max"
        )
    st.caption(
        "La moyenne et l'écart-type sont calculés sur tout l'historique disponible ; seules les "
        "valeurs de la période choisie sont contrôlées."
    )

    if st.button(
        "🔎 Chercher les valeurs aberrantes",
        type="primary",
        disabled=not (choix and orgunit_id and debut <= fin),
        key="ab_go",
    ):
        try:
            with st.spinner("Analyse en cours…"):
                brut = valeurs_aberrantes(
                    url, mode, user, password, token, tuple(choix), orgunit_id,
                    debut.isoformat(), fin.isoformat(), algo, float(seuil), int(maxi),
                )
                de_ds, noms_de, noms_formulaires = elements_et_formulaires(
                    url, mode, user, password, token
                )
        except Exception as e:
            st.error(
                f"Détection impossible : {e}\n\nVérifiez que l'endpoint /outlierDetection est "
                "disponible sur votre instance et que vous avez les droits de lecture."
            )
            return

        if brut.empty:
            st.session_state["ab_result"] = {"df": None}
        else:
            de = premiere_colonne(brut, "de", "dataElement")
            ou = premiere_colonne(brut, "ou", "orgUnit")
            pe = premiere_colonne(brut, "pe", "period")
            formulaires, liens = [], []
            for d_id, o_id, p_id in zip(de, ou, pe):
                dss = de_ds.get(d_id, [])
                pref = [x for x in dss if x in choix] or dss
                formulaires.append(" | ".join(noms_formulaires.get(x, x) for x in pref))
                liens.append(lien_saisie(pref[0] if pref else "", o_id, p_id))
            nom_de = premiere_colonne(brut, "deName", "dataElementName")
            nom_ou = premiere_colonne(brut, "ouName", "orgUnitName")
            affich = pd.DataFrame(
                {
                    "Formulaire(s)": formulaires,
                    "Élément de données": nom_de.where(nom_de.notna(), de.map(noms_de)),
                    "Combinaison de catégories": premiere_colonne(brut, "cocName", "categoryOptionComboName"),
                    "Unité d'organisation": nom_ou.where(nom_ou.notna(), ou.map(noms_ou)),
                    "Période": pe,
                    "Valeur": pd.to_numeric(premiere_colonne(brut, "value"), errors="coerce"),
                    "Moyenne": pd.to_numeric(premiere_colonne(brut, "mean"), errors="coerce"),
                    "Médiane": pd.to_numeric(premiere_colonne(brut, "median"), errors="coerce"),
                    "Écart-type": pd.to_numeric(premiere_colonne(brut, "stdDev"), errors="coerce"),
                    "Borne basse": pd.to_numeric(premiere_colonne(brut, "lowerBound"), errors="coerce"),
                    "Borne haute": pd.to_numeric(premiere_colonne(brut, "upperBound"), errors="coerce"),
                    "Score": pd.to_numeric(
                        premiere_colonne(brut, "zScore", "modifiedZScore"), errors="coerce"
                    ),
                    "Lien de saisie": liens,
                }
            ).dropna(axis=1, how="all")
            st.session_state["ab_result"] = {
                "df": affich,
                "unite": noms_ou.get(orgunit_id, orgunit_id),
                "periode": f"{debut.isoformat()} → {fin.isoformat()}",
                "limite": len(affich) >= int(maxi),
            }

    res = st.session_state.get("ab_result")
    if not res:
        return
    if res["df"] is None:
        st.success("Aucune valeur aberrante détectée avec ces paramètres. ✅")
        return
    df = res["df"]
    st.subheader(f"Valeurs aberrantes : {res['unite']} ({res['periode']})")
    st.metric("Valeurs à vérifier", f"{len(df):,}".replace(",", " "))
    if res["limite"]:
        st.warning("Le nombre maximum de résultats est atteint : augmentez-le ou relevez le seuil.")
    st.download_button(
        "⬇️ Télécharger les valeurs aberrantes (CSV)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="dhis2_valeurs_aberrantes.csv",
        mime="text/csv",
    )
    for form, sous in df.groupby("Formulaire(s)", sort=True):
        sous = sous.drop(columns="Formulaire(s)")
        with st.expander(f"❌ {form or 'Formulaire inconnu'} : {len(sous)} valeur(s)", expanded=True):
            st.dataframe(
                sous,
                hide_index=True,
                use_container_width=True,
                height=min(500, 40 + 35 * len(sous)),
                column_config=config_liens(),
            )


# ------------------------------------------------------------------
# Validation : cohérence dans le temps (variations, séries constantes, négatifs)
# ------------------------------------------------------------------
def section_variations():
    st.caption(
        "Compare chaque valeur à la période précédente saisie (par unité et par élément de "
        "données, toutes combinaisons de catégories cumulées) pour repérer les variations brutales, "
        "les séries de valeurs identiques (copier-coller) et les valeurs négatives."
    )
    col1, col2 = st.columns(2)
    with col1:
        dataset_id = st.selectbox(
            "Formulaire",
            options=df_ds["id"].tolist(),
            format_func=lambda i: noms_ds.get(i, i),
            key="vt_ds",
        )
    with col2:
        orgunit_id = choisir_unite("vt")
    col3, col4 = st.columns(2)
    aujourdhui = datetime.date.today()
    with col3:
        debut = st.date_input("Début", value=datetime.date(annee_fin - 1, 1, 1), key="vt_debut")
    with col4:
        fin = st.date_input("Fin", value=aujourdhui, key="vt_fin")
    controles = st.multiselect(
        "Contrôles à appliquer",
        ["Variations brutales", "Séries constantes", "Valeurs négatives"],
        default=["Variations brutales", "Séries constantes", "Valeurs négatives"],
        key="vt_controles",
    )
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        hausse = st.number_input("Hausse signalée à partir de (%)", 10, 10000, 200, step=10, key="vt_hausse")
    with col6:
        baisse = st.number_input("Baisse signalée à partir de (%)", 10, 100, 80, step=5, key="vt_baisse")
    with col7:
        base_min = st.number_input(
            "Valeur précédente minimale", 1, 100000, 10, step=1,
            help="Évite de signaler des variations sur de très petits effectifs.", key="vt_base",
        )
    with col8:
        nb_constant = st.number_input(
            "Périodes identiques consécutives", 2, 24, 4, step=1, key="vt_const"
        )
    st.caption(
        "⚠️ Une grande unité sur une longue période peut produire un volume important de données."
    )

    if st.button(
        "🔎 Analyser la cohérence dans le temps",
        type="primary",
        disabled=not (dataset_id and orgunit_id and controles and debut <= fin),
        key="vt_go",
    ):
        try:
            with st.spinner("Extraction et analyse en cours…"):
                data = valeurs_formulaire(
                    url, mode, user, password, token, dataset_id, orgunit_id,
                    debut.isoformat(), fin.isoformat(),
                )
                _, noms_de, _ = elements_et_formulaires(url, mode, user, password, token)
        except Exception as e:
            st.error(f"Extraction impossible : {e}")
            return
        if data.empty:
            st.session_state["vt_result"] = {"df": None, "vide": True}
        else:
            anomalies = anomalies_temporelles(
                data, controles, float(hausse), float(baisse), float(base_min), int(nb_constant)
            )
            if anomalies.empty:
                st.session_state["vt_result"] = {"df": None, "vide": False}
            else:
                anomalies["Unité d'organisation"] = (
                    anomalies["orgUnit"].map(noms_ou).fillna(anomalies["orgUnit"])
                )
                anomalies["Élément de données"] = (
                    anomalies["dataElement"].map(noms_de).fillna(anomalies["dataElement"])
                )
                anomalies["Lien de saisie"] = [
                    lien_saisie(dataset_id, o, p) for o, p in zip(anomalies["orgUnit"], anomalies["Période"])
                ]
                st.session_state["vt_result"] = {
                    "df": anomalies.drop(columns=["orgUnit", "dataElement"]),
                    "vide": False,
                    "formulaire": noms_ds.get(dataset_id, dataset_id),
                    "unite": noms_ou.get(orgunit_id, orgunit_id),
                    "periode": f"{debut.isoformat()} → {fin.isoformat()}",
                    "nb_valeurs": len(data),
                }

    res = st.session_state.get("vt_result")
    if not res:
        return
    if res["df"] is None:
        if res["vide"]:
            st.warning("Aucune donnée trouvée pour ce formulaire, cette unité et cette période.")
        else:
            st.success("Aucune anomalie détectée avec ces paramètres. ✅")
        return
    df = res["df"]
    st.subheader(f"{res['formulaire']} : {res['unite']} ({res['periode']})")
    c1, c2 = st.columns(2)
    c1.metric("Anomalies détectées", f"{len(df):,}".replace(",", " "))
    c2.metric("Valeurs analysées", f"{res['nb_valeurs']:,}".replace(",", " "))
    ordre = ["Unité d'organisation", "Élément de données", "Période", "Valeur",
             "Valeur précédente", "Variation (%)", "Détail", "Lien de saisie"]
    st.download_button(
        "⬇️ Télécharger les anomalies (CSV)",
        data=df[["Anomalie"] + ordre].to_csv(index=False).encode("utf-8-sig"),
        file_name="dhis2_anomalies_temporelles.csv",
        mime="text/csv",
    )
    for type_anomalie in ["Hausse brutale", "Baisse brutale", "Série constante", "Valeur négative"]:
        sous = df[df["Anomalie"] == type_anomalie]
        if sous.empty:
            continue
        sous = sous[ordre]
        if type_anomalie in ("Hausse brutale", "Baisse brutale"):
            sous = sous.reindex(sous["Variation (%)"].abs().sort_values(ascending=False).index)
        else:
            sous = sous.sort_values(["Unité d'organisation", "Élément de données", "Période"])
        sous = sous.dropna(axis=1, how="all")
        with st.expander(f"❌ {type_anomalie} : {len(sous)} cas", expanded=True):
            st.dataframe(
                sous,
                hide_index=True,
                use_container_width=True,
                height=min(500, 40 + 35 * len(sous)),
                column_config={
                    **config_liens(),
                    "Variation (%)": st.column_config.NumberColumn(format="%.0f %%"),
                },
            )


# ------------------------------------------------------------------
# Validation : données manquantes et silencieuses (unités qui ne déclarent pas)
# ------------------------------------------------------------------
PAS_MOIS = {"Monthly": 1, "Quarterly": 3, "SixMonthly": 6, "Yearly": 12}


def periodes_attendues(type_periode, debut, fin, aujourdhui):
    """Périodes DHIS2 entièrement écoulées entre debut et fin, sous forme (id, début, fin).
    Renvoie None si le type de période n'est pas géré."""
    limite = min(fin, aujourdhui)
    res = []
    if type_periode in PAS_MOIS:
        n = PAS_MOIS[type_periode]
        y, m = debut.year, debut.month
        if debut.day > 1:
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        while (m - 1) % n != 0:
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        while True:
            total = y * 12 + (m - 1) + n
            y2, m2 = total // 12, total % 12 + 1
            debut_p = datetime.date(y, m, 1)
            fin_p = datetime.date(y2, m2, 1) - datetime.timedelta(days=1)
            if fin_p > limite:
                break
            if type_periode == "Monthly":
                pid = f"{y}{m:02d}"
            elif type_periode == "Quarterly":
                pid = f"{y}Q{(m - 1) // 3 + 1}"
            elif type_periode == "SixMonthly":
                pid = f"{y}S{(m - 1) // 6 + 1}"
            else:
                pid = f"{y}"
            res.append((pid, debut_p, fin_p))
            y, m = y2, m2
        return res
    if type_periode == "Weekly":
        d = debut + datetime.timedelta(days=(7 - debut.weekday()) % 7)
        while d + datetime.timedelta(days=6) <= limite:
            iso = d.isocalendar()
            res.append((f"{iso[0]}W{iso[1]}", d, d + datetime.timedelta(days=6)))
            d += datetime.timedelta(days=7)
        return res
    return None


def analyser_saisie(unites: pd.DataFrame, data: pd.DataFrame, ids_periodes: list):
    """Croise unités attendues x périodes attendues avec les valeurs saisies.
    Renvoie (grille, résumé par unité)."""
    grille = pd.MultiIndex.from_product(
        [unites["id"].tolist(), ids_periodes], names=["orgUnit", "Période"]
    ).to_frame(index=False)
    if data.empty:
        presence = pd.DataFrame(columns=["orgUnit", "Période", "nb_valeurs", "nb_non_nuls"])
    else:
        v = pd.to_numeric(data["value"], errors="coerce")
        d = data.assign(_non_nul=(v.isna() | (v != 0)).astype(int))
        presence = (
            d.groupby(["orgUnit", "period"])
            .agg(nb_valeurs=("value", "size"), nb_non_nuls=("_non_nul", "sum"))
            .reset_index()
            .rename(columns={"period": "Période"})
        )
    grille = grille.merge(presence, on=["orgUnit", "Période"], how="left")
    grille["saisi"] = grille["nb_valeurs"].notna()
    grille["nb_non_nuls"] = grille["nb_non_nuls"].fillna(0)
    grille["rang"] = grille["Période"].map({p: i for i, p in enumerate(ids_periodes)})
    historique = grille.groupby("orgUnit")["nb_non_nuls"].transform("max") > 0
    grille["a_zero"] = grille["saisi"] & (grille["nb_non_nuls"] == 0) & historique

    n = len(ids_periodes)
    lignes = []
    for ou, g in grille.sort_values("rang").groupby("orgUnit", sort=False):
        saisi = g["saisi"].tolist()
        retard = 0
        for ok in reversed(saisi):
            if ok:
                break
            retard += 1
        saisies = [p for p, ok in zip(g["Période"], saisi) if ok]
        lignes.append(
            {
                "orgUnit": ou,
                "attendus": n,
                "saisis": len(saisies),
                "manquants": n - len(saisies),
                "derniere_saisie": saisies[-1] if saisies else "Jamais",
                "retard": retard,
                "premiere_manquante": ids_periodes[n - retard] if retard else "",
            }
        )
    resume = pd.DataFrame(lignes)
    return grille, resume


@st.cache_data(ttl=3600, show_spinner=False)
def type_periode_formulaire(url, mode, user, password, token, dataset) -> str:
    s = make_session(mode, user, password, token)
    r = s.get(f"{url}/dataSets/{dataset}.json", params={"fields": "periodType"}, timeout=120)
    r.raise_for_status()
    return r.json().get("periodType", "")


@st.cache_data(ttl=3600, show_spinner=False)
def unites_attendues(url, mode, user, password, token, dataset, orgunit) -> pd.DataFrame:
    """Unités (l'unité choisie et ses descendantes) auxquelles le formulaire est affecté."""
    s = make_session(mode, user, password, token)
    r = s.get(
        f"{url}/organisationUnits.json",
        params={
            "fields": "id,displayName,ancestors[displayName]",
            "paging": "false",
            "filter": [f"path:like:{orgunit}", f"dataSets.id:eq:{dataset}"],
        },
        timeout=300,
    )
    r.raise_for_status()
    lignes = [
        {
            "id": u["id"],
            "Unité d'organisation": u.get("displayName", u["id"]),
            "Hiérarchie": " / ".join(a.get("displayName", "") for a in u.get("ancestors", [])),
        }
        for u in r.json().get("organisationUnits", [])
    ]
    return pd.DataFrame(lignes, columns=["id", "Unité d'organisation", "Hiérarchie"])


def section_manquantes():
    st.caption(
        "Les règles de validation ne se déclenchent pas quand une valeur est absente : une unité "
        "qui ne déclare rien ne génère donc aucune erreur. Cette section croise les unités auxquelles "
        "le formulaire est affecté avec les valeurs réellement saisies, pour repérer les unités "
        "silencieuses, les retards de saisie et les formulaires entièrement à zéro."
    )
    col1, col2 = st.columns(2)
    with col1:
        dataset_id = st.selectbox(
            "Formulaire",
            options=df_ds["id"].tolist(),
            format_func=lambda i: noms_ds.get(i, i),
            key="mq_ds",
        )
    with col2:
        orgunit_id = choisir_unite("mq")
    col3, col4, col5 = st.columns(3)
    aujourdhui = datetime.date.today()
    with col3:
        debut = st.date_input(
            "Début", value=datetime.date(aujourdhui.year - 1, aujourdhui.month, 1), key="mq_debut"
        )
    with col4:
        fin = st.date_input("Fin", value=aujourdhui, key="mq_fin")
    with col5:
        seuil_retard = st.number_input(
            "Retard signalé à partir de (périodes consécutives manquantes)",
            1, 60, 2, step=1, key="mq_retard",
        )
    st.caption(
        "Seules les périodes entièrement écoulées sont analysées. "
        "⚠️ Une grande unité sur une longue période peut produire un volume important de données."
    )

    if st.button(
        "🔎 Détecter les données manquantes",
        type="primary",
        disabled=not (dataset_id and orgunit_id and debut <= fin),
        key="mq_go",
    ):
        try:
            with st.spinner("Extraction et analyse en cours…"):
                type_p = type_periode_formulaire(url, mode, user, password, token, dataset_id)
                periodes = periodes_attendues(type_p, debut, fin, aujourdhui)
                if periodes is None:
                    st.session_state["mq_result"] = {"erreur": f"Type de période non géré : {type_p}."}
                    return
                if not periodes:
                    st.session_state["mq_result"] = {
                        "erreur": "Aucune période entièrement écoulée dans l'intervalle choisi."
                    }
                    return
                unites = unites_attendues(url, mode, user, password, token, dataset_id, orgunit_id)
                if unites.empty:
                    st.session_state["mq_result"] = {
                        "erreur": "Ce formulaire n'est affecté à aucune unité sous l'unité choisie."
                    }
                    return
                data = valeurs_formulaire(
                    url, mode, user, password, token, dataset_id, orgunit_id,
                    debut.isoformat(), fin.isoformat(),
                )
        except Exception as e:
            st.error(f"Extraction impossible : {e}")
            return
        ids_p = [p[0] for p in periodes]
        grille, resume = analyser_saisie(unites, data, ids_p)
        infos = unites.set_index("id")
        resume["Unité d'organisation"] = resume["orgUnit"].map(infos["Unité d'organisation"])
        resume["Hiérarchie"] = resume["orgUnit"].map(infos["Hiérarchie"])
        resume["Taux de saisie (%)"] = (100 * resume["saisis"] / resume["attendus"]).round(0)
        resume["Lien de saisie"] = [
            lien_saisie(dataset_id, o, p) for o, p in zip(resume["orgUnit"], resume["premiere_manquante"])
        ]
        manquants = grille[~grille["saisi"]].copy()
        zeros = grille[grille["a_zero"]].copy()
        for t in (manquants, zeros):
            t["Unité d'organisation"] = t["orgUnit"].map(infos["Unité d'organisation"])
            t["Hiérarchie"] = t["orgUnit"].map(infos["Hiérarchie"])
            t["Lien de saisie"] = [lien_saisie(dataset_id, o, p) for o, p in zip(t["orgUnit"], t["Période"])]
        st.session_state["mq_result"] = {
            "formulaire": noms_ds.get(dataset_id, dataset_id),
            "unite": noms_ou.get(orgunit_id, orgunit_id),
            "type_periode": type_p,
            "periodes": f"{ids_p[0]} → {ids_p[-1]} ({len(ids_p)})",
            "resume": resume,
            "manquants": manquants,
            "zeros": zeros,
            "seuil": int(seuil_retard),
        }

    res = st.session_state.get("mq_result")
    if not res:
        return
    if "erreur" in res:
        st.warning(res["erreur"])
        return
    resume, manquants, zeros = res["resume"], res["manquants"], res["zeros"]
    attendus, saisis = int(resume["attendus"].sum()), int(resume["saisis"].sum())
    silencieuses = resume[resume["saisis"] == 0]
    en_retard = resume[(resume["saisis"] > 0) & (resume["retard"] >= res["seuil"])]
    st.subheader(f"{res['formulaire']} : {res['unite']}")
    st.caption(f"Période {res['type_periode']} : {res['periodes']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unités attendues", f"{len(resume):,}".replace(",", " "))
    c2.metric("Taux de saisie", f"{100 * saisis / attendus:.0f} %" if attendus else "n/a")
    c3.metric("Unités silencieuses", f"{len(silencieuses):,}".replace(",", " "))
    c4.metric("Unités en retard", f"{len(en_retard):,}".replace(",", " "))

    colonnes = ["Unité d'organisation", "Hiérarchie", "attendus", "saisis", "manquants",
                "Taux de saisie (%)", "derniere_saisie", "retard", "Lien de saisie"]
    renommer = {"attendus": "Périodes attendues", "saisis": "Périodes saisies",
                "manquants": "Périodes manquantes", "derniere_saisie": "Dernière période saisie",
                "retard": "Retard (périodes)"}
    config = {**config_liens(), "Taux de saisie (%)": st.column_config.NumberColumn(format="%.0f %%")}
    st.download_button(
        "⬇️ Télécharger le résumé par unité (CSV)",
        data=resume[colonnes].rename(columns=renommer)
        .sort_values("Périodes manquantes", ascending=False)
        .to_csv(index=False).encode("utf-8-sig"),
        file_name="dhis2_saisie_par_unite.csv",
        mime="text/csv",
    )

    def tableau(titre, df, cols, tri, ouvert=True, ico="❌", plafond=5000, croissant=True):
        if df.empty:
            st.success(f"✅ {titre} : aucun cas.")
            return
        sous = df[cols].sort_values(tri, ascending=croissant).rename(columns=renommer)
        with st.expander(f"{ico} {titre} : {len(df)} cas", expanded=ouvert):
            if len(sous) > plafond:
                st.warning(f"Affichage limité aux {plafond} premières lignes : téléchargez le CSV pour tout voir.")
            st.dataframe(
                sous.head(plafond),
                hide_index=True,
                use_container_width=True,
                height=min(500, 40 + 35 * min(len(sous), plafond)),
                column_config=config,
            )

    tableau("Unités silencieuses (aucune saisie sur la période)", silencieuses, colonnes,
            ["Hiérarchie", "Unité d'organisation"], ico="🔇")
    tableau(f"Retard de saisie (au moins {res['seuil']} périodes consécutives manquantes)", en_retard,
            colonnes, ["retard"], ico="⏳", croissant=False)
    tableau("Formulaires entièrement à zéro (unité ayant déjà saisi des valeurs non nulles)", zeros,
            ["Unité d'organisation", "Hiérarchie", "Période", "Lien de saisie"],
            ["Unité d'organisation", "Période"], ouvert=False, ico="⚠️")
    tableau("Détail des périodes non saisies", manquants,
            ["Unité d'organisation", "Hiérarchie", "Période", "Lien de saisie"],
            ["Unité d'organisation", "Période"], ouvert=False, ico="📭")


def onglet_validation():
    sous_regles, sous_aberrantes, sous_variations, sous_manquantes = st.tabs(
        ["📋 Règles de validation", "📈 Valeurs aberrantes", "⏱️ Variations dans le temps",
         "📭 Données manquantes"]
    )
    with sous_regles:
        section_regles()
    with sous_aberrantes:
        section_aberrantes()
    with sous_variations:
        section_variations()
    with sous_manquantes:
        section_manquantes()


tab_ext, tab_comp, tab_qd = st.tabs(["📥 Extraction", "✅ Complétude", "🔍 Validation"])
with tab_ext:
    onglet_extraction()
with tab_comp:
    onglet_completude()
with tab_qd:
    onglet_validation()