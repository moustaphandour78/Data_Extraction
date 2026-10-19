"""
Application Streamlit : extraction de données DHIS2
(un formulaire x une unité d'organisation x N dernières années)

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
st.title("📊 Extraction de données DHIS2")
st.caption("Un formulaire × une unité d'organisation × les N dernières années")

try:
    df_ds = lister(url, mode, user, password, token, "dataSets")
    df_ou = lister(url, mode, user, password, token, "organisationUnits")
except Exception as e:
    st.error(f"Connexion impossible : {e}")
    st.stop()

st.sidebar.success(f"{len(df_ds)} formulaires, {len(df_ou)} unités d'organisation")

col1, col2 = st.columns(2)
with col1:
    ds_nom = st.selectbox("Formulaire (dataSet)", df_ds["displayName"])
    dataset_id = df_ds.loc[df_ds["displayName"] == ds_nom, "id"].iloc[0]
with col2:
    saisie_manuelle = st.checkbox("Saisir l'UID de l'unité manuellement")
    if saisie_manuelle:
        orgunit_id = st.text_input("UID de l'unité d'organisation")
    else:
        ou_nom = st.selectbox("Unité d'organisation", df_ou["displayName"])
        orgunit_id = df_ou.loc[df_ou["displayName"] == ou_nom, "id"].iloc[0]

col3, col4 = st.columns(2)
with col3:
    nb_annees = st.slider("Nombre d'années", 1, 15, 5)
with col4:
    enfants = st.checkbox("Inclure les unités enfants", value=False)

annee_fin = datetime.date.today().year
annees = list(range(annee_fin - nb_annees + 1, annee_fin + 1))
st.info(f"Période extraite : {annees[0]} → {annees[-1]}")

if st.button("🚀 Lancer l'extraction", type="primary", disabled=not orgunit_id):
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
        st.stop()
    barre.empty()

    if not morceaux:
        st.warning(
            "Aucune donnée trouvée. Vérifiez le formulaire, l'unité d'organisation "
            "et vos droits de lecture."
        )
        st.session_state.pop("resultat", None)
    else:
        data = pd.concat(morceaux, ignore_index=True)
        # Noms des éléments de données
        s2 = make_session(mode, user, password, token)
        r = s2.get(
            f"{url}/dataElements.json",
            params={"fields": "id,displayName", "paging": "false"},
            timeout=120,
        )
        r.raise_for_status()
        noms_de = {x["id"]: x["displayName"] for x in r.json()["dataElements"]}
        data["dataElement_nom"] = data["dataElement"].map(noms_de)
        data["orgUnit_nom"] = data["orgUnit"].map(
            dictionnaire_noms(url, mode, user, password, token, "organisationUnits")
        )
        data["value_num"] = pd.to_numeric(data["value"], errors="coerce")
        st.session_state["resultat"] = data
        st.session_state["nom_fichier"] = f"dhis2_{dataset_id}_{orgunit_id}.csv"

# ------------------------------------------------------------------
# Résultats
# ------------------------------------------------------------------
if "resultat" in st.session_state:
    data = st.session_state["resultat"]

    st.subheader("✅ Contrôles (à comparer avec DHIS2)")
    ctrl = (
        data.groupby("annee_extraction")
        .agg(nb_lignes=("value", "size"), somme_valeurs=("value_num", "sum"))
        .reset_index()
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(ctrl, hide_index=True, use_container_width=True)
        st.metric("Total de lignes", f"{len(data):,}".replace(",", " "))
    with c2:
        st.bar_chart(ctrl, x="annee_extraction", y="nb_lignes")

    st.subheader("Données extraites")
    st.dataframe(data, use_container_width=True, height=400)

    st.download_button(
        "⬇️ Télécharger en CSV",
        data=data.to_csv(index=False).encode("utf-8-sig"),
        file_name=st.session_state["nom_fichier"],
        mime="text/csv",
    )
