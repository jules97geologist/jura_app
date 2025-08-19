import streamlit as st
import toml
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler

model_kmeans = joblib.load("kmeans_model.joblib")
model_dbscan = joblib.load("model_dbscan.joblib")
model_isfor = joblib.load("model_isfor.joblib")

modeles = {
    "K-Means": model_kmeans,
    "DBSCAN": model_dbscan,
    "Isolation Forest": model_isfor
}
# Afficher le titre de l'appli
st.title("Prédiction de la Pollution dans la zone de Jura.")
# Ajouter une image comme logo
st.sidebar.image("jura.png", use_container_width=True)
# Injecter du CSS pour fixer la largeur de la sidebar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 350px;
        max-width: 350px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Ajouter une sidebar pour les radiobuttons et boutton de prédiction
st.sidebar.subheader("Merci de choisir le modèle à utiliser.")
#  Boutons pour le choix du modèle
choix = st.sidebar.radio(
    "Modèles",
    list(modeles.keys()),  # options
    index=1,  # valeur par défaut (Banane)
)
# Confirmation du modèle choisi
st.sidebar.write(f"Modèle choisi: {choix}")
#  importation des packages requis
import streamlit as st
import pandas as pd
#  Menu d'import de fichier
uploaded_file = st.file_uploader("Importez votre fichier", type=["csv", "xlsx"])

try:
    if uploaded_file is None:
        st.markdown("Merci de charger d'abord votre fichier!")
    else:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, sep=",")
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)

        st.success(f"Fichier {uploaded_file.name} importé avec succès ✅")
        st.dataframe(df)

except Exception as e:
    st.error(f"Erreur lors de l'importation du fichier : {e}")

#  Action quand l'utilisateur choisi un modèle et 
# clique sur le bouton prédire pour la prédiction
if st.sidebar.button(label = "Prédire", type = "primary", use_container_width=True):
    try:
        # ⚠ Adapter la préparation des features
        X = df.select_dtypes(include=['float64', 'int64']).drop('id', axis = 1).dropna()
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(X)

        # Récupérer le modèle choisi
        modele_selectionne = modeles[choix]
        # Prediction
        if modele_selectionne == model_dbscan:
            predictions = modele_selectionne.fit_predict(scaled_data)
        else:
            predictions = modele_selectionne.predict(scaled_data)
        #  Affichage des résultats
        df["Résultat"] = predictions
        st.success("Prédiction terminée ✅")
        st.dataframe(df)

        # Option de téléchargement en csv
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger les résultats", csv, "resultats.csv", "text/csv")

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")