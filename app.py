import streamlit as st 
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import joblib
import os
import json

from src.MLOpsProject.pipeline.prediction import PredictionPipeline

# --- Chargement des données ---
def get_clean_data():
    data = pd.read_csv("artifacts/data_transformation/train.csv")
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


# --- Barre latérale ---
def add_sidebar():
    st.sidebar.header("Sélection du modèle & mesures cellulaires")

    # Noms des modèles
    models_path = Path("artifacts/model_trainer")
    model_files = [f.name for f in models_path.glob("*.pkl")]
    selected_model = st.sidebar.selectbox("Choisissez un modèle", model_files)

    # Curseurs de caractéristiques
    data = get_clean_data()

    slider_labels = [
        ("Rayon (moyenne)", "radius_mean"),
        ("Texture (moyenne)", "texture_mean"),
        ("Périmètre (moyenne)", "perimeter_mean"),
        ("Aire (moyenne)", "area_mean"),
        ("Lissage (moyenne)", "smoothness_mean"),
        ("Compacité (moyenne)", "compactness_mean"),
        ("Concavité (moyenne)", "concavity_mean"),
        ("Points concaves (moyenne)", "concave points_mean"),
        ("Symétrie (moyenne)", "symmetry_mean"),
        ("Dimension fractale (moyenne)", "fractal_dimension_mean"),
        ("Rayon (écart-type)", "radius_se"),
        ("Texture (écart-type)", "texture_se"),
        ("Périmètre (écart-type)", "perimeter_se"),
        ("Aire (écart-type)", "area_se"),
        ("Lissage (écart-type)", "smoothness_se"),
        ("Compacité (écart-type)", "compactness_se"),
        ("Concavité (écart-type)", "concavity_se"),
        ("Points concaves (écart-type)", "concave points_se"),
        ("Symétrie (écart-type)", "symmetry_se"),
        ("Dimension fractale (écart-type)", "fractal_dimension_se"),
        ("Rayon (pire)", "radius_worst"),
        ("Texture (pire)", "texture_worst"),
        ("Périmètre (pire)", "perimeter_worst"),
        ("Aire (pire)", "area_worst"),
        ("Lissage (pire)", "smoothness_worst"),
        ("Compacité (pire)", "compactness_worst"),
        ("Concavité (pire)", "concavity_worst"),
        ("Points concaves (pire)", "concave points_worst"),
        ("Symétrie (pire)", "symmetry_worst"),
        ("Dimension fractale (pire)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )

    return input_dict, selected_model


# --- Mise à l'échelle pour le radar chart ---
def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict


# --- Radar Chart ---
def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area',
                  'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Valeurs Moyennes'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Erreur Standard'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Valeurs Pires'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True
    )

    return fig


# --- Affichage des prédictions ---
def add_predictions(input_data, model_name):
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    pipeline = PredictionPipeline(models_dir="artifacts/model_trainer")
    prediction = pipeline.predict(input_array, model_name=model_name)

    # Charger les métriques
    with open("artifacts/model_evaluation/metrics.json") as f:
        metrics = json.load(f)

    model_metrics = metrics.get(model_name, {})

    st.subheader("Prédiction du cluster cellulaire")
    st.write("Le cluster est :")

    if prediction[0] == 0:
        st.markdown("<span class='diagnosis benign'>Bénin</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='diagnosis malicious'>Malin</span>", unsafe_allow_html=True)

    st.markdown(
        """
        <small>Cette application peut assister les professionnels de santé, 
        mais ne remplace pas un diagnostic médical professionnel.</small>
        """,
        unsafe_allow_html=True
    )

    if model_metrics:
        st.markdown("---")
        st.markdown("### 📊 Performance du modèle sélectionné")
        st.write(f"**Accuracy** : {model_metrics['accuracy']:.2f}")
        st.write(f"**Precision** : {model_metrics['precision']:.2f}")
        st.write(f"**Recall** : {model_metrics['recall']:.2f}")
        st.write(f"**F1 Score** : {model_metrics['f1_score']:.2f}")
    else:
        st.warning("Aucune métrique trouvée pour ce modèle.")


# --- Bar Chart : comparaison des modèles ---
def get_model_comparison_chart():
    try:
        with open("artifacts/model_evaluation/metrics.json") as f:
            metrics = json.load(f)

        model_names = []
        accuracies = []

        for model, scores in metrics.items():
            model_names.append(model)
            accuracies.append(scores.get("accuracy", 0))

        fig = go.Figure(data=[
            go.Bar(
                x=model_names,
                y=accuracies,
                marker_color='mediumturquoise',
                text=[f"{acc:.2f}" for acc in accuracies],
                textposition='auto'
            )
        ])

        fig.update_layout(
            title="Comparaison des précisions des modèles",
            xaxis_title="Modèles",
            yaxis_title="Précision",
            yaxis=dict(range=[0, 1]),
            template="plotly_white"
        )

        return fig

    except Exception as e:
        st.error(f"Erreur lors du chargement des métriques : {e}")
        return None


# --- Application principale ---
def main():
    st.set_page_config(
        page_title="Prédicteur de cancer du sein",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    input_data, selected_model = add_sidebar()

    with st.container():
        st.title("Prédicteur de Cancer du Sein")
        st.write(
            "Connectez cette application à votre laboratoire pour aider à diagnostiquer un cancer du sein. "
            "Mettez à jour les mesures via la barre latérale. L'application utilise le machine learning pour "
            "classifier l’échantillon comme bénin ou malin."
        )

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart, use_container_width=True)

    with col2:
        add_predictions(input_data, selected_model)

    with st.container():
        st.markdown("### 📈 Comparaison des modèles")
        model_chart = get_model_comparison_chart()
        if model_chart:
            st.plotly_chart(model_chart, use_container_width=True)

    st.markdown(
        """
        <hr>
        <div style="text-align: center; color: gray; font-size: 14px;">
        Modèle et application créés par <strong>EL FAIEZ Sarra</strong>.
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    import os
    os.system("streamlit run app.py --server.address=0.0.0.0 --server.port=8501")
    main()
    
