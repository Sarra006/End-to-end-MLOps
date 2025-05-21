import joblib 
import numpy as np
import pandas as pd
from pathlib import Path



class PredictionPipeline:
    def __init__(self, models_dir='artifacts/model_trainer'):
        self.models_dir = Path(models_dir)

    def predict(self, data, model_name: str):
        """
        data: DataFrame or array-like structure to predict on
        model_name: Nom du fichier modèle à utiliser (ex: 'random_forest_mdel.pkl')
        """
        model_path = self.models_dir / model_name

        if not model_path.exists():
            raise FileNotFoundError(f"Le modèle '{model_name}' n'existe pas dans {self.models_dir}")

        model = joblib.load(model_path)
        prediction = model.predict(data)

        return prediction