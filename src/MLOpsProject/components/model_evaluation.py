import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from pathlib import Path

from MLOpsProject.entity.config_entity import ModelEvaluationConfig
from MLOpsProject.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        return accuracy, precision, recall, f1

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Parcourir tous les modèles enregistrés
        model_files = [f for f in os.listdir(self.config.models_path) if f.endswith(('.pkl', '.joblib'))]
        results = {}

        for model_file in model_files:
            model_path = os.path.join(self.config.models_path, model_file)
            model = joblib.load(model_path)

            with mlflow.start_run(run_name=model_file):
                predictions = model.predict(test_x)

                accuracy, precision, recall, f1 = self.eval_metrics(test_y, predictions)

                # Enregistrer les métriques en local
                results[model_file] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                }

                mlflow.log_param("model_file", model_file)
                mlflow.log_metrics({
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                })

                # Log du modèle
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name=model_file.replace(".pkl", "").replace(".joblib", ""))
                else:
                    mlflow.sklearn.log_model(model, "model")

        # Sauvegarde globale des scores
        save_json(path=Path(self.config.metrics_file_name), data=results)

    