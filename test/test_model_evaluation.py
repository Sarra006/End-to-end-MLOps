import pytest
import numpy as np
import sys
import os

# ✅ Ajout du dossier src au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from MLOpsProject.components.model_evaluation import ModelEvaluation
from MLOpsProject.entity.config_entity import ModelEvaluationConfig

@pytest.fixture
def config():
    return ModelEvaluationConfig(
        test_data_path="dummy_path",
        target_column="target",
        mlflow_uri="dummy_uri",
        models_path="dummy_models_path",
        metrics_file_name="dummy_metrics.json",
        root_dir="."
    )

def test_eval_metrics(config):
    evaluator = ModelEvaluation(config)
    actual = np.array([1, 0, 1, 1, 0])
    pred = np.array([1, 0, 0, 1, 0])
    accuracy, precision, recall, f1 = evaluator.eval_metrics(actual, pred)
    assert 0 <= accuracy <= 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1
