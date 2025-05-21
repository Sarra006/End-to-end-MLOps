import pytest
from unittest.mock import patch, MagicMock
from src.MLOpsProject.components.model_trainer import ModelTrainer
from src.MLOpsProject.entity.config_entity import ModelTrainerConfig
import pandas as pd

@pytest.fixture
def config(tmp_path):
    import pandas as pd
    train_data = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "target": [0, 1, 0, 1, 0]
    })
    test_data = pd.DataFrame({
        "feature1": [6, 7, 8, 9, 10],
        "target": [1, 0, 1, 0, 1]
    })
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    return ModelTrainerConfig(
        train_data_path=str(train_path),
        test_data_path=str(test_path),
        target_column="target",
        root_dir=str(tmp_path)
    )

@patch("src.MLOpsProject.components.model_trainer.joblib.dump")
def test_train(mock_dump, config):
    trainer = ModelTrainer(config)
    trainer.train()
    assert mock_dump.call_count > 0
