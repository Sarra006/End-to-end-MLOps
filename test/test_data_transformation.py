import pytest
import pandas as pd
from src.MLOpsProject.components.data_transformation import DataTransformation
from src.MLOpsProject.entity.config_entity import DataTransformationConfig

@pytest.fixture
def config(tmp_path):
    # Create a dummy CSV file for testing
    data = {
        "Unnamed: 32": [1, 2],
        "id": [1, 2],
        "diagnosis": ["B", "M"],
        "feature1": [0.1, 0.2]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "data.csv"
    df.to_csv(file_path, index=False)
    return DataTransformationConfig(data_path=str(file_path), root_dir=str(tmp_path))

def test_preprocess_data(config):
    dt = DataTransformation(config)
    df = pd.read_csv(config.data_path)
    processed_df = dt.preprocess_data(df)
    assert "Unnamed: 32" not in processed_df.columns
    assert "id" not in processed_df.columns
    assert set(processed_df["diagnosis"].unique()) <= {0, 1}
