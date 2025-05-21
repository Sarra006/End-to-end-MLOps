import pytest
import pandas as pd
from unittest.mock import patch, mock_open
from src.MLOpsProject.components.data_validation import DataValiadtion
from src.MLOpsProject.entity.config_entity import DataValidationConfig

@pytest.fixture
def config(tmp_path):
    # Create a dummy CSV file for testing
    data = {
        "col1": [1, 2],
        "col2": [3, 4]
    }
    file_path = tmp_path / "data.csv"
    pd.DataFrame(data).to_csv(file_path, index=False)
    schema = {"col1": "int", "col2": "int"}
    status_file = tmp_path / "status.txt"
    return DataValidationConfig(unzip_data_dir=str(file_path), all_schema=schema, STATUS_FILE=str(status_file), root_dir=str(tmp_path))

def test_validate_all_columns_valid(config):
    validator = DataValiadtion(config)
    result = validator.validate_all_columns()
    assert result is True

def test_validate_all_columns_invalid(config):
    # Create a new config with modified schema to cause validation failure
    from src.MLOpsProject.entity.config_entity import DataValidationConfig
    new_config = DataValidationConfig(
        unzip_data_dir=config.unzip_data_dir,
        all_schema={"col1": "int"},
        STATUS_FILE=config.STATUS_FILE,
        root_dir=config.root_dir
    )
    validator = DataValiadtion(new_config)
    result = validator.validate_all_columns()
    assert result is False
