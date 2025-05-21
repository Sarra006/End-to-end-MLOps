import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# ✅ Ajout du dossier src au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# ✅ Imports après avoir modifié sys.path
from MLOpsProject.components.data_ingestion import DataIngestion
from MLOpsProject.entity.config_entity import DataIngestionConfig


@pytest.fixture
def config():
    return DataIngestionConfig(
        source_URL="http://example.com/file.zip",
        local_data_file="../artifacts/data_ingestion/data.zip",
        unzip_dir="../artifacts/data_ingestion",
        root_dir="../artifacts"
    )

def test_download_file_file_exists(config):
    ingestion = DataIngestion(config)
    with patch("os.path.exists", return_value=True), \
         patch("MLOpsProject.utils.common.get_size", return_value=123), \
         patch("os.path.getsize", return_value=123):
        ingestion.download_file()

def test_download_file_file_not_exists(config):
    ingestion = DataIngestion(config)
    with patch("os.path.exists", return_value=False), \
         patch("MLOpsProject.components.data_ingestion.request.urlretrieve") as mock_urlretrieve:
        mock_urlretrieve.return_value = ("file.zip", {"header": "value"})
        ingestion.download_file()
        mock_urlretrieve.assert_called_once_with(url=config.source_URL, filename=config.local_data_file)
