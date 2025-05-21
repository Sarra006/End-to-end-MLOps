import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import app

def test_get_clean_data(monkeypatch, tmp_path):
    import pandas as pd
    # Create dummy CSV file
    data = pd.DataFrame({
        "diagnosis": ["M", "B"],
        "radius_mean": [1.0, 2.0]
    })
    file_path = tmp_path / "train.csv"
    data.to_csv(file_path, index=False)

    monkeypatch.setattr(app, "get_clean_data", lambda: pd.read_csv(file_path).assign(diagnosis=lambda df: df['diagnosis'].map({'M':1, 'B':0})))
    df = app.get_clean_data()
    assert "diagnosis" in df.columns
    assert set(df["diagnosis"].unique()) <= {0, 1}

from unittest.mock import patch, mock_open

def test_get_scaled_values():
    import pandas as pd
    input_dict = {
        "radius_mean": 1.0,
        "texture_mean": 2.0,
        "perimeter_mean": 3.0,
        "area_mean": 4.0,
        "smoothness_mean": 5.0,
        "compactness_mean": 6.0,
        "concavity_mean": 7.0,
        "concave points_mean": 8.0,
        "symmetry_mean": 9.0,
        "fractal_dimension_mean": 10.0
    }
    df = pd.DataFrame({
        "radius_mean": [0, 2],
        "texture_mean": [0, 4],
        "perimeter_mean": [0, 6],
        "area_mean": [0, 8],
        "smoothness_mean": [0, 10],
        "compactness_mean": [0, 12],
        "concavity_mean": [0, 14],
        "concave points_mean": [0, 16],
        "symmetry_mean": [0, 18],
        "fractal_dimension_mean": [0, 20],
        "diagnosis": [0, 1]
    })
    with patch('pandas.read_csv', return_value=df):
        scaled = app.get_scaled_values(input_dict)
    for key in input_dict:
        assert 0 <= scaled[key] <= 1

def test_get_radar_chart():
    input_data = {
        "radius_mean": 0.5,
        "texture_mean": 0.5,
        "perimeter_mean": 0.5,
        "area_mean": 0.5,
        "smoothness_mean": 0.5,
        "compactness_mean": 0.5,
        "concavity_mean": 0.5,
        "concave points_mean": 0.5,
        "symmetry_mean": 0.5,
        "fractal_dimension_mean": 0.5,
        "radius_se": 0.5,
        "texture_se": 0.5,
        "perimeter_se": 0.5,
        "area_se": 0.5,
        "smoothness_se": 0.5,
        "compactness_se": 0.5,
        "concavity_se": 0.5,
        "concave points_se": 0.5,
        "symmetry_se": 0.5,
        "fractal_dimension_se": 0.5,
        "radius_worst": 0.5,
        "texture_worst": 0.5,
        "perimeter_worst": 0.5,
        "area_worst": 0.5,
        "smoothness_worst": 0.5,
        "compactness_worst": 0.5,
        "concavity_worst": 0.5,
        "concave points_worst": 0.5,
        "symmetry_worst": 0.5,
        "fractal_dimension_worst": 0.5
    }
    fig = app.get_radar_chart(input_data)
    assert isinstance(fig, go.Figure)

from unittest.mock import patch, mock_open

def test_get_model_comparison_chart():
    import json
    import plotly.graph_objects as go
    metrics = {
        "model1": {"accuracy": 0.9},
        "model2": {"accuracy": 0.8}
    }
    m_open = mock_open(read_data=json.dumps(metrics))
    with patch("builtins.open", m_open):
        with patch("json.load", return_value=metrics):
            with patch("plotly.graph_objects.Figure", return_value=go.Figure()):
                fig = app.get_model_comparison_chart()
    assert fig is not None
