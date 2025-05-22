import pytest

if __name__ == "__main__":
    pytest.main(["-v", "./test/test_data_transformation.py", "./test/test_model_trainer.py", "./test/test_model_evaluation.py", "./test/test_app.py"])
