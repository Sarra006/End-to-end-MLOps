import pandas as pd
import os
from MLOpsProject.entity.config_entity import ModelTrainerConfig
from src.MLOpsProject import logger

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]


        # Liste des modèles à tester
        models = {
            "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs'),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='minkowski'),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, criterion='gini'),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            "Support Vector Machine": SVC(C=1.0, kernel='rbf', probability=True),
            "Naive Bayes": GaussianNB(var_smoothing=1e-9),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),
            "AdaBoost": AdaBoostClassifier(n_estimators=50, learning_rate=1.0),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', learning_rate=0.1, max_depth=3, n_estimators=100)
        }

        # Entraînement et évaluation
        for name, model in models.items():
            model.fit(train_x, train_y)
            preds = model.predict(test_x)
            acc = accuracy_score(test_y, preds)
            print(f"\nModel: {name}")
            print(f"Accuracy: {acc:.4f}")
            print(classification_report(test_y, preds))
            
            # Sauvegarde du modèle
            joblib.dump(model, os.path.join(self.config.root_dir, f"{name.replace(' ', '_').lower()}_model.pkl"))

