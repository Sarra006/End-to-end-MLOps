import os
from MLOpsProject.entity.config_entity import DataTransformationConfig
from src.MLOpsProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Dropping unnecessary columns
        if "Unnamed: 32" in df.columns:
            df.drop("Unnamed: 32", axis=1, inplace=True)
            logger.info("Dropped column: Unnamed: 32")

        if "id" in df.columns:
            df.drop("id", axis=1, inplace=True)
            logger.info("Dropped column: id")

        # 2. Encoding categorical variables
        if df['diagnosis'].dtype == object:
            df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})
            logger.info("Encoded 'diagnosis' column: B -> 0, M -> 1")

        return df

    def train_test_spliting(self):
        # Load data
        df = pd.read_csv(self.config.data_path)
        logger.info(f"Loaded data from {self.config.data_path}")

        # Preprocess
        df = self.preprocess_data(df)

        # 3. Splitting into X and y
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']

        # 4. Splitting data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info("Performed train-test split")

        # 5. Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logger.info("Applied StandardScaler to the data")

        # Combine X and y back into train/test DataFrames
        train = pd.DataFrame(X_train_scaled, columns=X.columns)
        train['diagnosis'] = y_train.reset_index(drop=True)

        test = pd.DataFrame(X_test_scaled, columns=X.columns)
        test['diagnosis'] = y_test.reset_index(drop=True)

        # Save files
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)
        logger.info("Saved processed train and test sets")

        print("Train shape:", train.shape)
        print("Test shape:", test.shape)
