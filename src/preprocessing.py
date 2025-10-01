"""Preprocessing helpers: encoding, scaling, splitting, feature engineering."""
from typing import Tuple, List, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

TARGET = "Revenue"  # for UCI dataset

def build_preprocess_pipeline(df: pd.DataFrame) -> Tuple[Pipeline, List[str]]:
    """Create a ColumnTransformer/Pipeline for one-hot encoding and scaling."""
    numeric = df.select_dtypes(include=['int64','float64','int32','float32']).columns.tolist()
    if TARGET in numeric:
        numeric.remove(TARGET)
    categorical = [c for c in df.columns if c not in numeric + [TARGET]]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    pipe = Pipeline(steps=[("pre", pre)])
    return pipe, categorical

def train_val_test_split(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """Split into train, val, test (stratified if binary)."""
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    rel_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=rel_val, stratify=y_temp, random_state=random_state)
    return X_train, y_train, X_val, y_val, X_test, y_test
