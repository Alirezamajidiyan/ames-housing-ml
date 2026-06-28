# src/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_and_split_data(file_path, target_column="SalePrice", test_size=0.2, random_state=42):
    """
    بارگذاری دیتاست و تقسیم آن به مجموعه‌های آموزش و تست
    """
    df = pd.read_csv(file_path)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def get_preprocessor(X):
    """
    تعریف خط لوله (Pipeline) پیش‌پردازش برای ستون‌های عددی و دسته‌ای
    """
    # شناسایی نوع ستون‌ها
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()

    # تعریف ترنسفورمرها
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # ترکیب ترنسفورمرها در یک ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    
    return preprocessor
