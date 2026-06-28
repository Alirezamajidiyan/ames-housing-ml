# src/train.py

from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from src.preprocessing import load_and_split_data, get_preprocessor


def train_xgboost_model(file_path):
    """
    این تابع:
    1) داده را لود می‌کند
    2) train/test split انجام می‌دهد
    3) preprocessor را می‌سازد
    4) pipeline مدل XGBoost را train می‌کند
    5) metricها را برمی‌گرداند
    """

    # داده‌ها را می‌خوانیم و split می‌کنیم
    X_train, X_test, y_train, y_test = load_and_split_data(file_path=file_path)

    # preprocessor را بر اساس X_train می‌سازیم
    preprocessor = get_preprocessor(X_train)

    # مدل XGBoost
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    # pipeline نهایی
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # آموزش مدل
    pipeline.fit(X_train, y_train)

    # پیش‌بینی
    y_pred = pipeline.predict(X_test)

    # محاسبه metricها
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results = {
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2
    }

    return pipeline, results
