# src/train.py

from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.preprocessing import load_and_split_data, get_preprocessor


def build_model(model_name):
    """
    بر اساس اسم مدل، مدل مناسب را می‌سازد.
    """

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    }

    if model_name not in models:
        raise ValueError(
            f"مدل '{model_name}' معتبر نیست. "
            f"مدل‌های مجاز: {list(models.keys())}"
        )

    return models[model_name]


def train_and_evaluate_model(file_path, model_name):
    """
    یک مدل مشخص را train می‌کند و metricها را برمی‌گرداند.
    """

    # بارگذاری و تقسیم داده
    X_train, X_test, y_train, y_test = load_and_split_data(file_path=file_path)

    # ساخت preprocessor
    preprocessor = get_preprocessor(X_train)

    # ساخت مدل
    model = build_model(model_name)

    # ساخت pipeline نهایی
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # آموزش
    pipeline.fit(X_train, y_train)

    # پیش‌بینی
    y_pred = pipeline.predict(X_test)

    # ارزیابی
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results = {
        "model_name": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2
    }

    return pipeline, results


def train_all_models(file_path):
    """
    همه مدل‌ها را train می‌کند و نتایج را به صورت لیست برمی‌گرداند.
    """

    model_names = ["linear_regression", "random_forest", "xgboost"]

    trained_models = {}
    results_list = []

    for model_name in model_names:
        pipeline, results = train_and_evaluate_model(
            file_path=file_path,
            model_name=model_name
        )

        trained_models[model_name] = pipeline
        results_list.append(results)

    return trained_models, results_list
