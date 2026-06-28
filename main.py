# main.py

import os
import joblib

from src.train import train_xgboost_model


def main():
    """
    اجرای کامل فرایند آموزش مدل و ذخیره‌سازی آن
    """

    # مسیر فایل داده
    data_path = "data/raw/ames_housing.csv"

    # آموزش مدل
    pipeline, results = train_xgboost_model(data_path)

    # ساخت پوشه models اگر وجود نداشته باشد
    os.makedirs("models", exist_ok=True)

    # ذخیره مدل
    model_output_path = "models/xgboost_pipeline.joblib"
    joblib.dump(pipeline, model_output_path)

    # نمایش نتایج
    print("===== Training Completed =====")
    print(f"RMSE: {results['rmse']:.2f}")
    print(f"MAE: {results['mae']:.2f}")
    print(f"R2 Score: {results['r2_score']:.4f}")
    print(f"Model saved to: {model_output_path}")


if __name__ == "__main__":
    main()
