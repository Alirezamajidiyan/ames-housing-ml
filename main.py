# main.py

import os
import joblib

from src.train import train_all_models
from src.evaluate import (
    create_results_dataframe,
    get_best_model,
    save_results_report,
    plot_model_comparison,
)


def main():
    # مسیر فایل دیتاست اصلی
    file_path = "data/raw/ames_housing.csv"

    # اگر پوشه‌های لازم وجود نداشتند، آن‌ها را می‌سازیم
    os.makedirs("reports", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("Starting model training...")

    # همه مدل‌ها را آموزش می‌دهیم و نتایج را می‌گیریم
    trained_models, results_list = train_all_models(file_path)

    # نتایج را به یک جدول مرتب‌شده تبدیل می‌کنیم
    results_df = create_results_dataframe(results_list)

    # گزارش نهایی مدل‌ها را به صورت CSV ذخیره می‌کنیم
    save_results_report(results_df, output_path="reports/model_comparison.csv")

    # نمودار مقایسه مدل‌ها را هم ذخیره می‌کنیم
    plot_model_comparison(
        results_df,
        output_path="reports/figures/model_comparison.png"
    )

    # بهترین مدل را از روی کمترین مقدار RMSE پیدا می‌کنیم
    best_model_row = get_best_model(results_df)
    best_model_name = best_model_row["model_name"]

    print("\nSaving trained model pipelines...")

    # همه pipelineهای آموزش‌دیده را داخل پوشه models ذخیره می‌کنیم
    for model_name, pipeline in trained_models.items():
        model_output_path = f"models/{model_name}_pipeline.joblib"
        joblib.dump(pipeline, model_output_path)
        print(f"Saved: {model_output_path}")

    print("\nModel comparison results:")
    for _, row in results_df.iterrows():
        print(
            f"{row['model_name']}: "
            f"RMSE={row['rmse']:.2f} | "
            f"MAE={row['mae']:.2f} | "
            f"R2={row['r2_score']:.4f}"
        )

    print(f"\nBest model: {best_model_name}")
    print("\nGenerated files:")
    print("- reports/model_comparison.csv")
    print("- reports/figures/model_comparison.png")
    print("- models/linear_regression_pipeline.joblib")
    print("- models/random_forest_pipeline.joblib")
    print("- models/xgboost_pipeline.joblib")
    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
