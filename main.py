# main.py

import os
import joblib

from src.train import train_all_models
from src.evaluate import (
    create_results_dataframe,
    get_best_model,
    save_results_report,
)


def main():
    file_path = "data/raw/ames_housing.csv"

    trained_models, results_list = train_all_models(file_path)

    results_df = create_results_dataframe(results_list)
    save_results_report(results_df, output_path="reports/model_comparison.csv")

    best_model_row = get_best_model(results_df)
    best_model_name = best_model_row["model_name"]
    best_model_pipeline = trained_models[best_model_name]

    os.makedirs("models", exist_ok=True)
    model_output_path = f"models/{best_model_name}_pipeline.joblib"
    joblib.dump(best_model_pipeline, model_output_path)

    print("Model comparison results:")
    print(results_df.to_string(index=False))
    print("\nBest model:")
    print(best_model_row.to_string())
    print(f"\nSaved best model to: {model_output_path}")


if __name__ == "__main__":
    main()
