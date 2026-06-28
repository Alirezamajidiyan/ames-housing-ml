# src/evaluate.py

import os
import pandas as pd


def create_results_dataframe(results_list):
    """
    لیست نتایج مدل‌ها را به یک DataFrame تبدیل می‌کند
    و آن را بر اساس RMSE مرتب می‌کند.
    """

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by="rmse", ascending=True).reset_index(drop=True)

    return results_df


def get_best_model(results_df):
    """
    بهترین مدل را از روی کمترین RMSE برمی‌گرداند.
    """

    best_model_row = results_df.iloc[0]
    return best_model_row


def save_results_report(results_df, output_path="reports/model_comparison.csv"):
    """
    جدول مقایسه مدل‌ها را داخل پوشه reports ذخیره می‌کند.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
