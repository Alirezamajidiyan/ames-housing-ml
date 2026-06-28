# src/evaluate.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_results_dataframe(results_list):
    """
    این تابع لیست نتایج مدل‌ها را می‌گیرد،
    آن را به DataFrame تبدیل می‌کند
    و بر اساس RMSE از بهترین به ضعیف‌ترین مرتب می‌کند.
    """

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by="rmse", ascending=True).reset_index(drop=True)
    return results_df


def get_best_model(results_df):
    """
    این تابع بهترین مدل را از جدول نتایج برمی‌گرداند.
    چون جدول بر اساس RMSE صعودی مرتب شده،
    اولین سطر بهترین مدل است.
    """

    return results_df.iloc[0]


def save_results_report(results_df, output_path="reports/model_comparison.csv"):
    """
    این تابع جدول مقایسه مدل‌ها را
    در قالب فایل CSV ذخیره می‌کند.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)


def plot_model_comparison(results_df, output_path="reports/figures/model_comparison.png"):
    """
    این تابع سه نمودار برای مقایسه مدل‌ها می‌سازد:
    RMSE ،MAE و R2
    و در مسیر مشخص‌شده ذخیره می‌کند.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # یک شکل با سه نمودار کنار هم می‌سازیم
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # نمودار مقایسه RMSE
    sns.barplot(data=results_df, x="model_name", y="rmse", ax=axes[0], palette="Blues_d")
    axes[0].set_title("RMSE Comparison")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=15)

    # نمودار مقایسه MAE
    sns.barplot(data=results_df, x="model_name", y="mae", ax=axes[1], palette="Greens_d")
    axes[1].set_title("MAE Comparison")
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("MAE")
    axes[1].tick_params(axis="x", rotation=15)

    # نمودار مقایسه R2
    sns.barplot(data=results_df, x="model_name", y="r2_score", ax=axes[2], palette="Oranges_d")
    axes[2].set_title("R2 Comparison")
    axes[2].set_xlabel("Model")
    axes[2].set_ylabel("R2 Score")
    axes[2].tick_params(axis="x", rotation=15)

    # چیدمان نمودارها را مرتب می‌کنیم و فایل نهایی را ذخیره می‌کنیم
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
