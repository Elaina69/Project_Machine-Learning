"""
Module Visualization — Vẽ biểu đồ kết quả mô hình.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams.update({
    'figure.figsize': (14, 5),
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 100,
})


def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted",
                             n_points=500, save_path=None):
    """Biểu đồ overlay Actual vs Predicted (lấy n_points đầu tiên)."""
    n = min(n_points, len(y_true))
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(range(n), y_true[:n], label='Actual', alpha=0.8, linewidth=1)
    ax.plot(range(n), y_pred[:n], label='Predicted', alpha=0.8, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Flow')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_scatter_pred(y_true, y_pred, title="Scatter: Actual vs Predicted",
                      save_path=None):
    """Scatter plot Actual vs Predicted với đường y=x."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.2, s=5)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='y=x')
    ax.set_xlabel('Actual Flow')
    ax.set_ylabel('Predicted Flow')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_error_distribution(y_true, y_pred, title="Error Distribution",
                            save_path=None):
    """Histogram phân phối sai số (residuals)."""
    errors = np.array(y_true) - np.array(y_pred)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel('Error (Actual - Predicted)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_metrics_bar(results_df: pd.DataFrame, metric='test_RMSE',
                     title=None, save_path=None):
    """Bar chart metrics cho các mô hình trong 1 baseline."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = sns.color_palette("Set2", len(results_df))
    bars = ax.bar(results_df['model'], results_df[metric], color=colors)
    ax.set_title(title or f'{metric} theo mô hình')
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, results_df[metric]):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_all_model_results(trained_models: dict, y_test, baseline_name: str,
                           save_dir: str = None, n_points: int = 300):
    """Vẽ Actual vs Predicted cho tất cả mô hình trong 1 baseline."""
    n_models = len(trained_models)
    fig, axes = plt.subplots(n_models, 1, figsize=(16, 3.5 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]

    n = min(n_points, len(y_test))
    for ax, (name, data) in zip(axes, trained_models.items()):
        y_pred = data['y_pred_test']
        ax.plot(range(n), y_test[:n], label='Actual', alpha=0.8, linewidth=1)
        ax.plot(range(n), y_pred[:n], label='Predicted', alpha=0.8, linewidth=1)
        metrics = data['test_metrics']
        ax.set_title(f"{name} — MAE={metrics['MAE']:.2f}  "
                     f"RMSE={metrics['RMSE']:.2f}  R²={metrics['R2']:.4f}")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Actual vs Predicted — {baseline_name}', fontsize=15, y=1.01)
    plt.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{save_dir}/{baseline_name}_all_models.png", bbox_inches='tight')
    plt.show()
