"""
Module Comparison — So sánh kết quả giữa 2 Baselines.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def merge_results(results_a: pd.DataFrame, results_b: pd.DataFrame) -> pd.DataFrame:
    """Gộp kết quả 2 baselines thành 1 bảng để so sánh."""
    df = pd.concat([results_a, results_b], ignore_index=True)
    return df


def create_comparison_table(results_a: pd.DataFrame,
                            results_b: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo bảng so sánh ngang: mỗi dòng = 1 mô hình,
    cột gồm metrics của cả A và B.
    """
    a = results_a.set_index('model')
    b = results_b.set_index('model')

    compare = pd.DataFrame(index=a.index)
    for metric in ['test_MAE', 'test_RMSE', 'test_MAPE', 'test_R2']:
        short = metric.replace('test_', '')
        compare[f'A_{short}'] = a[metric] if metric in a.columns else np.nan
        compare[f'B_{short}'] = b[metric] if metric in b.columns else np.nan

    # Xác định baseline nào tốt hơn (MAE thấp hơn = tốt hơn)
    compare['better'] = compare.apply(
        lambda row: 'A' if row.get('A_MAE', np.inf) < row.get('B_MAE', np.inf) else 'B',
        axis=1
    )
    return compare.reset_index()


def plot_comparison_grouped_bar(results_a: pd.DataFrame, results_b: pd.DataFrame,
                                metric='test_RMSE', save_dir=None):
    """Grouped bar chart so sánh 1 metric giữa 2 baselines."""
    models = results_a['model'].tolist()
    vals_a = results_a[metric].tolist()
    vals_b = results_b[metric].tolist()

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars_a = ax.bar(x - width/2, vals_a, width, label='Baseline A', color='#4C72B0')
    bars_b = ax.bar(x + width/2, vals_b, width, label='Baseline B', color='#DD8452')

    ax.set_ylabel(metric.replace('test_', ''))
    ax.set_title(f'So sánh {metric.replace("test_", "")} — Baseline A vs B')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars_a, vals_a):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars_b, vals_b):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{save_dir}/comparison_{metric}.png", bbox_inches='tight')
    plt.show()


def plot_comparison_all_metrics(results_a: pd.DataFrame, results_b: pd.DataFrame,
                                save_dir=None):
    """Vẽ grouped bar cho tất cả 4 metrics."""
    for metric in ['test_MAE', 'test_RMSE', 'test_MAPE', 'test_R2']:
        if metric in results_a.columns and metric in results_b.columns:
            plot_comparison_grouped_bar(results_a, results_b, metric, save_dir)


def plot_comparison_radar(results_a: pd.DataFrame, results_b: pd.DataFrame,
                          save_dir=None):
    """Radar chart so sánh R² giữa 2 baselines (R² cao hơn = tốt hơn)."""
    models = results_a['model'].tolist()
    r2_a = results_a['test_R2'].tolist()
    r2_b = results_b['test_R2'].tolist()

    # Chuẩn bị radar
    N = len(models)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    r2_a += r2_a[:1]
    r2_b += r2_b[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, r2_a, alpha=0.25, color='#4C72B0')
    ax.plot(angles, r2_a, 'o-', label='Baseline A', color='#4C72B0', linewidth=2)
    ax.fill(angles, r2_b, alpha=0.25, color='#DD8452')
    ax.plot(angles, r2_b, 'o-', label='Baseline B', color='#DD8452', linewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(models, size=9)
    ax.set_title('So sánh R² — Baseline A vs B', size=14, pad=20)
    ax.legend(loc='lower right')

    plt.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{save_dir}/comparison_radar_r2.png", bbox_inches='tight')
    plt.show()


def generate_conclusion(results_a: pd.DataFrame, results_b: pd.DataFrame) -> str:
    """Tạo nhận xét tự động dựa trên kết quả."""
    lines = []
    lines.append("=" * 60)
    lines.append("📋 NHẬN XÉT & KẾT LUẬN")
    lines.append("=" * 60)

    # Tìm mô hình tốt nhất mỗi baseline
    best_a_idx = results_a['test_RMSE'].idxmin()
    best_b_idx = results_b['test_RMSE'].idxmin()
    best_a = results_a.loc[best_a_idx]
    best_b = results_b.loc[best_b_idx]

    lines.append(f"\n🏆 Mô hình tốt nhất Baseline A: {best_a['model']}")
    lines.append(f"   RMSE={best_a['test_RMSE']:.2f}, MAE={best_a['test_MAE']:.2f}, "
                 f"R²={best_a['test_R2']:.4f}")
    lines.append(f"\n🏆 Mô hình tốt nhất Baseline B: {best_b['model']}")
    lines.append(f"   RMSE={best_b['test_RMSE']:.2f}, MAE={best_b['test_MAE']:.2f}, "
                 f"R²={best_b['test_R2']:.4f}")

    # So sánh tổng thể
    avg_rmse_a = results_a['test_RMSE'].mean()
    avg_rmse_b = results_b['test_RMSE'].mean()
    better = 'A' if avg_rmse_a < avg_rmse_b else 'B'
    lines.append(f"\n📊 RMSE trung bình: A={avg_rmse_a:.2f}, B={avg_rmse_b:.2f}")
    lines.append(f"   → Baseline {better} có kết quả dự báo tốt hơn tổng thể.")

    # So sánh từng mô hình
    lines.append(f"\n📝 So sánh chi tiết từng mô hình:")
    for _, row_a in results_a.iterrows():
        name = row_a['model']
        row_b = results_b[results_b['model'] == name]
        if len(row_b) == 0:
            continue
        row_b = row_b.iloc[0]
        rmse_a = row_a['test_RMSE']
        rmse_b = row_b['test_RMSE']
        winner = 'A' if rmse_a < rmse_b else 'B'
        diff = abs(rmse_a - rmse_b)
        lines.append(f"   {name}: A={rmse_a:.2f} vs B={rmse_b:.2f} "
                     f"→ Baseline {winner} tốt hơn (chênh {diff:.2f})")

    conclusion = "\n".join(lines)
    print(conclusion)
    return conclusion
