"""
Module EDA — Khám phá và phân tích dữ liệu PeMSD3.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Cấu hình mặc định cho matplotlib
plt.rcParams.update({
    'figure.figsize': (14, 5),
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 100,
})
TIME_SLOT_LABELS = {0: 'Đêm', 1: 'Sáng', 2: 'Trưa', 3: 'Chiều', 4: 'Tối'}


def describe_per_sensor(df: pd.DataFrame) -> pd.DataFrame:
    """Thống kê mô tả (mean, std, min, max) cho flow, speed, occupancy per sensor."""
    stats = df.groupby('sensor_id')[['flow', 'speed', 'occupancy']].describe()
    stats = stats.round(2)
    return stats


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Kiểm tra missing values và timestamp gaps per sensor."""
    results = []
    for sid, group in df.groupby('sensor_id'):
        group = group.sort_values('timestamp')
        n_rows = len(group)
        n_null = int(group[['flow', 'speed', 'occupancy']].isnull().sum().sum())
        # Kiểm tra gaps (khoảng > 5 phút giữa 2 dòng liên tiếp)
        diffs = group['timestamp'].diff().dt.total_seconds()
        n_gaps = int((diffs > 300 + 30).sum())  # >5.5 phút = gap
        results.append({
            'sensor_id': sid,
            'n_rows': n_rows,
            'n_null_values': n_null,
            'missing_rate_%': round(n_null / (n_rows * 3) * 100, 2),
            'n_time_gaps': n_gaps,
        })
    return pd.DataFrame(results)


def plot_timeseries_flow(df: pd.DataFrame, save_dir: str = None):
    """Vẽ biểu đồ time-series flow cho từng sensor."""
    sensors = sorted(df['sensor_id'].unique())
    n = len(sensors)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, sid in zip(axes, sensors):
        subset = df[df['sensor_id'] == sid].sort_values('timestamp')
        ax.plot(subset['timestamp'], subset['flow'], linewidth=0.5, alpha=0.8)
        ax.set_ylabel('Flow')
        ax.set_title(f'Flow — {sid}')
        ax.grid(True, alpha=0.3)

    plt.xlabel('Timestamp')
    plt.suptitle('Time-series Flow cho từng Sensor', fontsize=15, y=1.01)
    plt.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{save_dir}/timeseries_flow.png", bbox_inches='tight')
    plt.show()


def plot_distribution(df: pd.DataFrame, save_dir: str = None):
    """So sánh distribution flow giữa các sensors (boxplot)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col in zip(axes, ['flow', 'speed', 'occupancy']):
        sns.boxplot(data=df, x='sensor_id', y=col, ax=ax, palette='Set2')
        ax.set_title(f'Distribution of {col}')
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{save_dir}/distribution_boxplot.png", bbox_inches='tight')
    plt.show()


def plot_correlation(
    df: pd.DataFrame,
    save_dir: str = None,
    feature_cols: list[str] | None = None,
    target_col: str = 'flow_target',
):
    """Vẽ heatmap correlation sau feature engineering.

    Nếu truyền feature_cols, heatmap dùng đúng các feature model-ready và target.
    Nếu không, hàm fallback về các cột numeric hiện có trong DataFrame.
    """
    if feature_cols is not None:
        corr_cols = [col for col in [*feature_cols, target_col] if col in df.columns]
    else:
        corr_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(corr_cols) < 2:
        raise ValueError('Cần ít nhất 2 cột numeric để vẽ correlation heatmap.')

    corr = df[corr_cols].corr()
    n_cols = len(corr_cols)
    figsize = (max(10, min(24, n_cols * 0.45)), max(8, min(24, n_cols * 0.45)))
    annot = n_cols <= 12

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        annot=annot,
        cmap='coolwarm',
        center=0,
        fmt='.2f',
        ax=ax,
        square=True,
        cbar_kws={'shrink': 0.75},
    )
    ax.set_title('Correlation Matrix sau Feature Engineering')
    ax.tick_params(axis='x', rotation=90)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{save_dir}/correlation_heatmap.png", bbox_inches='tight')
    plt.show()


def plot_hourly_pattern(df: pd.DataFrame, save_dir: str = None):
    """Biểu đồ flow trung bình theo giờ (weekday vs weekend)."""
    df_tmp = df.copy()
    df_tmp['hour'] = df_tmp['timestamp'].dt.hour
    df_tmp['is_weekend'] = (df_tmp['timestamp'].dt.dayofweek >= 5)
    df_tmp['day_type'] = df_tmp['is_weekend'].map({True: 'Weekend', False: 'Weekday'})

    fig, ax = plt.subplots(figsize=(12, 5))
    for dt in ['Weekday', 'Weekend']:
        subset = df_tmp[df_tmp['day_type'] == dt]
        hourly = subset.groupby('hour')['flow'].mean()
        ax.plot(hourly.index, hourly.values, marker='o', label=dt, linewidth=2)

    ax.set_xlabel('Giờ trong ngày')
    ax.set_ylabel('Flow trung bình')
    ax.set_title('Flow trung bình theo giờ — Weekday vs Weekend')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(24))
    plt.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{save_dir}/hourly_pattern.png", bbox_inches='tight')
    plt.show()


def run_full_eda(df: pd.DataFrame, save_dir: str = None):
    """Chạy toàn bộ EDA pipeline."""
    print("=" * 60)
    print("📊 BẮT ĐẦU EDA")
    print("=" * 60)

    # 1. Thống kê mô tả
    print("\n--- 1. Thống kê mô tả per sensor ---")
    stats = describe_per_sensor(df)
    print(stats.to_string())

    # 2. Missing data
    print("\n--- 2. Kiểm tra Missing Data ---")
    missing = check_missing(df)
    print(missing.to_string(index=False))

    # 3. Plots
    print("\n--- 3. Biểu đồ ---")
    plot_timeseries_flow(df, save_dir)
    plot_distribution(df, save_dir)
    plot_hourly_pattern(df, save_dir)

    print("\n✅ EDA hoàn tất!")
    return stats, missing
