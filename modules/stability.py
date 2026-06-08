"""Module Stability — Monte Carlo và time sliding validation.

Các kiểm định trong module này dùng cho giai đoạn sau tối ưu siêu tham số:
    1. Monte Carlo: giữ nguyên split thời gian, chỉ thay đổi random_state
       của thuật toán để đo độ ổn định do tính ngẫu nhiên khi huấn luyện.
    2. Time sliding validation: trượt cửa sổ train/test theo thời gian để
       kiểm tra RMSE có ổn định qua các giai đoạn khác nhau của dữ liệu.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from modules.models import evaluate
from configs.pymooSearchSpaces import SEARCH_SPACES


OBJECTIVE_COLS = {'n_gen', 'RMSE', 'train_time_s', 'complexity'}


def params_from_solution(model_name: str, solution) -> dict:
    """Lấy dict siêu tham số từ 1 dòng Pareto/best compromise."""
    if model_name not in SEARCH_SPACES:
        raise ValueError(f"Model '{model_name}' không có search space Pymoo.")

    if isinstance(solution, pd.Series):
        row = solution.to_dict()
    elif isinstance(solution, dict):
        row = solution
    else:
        row = dict(solution)

    params = {}
    for name, _, _, dtype in SEARCH_SPACES[model_name]['params']:
        if name not in row:
            raise KeyError(f"Thiếu tham số '{name}' trong nghiệm tối ưu.")
        value = row[name]
        if dtype == int:
            value = int(round(value))
        elif dtype == float:
            value = float(value)
        params[name] = value
    return params


def build_model(model_name: str, params: dict, random_state: int = 42,
                n_jobs: int = -1):
    """Tạo model từ tên + params tối ưu, hỗ trợ các model có search space."""
    if model_name == '5_RandomForest':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            **params,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    if model_name == '6_XGBoost':
        from xgboost import XGBRegressor
        return XGBRegressor(
            **params,
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=0,
        )

    if model_name == '4_DecisionTree':
        from sklearn.tree import DecisionTreeRegressor
        return DecisionTreeRegressor(
            **params,
            random_state=random_state,
        )

    if model_name == '2_Ridge':
        from sklearn.linear_model import Ridge
        return Ridge(**params)

    if model_name == '3_KNN':
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(
            **params,
            weights='distance',
            n_jobs=n_jobs,
        )

    raise ValueError(f"Model '{model_name}' chưa được hỗ trợ trong stability.")


def _as_1d(values):
    return np.asarray(values).reshape(-1)


def _concat_xy(X_train, y_train, X_valid=None, y_valid=None):
    """Gộp train + valid sau khi chọn hyperparameters, giữ test cố định."""
    if X_valid is None or y_valid is None:
        return np.asarray(X_train), _as_1d(y_train)
    X_fit = np.vstack([np.asarray(X_train), np.asarray(X_valid)])
    y_fit = np.concatenate([_as_1d(y_train), _as_1d(y_valid)])
    return X_fit, y_fit


def run_monte_carlo(
        model_name: str,
        params: dict,
        X_train,
        y_train,
        X_test,
        y_test,
        baseline_name: str,
        X_valid=None,
        y_valid=None,
        seeds=None,
        n_runs: int = 30,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
    """Chạy Monte Carlo bằng cách đổi random_state, không đổi split dữ liệu.

    Nếu truyền X_valid/y_valid, model sẽ được fit lại trên train+valid vì
    hyperparameters đã được chọn xong; tập test vẫn giữ nguyên để đánh giá.
    """
    if seeds is None:
        seeds = list(range(n_runs))
    else:
        seeds = list(seeds)
        n_runs = len(seeds)

    X_fit, y_fit = _concat_xy(X_train, y_train, X_valid, y_valid)
    y_test = _as_1d(y_test)

    records = []
    print(f"\n{'='*60}")
    print(f"🎲 MONTE CARLO — {baseline_name} — {model_name}")
    print(f"   Runs: {n_runs} | Split thời gian: giữ nguyên | Chỉ đổi random_state")
    print(f"{'='*60}")

    for run_idx, seed in enumerate(seeds, start=1):
        model = build_model(model_name, params, random_state=int(seed),
                            n_jobs=n_jobs)
        t0 = time.time()
        model.fit(X_fit, y_fit)
        train_time = time.time() - t0
        y_pred = model.predict(X_test)
        metrics = evaluate(y_test, y_pred, y_fit)

        records.append({
            'baseline': baseline_name,
            'model': model_name,
            'run': run_idx,
            'random_state': int(seed),
            'train_time_s': round(train_time, 4),
            **metrics,
        })

        print(f"   Run {run_idx:02d}/{n_runs} | seed={seed} | "
              f"RMSE={metrics['RMSE']:.4f} | MAE={metrics['MAE']:.4f}")

    return pd.DataFrame(records)


def summarize_monte_carlo(mc_results: pd.DataFrame,
                          metric: str = 'RMSE',
                          confidence: float = 0.95) -> pd.DataFrame:
    """Tóm tắt mean/std/CV và khoảng tin cậy của Monte Carlo."""
    z = 1.96 if confidence == 0.95 else 1.96
    rows = []
    for (baseline, model), group in mc_results.groupby(['baseline', 'model']):
        values = group[metric].astype(float).values
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        se = std / np.sqrt(len(values)) if len(values) > 0 else 0.0
        rows.append({
            'baseline': baseline,
            'model': model,
            'n_runs': len(values),
            f'{metric}_mean': round(mean, 4),
            f'{metric}_std': round(std, 4),
            f'{metric}_min': round(float(np.min(values)), 4),
            f'{metric}_max': round(float(np.max(values)), 4),
            f'{metric}_ci95_low': round(mean - z * se, 4),
            f'{metric}_ci95_high': round(mean + z * se, 4),
            f'{metric}_cv_%': round((std / mean * 100) if mean else 0.0, 4),
        })
    return pd.DataFrame(rows).sort_values(
        [f'{metric}_mean', f'{metric}_std']
    ).reset_index(drop=True)


def plot_monte_carlo_boxplot(mc_results: pd.DataFrame, metric: str = 'RMSE',
                             save_path: str = None):
    """Biểu đồ hộp cho phân phối metric qua các lần Monte Carlo."""
    df = mc_results.copy()
    df['case'] = df['baseline'] + ' | ' + df['model']
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x='case', y=metric, ax=ax, palette='Set2')
    sns.stripplot(data=df, x='case', y=metric, ax=ax,
                  color='black', alpha=0.35, size=3)
    ax.set_title(f'Monte Carlo Stability — Boxplot {metric}')
    ax.set_xlabel('')
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=25)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def plot_monte_carlo_kde(mc_results: pd.DataFrame, metric: str = 'RMSE',
                         save_path: str = None):
    """Histogram tần suất kèm đường KDE cho kết quả Monte Carlo."""
    df = mc_results.copy()
    df['case'] = df['baseline'] + ' | ' + df['model']
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.histplot(data=df, x=metric, hue='case', kde=True,
                 element='step', stat='density', common_norm=False, ax=ax)
    ax.set_title(f'Monte Carlo Stability — Frequency + KDE ({metric})')
    ax.set_xlabel(metric)
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def plot_monte_carlo_ci(mc_summary: pd.DataFrame, metric: str = 'RMSE',
                        save_path: str = None):
    """Biểu đồ mean và khoảng tin cậy 95% của Monte Carlo."""
    mean_col = f'{metric}_mean'
    low_col = f'{metric}_ci95_low'
    high_col = f'{metric}_ci95_high'

    df = mc_summary.copy()
    df['case'] = df['baseline'] + ' | ' + df['model']
    yerr = np.vstack([
        df[mean_col] - df[low_col],
        df[high_col] - df[mean_col],
    ])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.errorbar(df['case'], df[mean_col], yerr=yerr,
                fmt='o', capsize=6, color='#4C72B0')
    ax.set_title(f'Monte Carlo Stability — 95% Confidence Interval ({metric})')
    ax.set_xlabel('')
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=25)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def create_time_sliding_folds(df: pd.DataFrame,
                              train_ratio: float = 0.60,
                              test_ratio: float = 0.10,
                              step_ratio: float = 0.05,
                              n_folds: int = 5):
    """Tạo các fold trượt thời gian per sensor: train cố định, test cố định."""
    folds = []
    for fold_idx in range(n_folds):
        start_ratio = fold_idx * step_ratio
        train_start = start_ratio
        train_end = start_ratio + train_ratio
        test_start = train_end
        test_end = train_end + test_ratio

        train_parts, test_parts = [], []
        for _, group in df.groupby('sensor_id'):
            group = group.sort_values('timestamp').reset_index(drop=True)
            n = len(group)
            i0 = int(round(n * train_start))
            i1 = int(round(n * train_end))
            i2 = int(round(n * test_start))
            i3 = int(round(n * test_end))
            train_parts.append(group.iloc[i0:i1])
            test_parts.append(group.iloc[i2:i3])

        folds.append({
            'fold': fold_idx + 1,
            'train_pct': f'{int(train_start*100)}-{int(train_end*100)}%',
            'test_pct': f'{int(test_start*100)}-{int(test_end*100)}%',
            'train_df': pd.concat(train_parts, ignore_index=True),
            'test_df': pd.concat(test_parts, ignore_index=True),
        })
    return folds


def run_time_sliding_validation(
        df: pd.DataFrame,
        feature_cols: list,
        target_col: str,
        model_name: str,
        params: dict,
        baseline_name: str,
        random_state: int = 42,
        train_ratio: float = 0.60,
        test_ratio: float = 0.10,
        step_ratio: float = 0.05,
        n_folds: int = 5,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
    """Đánh giá model qua các cửa sổ trượt thời gian."""
    folds = create_time_sliding_folds(
        df,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        step_ratio=step_ratio,
        n_folds=n_folds,
    )

    records = []
    print(f"\n{'='*60}")
    print(f"🪟 TIME SLIDING — {baseline_name} — {model_name}")
    print(f"   Train {train_ratio:.0%} | Test {test_ratio:.0%} | "
          f"Step {step_ratio:.0%} | Folds {n_folds}")
    print(f"{'='*60}")

    for fold in folds:
        train_df = fold['train_df']
        test_df = fold['test_df']
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values

        model = build_model(model_name, params,
                            random_state=random_state, n_jobs=n_jobs)
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        y_pred = model.predict(X_test)
        metrics = evaluate(y_test, y_pred, y_train)

        records.append({
            'baseline': baseline_name,
            'model': model_name,
            'fold': fold['fold'],
            'train_pct': fold['train_pct'],
            'test_pct': fold['test_pct'],
            'n_train': len(train_df),
            'n_test': len(test_df),
            'train_time_s': round(train_time, 4),
            **metrics,
        })

        print(f"   Fold {fold['fold']} | Train {fold['train_pct']} | "
              f"Test {fold['test_pct']} | RMSE={metrics['RMSE']:.4f}")

    return pd.DataFrame(records)


def summarize_time_sliding(sliding_results: pd.DataFrame,
                           metric: str = 'RMSE',
                           stable_cv_threshold: float = 10.0) -> pd.DataFrame:
    """Tóm tắt độ ổn định qua các fold trượt thời gian."""
    rows = []
    for (baseline, model), group in sliding_results.groupby(['baseline', 'model']):
        values = group[metric].astype(float).values
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        cv = (std / mean * 100) if mean else 0.0
        rows.append({
            'baseline': baseline,
            'model': model,
            'n_folds': len(values),
            f'{metric}_mean': round(mean, 4),
            f'{metric}_std': round(std, 4),
            f'{metric}_min': round(float(np.min(values)), 4),
            f'{metric}_max': round(float(np.max(values)), 4),
            f'{metric}_range': round(float(np.max(values) - np.min(values)), 4),
            f'{metric}_cv_%': round(cv, 4),
            'stable_by_cv': cv <= stable_cv_threshold,
        })
    return pd.DataFrame(rows).sort_values(
        [f'{metric}_cv_%', f'{metric}_mean']
    ).reset_index(drop=True)


def plot_time_sliding_rmse(sliding_results: pd.DataFrame,
                           save_path: str = None):
    """Vẽ xu hướng RMSE trên tập test qua các fold trượt thời gian."""
    df = sliding_results.copy()
    df['case'] = df['baseline'] + ' | ' + df['model']

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x='fold', y='RMSE', hue='case',
                 marker='o', linewidth=2, ax=ax)

    for case, group in df.groupby('case'):
        mean_rmse = group['RMSE'].mean()
        ax.hlines(mean_rmse, group['fold'].min(), group['fold'].max(),
                  linestyles='dashed', alpha=0.45)

    ax.set_title('Xu hướng RMSE trên tập Test qua 5 cửa sổ trượt')
    ax.set_xlabel('Cửa sổ trượt / Fold')
    ax.set_ylabel('RMSE Test')
    ax.set_xticks(sorted(df['fold'].unique()))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


# ─── Final Model Selection ────────────────────────────────────────────────

DEFAULT_FINAL_SELECTION_WEIGHTS = {
    # Hiệu năng dự báo sau tối ưu, đo trên test cố định qua Monte Carlo.
    'mc_RMSE_mean': 0.30,
    # Khả năng tổng quát qua các giai đoạn thời gian khác nhau.
    'sliding_RMSE_mean': 0.25,
    # Chất lượng nghiệm cân bằng trên validation trong Pymoo.
    'pareto_valid_RMSE': 0.10,
    # Độ nhạy với random_state của thuật toán/model.
    'mc_RMSE_cv_%': 0.15,
    # Độ ổn định của dữ liệu qua các cửa sổ trượt thời gian.
    'sliding_RMSE_cv_%': 0.15,
    # Hai tiêu chí phụ từ Pymoo để tránh chọn mô hình quá nặng.
    'pareto_train_time_s': 0.025,
    'pareto_complexity': 0.025,
}


def get_final_selection_metrics(weights: dict = None) -> pd.DataFrame:
    """Liệt kê các tiêu chí dùng để chọn mô hình tối ưu cuối cùng.

    Tất cả tiêu chí trong bảng điểm đều được chuẩn hóa min-max theo hướng
    "càng thấp càng tốt". R²/MAE/MASE vẫn được đưa vào scorecard để báo cáo,
    nhưng mặc định không tham gia điểm tổng hợp vì RMSE là tiêu chí chính của
    đồ án và các kiểm định ổn định đều xoay quanh RMSE.
    """
    weights = DEFAULT_FINAL_SELECTION_WEIGHTS if weights is None else weights
    rows = [
        {
            'metric': 'mc_RMSE_mean',
            'group': 'Hiệu năng test sau Pymoo',
            'direction': 'min',
            'weight': weights.get('mc_RMSE_mean', 0),
            'meaning': 'RMSE trung bình trên test cố định qua các seed Monte Carlo.',
        },
        {
            'metric': 'sliding_RMSE_mean',
            'group': 'Hiệu năng theo thời gian',
            'direction': 'min',
            'weight': weights.get('sliding_RMSE_mean', 0),
            'meaning': 'RMSE trung bình qua 5 cửa sổ trượt thời gian.',
        },
        {
            'metric': 'pareto_valid_RMSE',
            'group': 'Nghiệm Pymoo',
            'direction': 'min',
            'weight': weights.get('pareto_valid_RMSE', 0),
            'meaning': 'RMSE validation của nghiệm cân bằng trên Pareto front.',
        },
        {
            'metric': 'mc_RMSE_cv_%',
            'group': 'Ổn định thuật toán',
            'direction': 'min',
            'weight': weights.get('mc_RMSE_cv_%', 0),
            'meaning': 'Hệ số biến thiên RMSE khi chỉ đổi random_state.',
        },
        {
            'metric': 'sliding_RMSE_cv_%',
            'group': 'Ổn định dữ liệu',
            'direction': 'min',
            'weight': weights.get('sliding_RMSE_cv_%', 0),
            'meaning': 'Hệ số biến thiên RMSE qua các fold trượt thời gian.',
        },
        {
            'metric': 'pareto_train_time_s',
            'group': 'Chi phí',
            'direction': 'min',
            'weight': weights.get('pareto_train_time_s', 0),
            'meaning': 'Thời gian huấn luyện của nghiệm cân bằng trong Pymoo.',
        },
        {
            'metric': 'pareto_complexity',
            'group': 'Chi phí',
            'direction': 'min',
            'weight': weights.get('pareto_complexity', 0),
            'meaning': 'Độ phức tạp mô hình, ví dụ n_estimators x max_depth.',
        },
    ]
    return pd.DataFrame(rows)


def _to_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _values_stats(values, confidence: float = 0.95):
    values = pd.Series(values).dropna().astype(float).values
    if len(values) == 0:
        return {
            'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan,
            'range': np.nan, 'cv_%': np.nan, 'ci95_low': np.nan,
            'ci95_high': np.nan, 'ci95_width': np.nan,
        }

    z = 1.96 if confidence == 0.95 else 1.96
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    se = std / np.sqrt(len(values)) if len(values) > 0 else 0.0
    low = mean - z * se
    high = mean + z * se
    return {
        'mean': mean,
        'std': std,
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'range': float(np.max(values) - np.min(values)),
        'cv_%': (std / mean * 100) if mean else 0.0,
        'ci95_low': low,
        'ci95_high': high,
        'ci95_width': high - low,
    }


def _solution_to_dict(solution) -> dict:
    if isinstance(solution, pd.Series):
        return solution.to_dict()
    if isinstance(solution, dict):
        return dict(solution)
    return dict(solution)


def _params_to_text(params: dict) -> str:
    if not params:
        return '{}'
    parts = []
    for key, value in params.items():
        if isinstance(value, float):
            parts.append(f'{key}={value:.6g}')
        else:
            parts.append(f'{key}={value}')
    return '{' + ', '.join(parts) + '}'


def _minmax_minimize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors='coerce')
    valid = values.dropna()
    if len(valid) == 0:
        return pd.Series(np.ones(len(series)), index=series.index)
    vmin = valid.min()
    vmax = valid.max()
    if np.isclose(vmax, vmin):
        return pd.Series(np.zeros(len(series)), index=series.index)
    score = (values - vmin) / (vmax - vmin)
    return score.fillna(1.0)


def build_final_model_scorecard(
        results_a: pd.DataFrame,
        results_b: pd.DataFrame,
        optimized_solutions: dict,
        optimized_params: dict,
        monte_carlo_results: pd.DataFrame,
        time_sliding_results: pd.DataFrame,
        weights: dict = None,
        stable_cv_threshold: float = 10.0,
    ) -> pd.DataFrame:
    """Ghép Pymoo + Monte Carlo + trượt thời gian để chọn mô hình cuối.

    Scorecard trả lời câu hỏi: "Bộ hyperparameters Pymoo này có thật sự là
    mô hình tốt nhất và ổn định nhất không?" bằng cách so cùng lúc:
        - chất lượng trên test cố định sau tối ưu,
        - độ dao động khi đổi random_state,
        - độ ổn định qua các cửa sổ thời gian,
        - RMSE validation của nghiệm Pymoo,
        - thời gian huấn luyện và độ phức tạp.

    final_score càng thấp càng tốt.
    """
    weights = DEFAULT_FINAL_SELECTION_WEIGHTS if weights is None else weights
    result_lookup = {
        'Baseline_A': results_a.set_index('model'),
        'Baseline_B': results_b.set_index('model'),
    }

    rows = []
    for baseline_name, model_params in optimized_params.items():
        for model_name, params in model_params.items():
            record = {
                'baseline': baseline_name,
                'model': model_name,
                'case': f'{baseline_name} | {model_name}',
                'optimized_params': _params_to_text(params),
            }

            # Kết quả hold-out ban đầu trước tối ưu, dùng làm mốc tham chiếu.
            base_results = result_lookup.get(baseline_name)
            if base_results is not None and model_name in base_results.index:
                base_row = base_results.loc[model_name]
                for metric in ['MAE', 'RMSE', 'MAPE', 'R2', 'MASE']:
                    record[f'holdout_test_{metric}'] = _to_float(
                        base_row.get(f'test_{metric}')
                    )

            # Nghiệm cân bằng trên Pareto front.
            solution = optimized_solutions.get(baseline_name, {}).get(model_name)
            if solution is not None:
                solution_dict = _solution_to_dict(solution)
                record['pareto_valid_RMSE'] = _to_float(solution_dict.get('RMSE'))
                record['pareto_train_time_s'] = _to_float(
                    solution_dict.get('train_time_s')
                )
                record['pareto_complexity'] = _to_float(
                    solution_dict.get('complexity')
                )

            mc_group = monte_carlo_results[
                (monte_carlo_results['baseline'] == baseline_name)
                & (monte_carlo_results['model'] == model_name)
            ]
            record['mc_n_runs'] = len(mc_group)
            if len(mc_group) > 0:
                rmse_stats = _values_stats(mc_group['RMSE'])
                for key, value in rmse_stats.items():
                    record[f'mc_RMSE_{key}'] = value
                for metric in ['MAE', 'MAPE', 'R2', 'MASE', 'train_time_s']:
                    if metric in mc_group.columns:
                        record[f'mc_{metric}_mean'] = float(
                            mc_group[metric].astype(float).mean()
                        )

            sliding_group = time_sliding_results[
                (time_sliding_results['baseline'] == baseline_name)
                & (time_sliding_results['model'] == model_name)
            ]
            record['sliding_n_folds'] = len(sliding_group)
            if len(sliding_group) > 0:
                rmse_stats = _values_stats(sliding_group['RMSE'])
                for key, value in rmse_stats.items():
                    record[f'sliding_RMSE_{key}'] = value
                for metric in ['MAE', 'MAPE', 'R2', 'MASE', 'train_time_s']:
                    if metric in sliding_group.columns:
                        record[f'sliding_{metric}_mean'] = float(
                            sliding_group[metric].astype(float).mean()
                        )

            record['stable_by_mc_cv'] = (
                record.get('mc_RMSE_cv_%', np.inf) <= stable_cv_threshold
            )
            record['stable_by_time_cv'] = (
                record.get('sliding_RMSE_cv_%', np.inf) <= stable_cv_threshold
            )
            record['stable_overall'] = (
                record['stable_by_mc_cv'] and record['stable_by_time_cv']
            )
            rows.append(record)

    scorecard = pd.DataFrame(rows)
    if scorecard.empty:
        return scorecard

    weighted_score = pd.Series(np.zeros(len(scorecard)), index=scorecard.index)
    total_weight = 0.0
    for col, weight in weights.items():
        if col not in scorecard.columns or weight <= 0:
            continue
        norm_col = f'{col}_norm'
        scorecard[norm_col] = _minmax_minimize(scorecard[col])
        weighted_score += scorecard[norm_col] * weight
        total_weight += weight

    if total_weight > 0:
        scorecard['final_score'] = weighted_score / total_weight
    else:
        scorecard['final_score'] = np.nan

    scorecard = scorecard.sort_values(
        ['final_score', 'mc_RMSE_mean', 'sliding_RMSE_cv_%'],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    scorecard.insert(0, 'rank', np.arange(1, len(scorecard) + 1))

    numeric_cols = scorecard.select_dtypes(include=[np.number]).columns
    scorecard[numeric_cols] = scorecard[numeric_cols].round(4)
    return scorecard


def print_final_model_conclusion(scorecard: pd.DataFrame) -> pd.Series:
    """In kết luận ngắn gọn từ scorecard chọn mô hình cuối cùng."""
    if scorecard is None or scorecard.empty:
        raise ValueError('scorecard rỗng, chưa có dữ liệu để kết luận.')

    best = scorecard.iloc[0]
    best_mc = scorecard.loc[scorecard['mc_RMSE_mean'].idxmin()]
    best_time = scorecard.loc[scorecard['sliding_RMSE_mean'].idxmin()]
    most_mc_stable = scorecard.loc[scorecard['mc_RMSE_cv_%'].idxmin()]
    most_time_stable = scorecard.loc[scorecard['sliding_RMSE_cv_%'].idxmin()]

    print(f"\n{'='*70}")
    print('KẾT LUẬN CHỌN MÔ HÌNH SAU PYMO O + STABILITY'.replace('PYMO O', 'PYMOO'))
    print(f"{'='*70}")
    print(
        f"Best overall: {best['case']} | final_score={best['final_score']:.4f} | "
        f"MC_RMSE={best['mc_RMSE_mean']:.4f} | "
        f"Sliding_RMSE={best['sliding_RMSE_mean']:.4f} | "
        f"MC_CV={best['mc_RMSE_cv_%']:.4f}% | "
        f"Sliding_CV={best['sliding_RMSE_cv_%']:.4f}%"
    )
    print(f"Params: {best['optimized_params']}")
    print(
        f"Best MC RMSE: {best_mc['case']} ({best_mc['mc_RMSE_mean']:.4f})"
    )
    print(
        f"Best time-sliding RMSE: {best_time['case']} "
        f"({best_time['sliding_RMSE_mean']:.4f})"
    )
    print(
        f"Ổn định nhất theo Monte Carlo CV: {most_mc_stable['case']} "
        f"({most_mc_stable['mc_RMSE_cv_%']:.4f}%)"
    )
    print(
        f"Ổn định nhất theo time-sliding CV: {most_time_stable['case']} "
        f"({most_time_stable['sliding_RMSE_cv_%']:.4f}%)"
    )

    if bool(best.get('stable_overall', False)):
        print('Kết luận: mô hình được chọn vừa có sai số thấp vừa đạt ngưỡng ổn định CV.')
    else:
        print('Kết luận: mô hình được chọn có điểm tổng hợp tốt nhất nhưng chưa đạt đủ ngưỡng CV.')

    return best
