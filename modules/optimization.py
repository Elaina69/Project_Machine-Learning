"""
Module Optimization — Tối ưu hóa đa mục tiêu siêu tham số bằng Pymoo.

Sử dụng NSGA-II/NSGA-III để tối ưu 3 mục tiêu:
    1. RMSE (minimize) — Tiêu chí chính đánh giá chất lượng dự báo
    2. Search time (minimize) — Thời gian huấn luyện/tìm kiếm tham số
    3. Model complexity (minimize) — Độ phức tạp của mô hình
"""
import numpy as np
import pandas as pd
import time
from pathlib import Path

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

import matplotlib.pyplot as plt


# ─── Model-specific Search Spaces ────────────────────────────────────────

SEARCH_SPACES = {
    '5_RandomForest': {
        'params': [
            ('n_estimators', 50, 500, int),      # x[0]
            ('max_depth', 3, 30, int),            # x[1]
            ('min_samples_split', 2, 20, int),    # x[2]
            ('min_samples_leaf', 1, 10, int),     # x[3]
        ],
        'complexity_fn': lambda params: params[0] * params[1],  # n_trees × depth
    },
    '6_XGBoost': {
        'params': [
            ('n_estimators', 50, 500, int),       # x[0]
            ('max_depth', 3, 15, int),            # x[1]
            ('learning_rate', 0.001, 0.3, float), # x[2]
            ('subsample', 0.5, 1.0, float),       # x[3]
            ('colsample_bytree', 0.5, 1.0, float),# x[4]
            ('reg_alpha', 0.0, 10.0, float),      # x[5]
            ('reg_lambda', 0.0, 10.0, float),     # x[6]
        ],
        'complexity_fn': lambda params: params[0] * params[1],
    },
    '4_DecisionTree': {
        'params': [
            ('max_depth', 3, 30, int),            # x[0]
            ('min_samples_split', 2, 30, int),    # x[1]
            ('min_samples_leaf', 1, 20, int),     # x[2]
        ],
        'complexity_fn': lambda params: params[0],
    },
    '2_Ridge': {
        'params': [
            ('alpha', 0.001, 100.0, float),       # x[0]
        ],
        'complexity_fn': lambda params: 1,  # Ridge luôn O(n_features)
    },
    '3_KNN': {
        'params': [
            ('n_neighbors', 3, 50, int),          # x[0]
        ],
        'complexity_fn': lambda params: params[0],
    },
}


# ─── Pymoo Problem ───────────────────────────────────────────────────────

class HyperparamOptProblem(Problem):
    """
    Bài toán tối ưu đa mục tiêu cho hyperparameter tuning.

    3 objectives (ALL minimize):
        f1 = RMSE trên validation set
        f2 = Thời gian huấn luyện (giây)
        f3 = Độ phức tạp mô hình (model-specific)
    """
    def __init__(self, model_name, model_class, X_train, y_train,
                 X_valid, y_valid, random_state=42, **kwargs):
        self.model_name = model_name
        self.model_class = model_class
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.random_state = random_state

        space = SEARCH_SPACES[model_name]
        self.param_defs = space['params']
        self.complexity_fn = space['complexity_fn']

        n_var = len(self.param_defs)
        xl = np.array([p[1] for p in self.param_defs], dtype=float)
        xu = np.array([p[2] for p in self.param_defs], dtype=float)

        super().__init__(n_var=n_var, n_obj=3, n_constr=0,
                         xl=xl, xu=xu, **kwargs)

        self.eval_history = []  # lưu lại mỗi lần evaluate

    def _decode_params(self, x):
        """Chuyển vector liên tục thành dict params."""
        params = {}
        decoded_vals = []
        for i, (name, lo, hi, dtype) in enumerate(self.param_defs):
            val = x[i]
            if dtype == int:
                val = int(round(val))
                val = max(lo, min(hi, val))
            params[name] = val
            decoded_vals.append(val)
        return params, decoded_vals

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((X.shape[0], 3))

        for i, x in enumerate(X):
            params, decoded_vals = self._decode_params(x)

            try:
                # Tạo model instance
                model = self._build_model(params)

                # Đo thời gian huấn luyện
                t0 = time.time()
                model.fit(self.X_train, self.y_train)
                train_time = time.time() - t0

                # Đánh giá trên validation
                y_pred = model.predict(self.X_valid)
                rmse = np.sqrt(np.mean((self.y_valid - y_pred) ** 2))

                # Tính complexity
                complexity = self.complexity_fn(decoded_vals)

                F[i, 0] = rmse
                F[i, 1] = train_time
                F[i, 2] = complexity

                self.eval_history.append({
                    'params': params,
                    'rmse': rmse,
                    'train_time': train_time,
                    'complexity': complexity,
                })

            except Exception as e:
                # Penalize failed evaluations
                F[i, 0] = 1e6
                F[i, 1] = 1e6
                F[i, 2] = 1e6

        out["F"] = F

    def _build_model(self, params):
        """Tạo model instance từ params dict."""
        from sklearn.linear_model import Ridge
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor

        if self.model_name == '5_RandomForest':
            return RandomForestRegressor(
                **params, random_state=self.random_state, n_jobs=-1
            )
        elif self.model_name == '6_XGBoost':
            from xgboost import XGBRegressor
            return XGBRegressor(
                **params, random_state=self.random_state,
                n_jobs=-1, verbosity=0
            )
        elif self.model_name == '4_DecisionTree':
            return DecisionTreeRegressor(
                **params, random_state=self.random_state
            )
        elif self.model_name == '2_Ridge':
            return Ridge(**params)
        elif self.model_name == '3_KNN':
            return KNeighborsRegressor(**params, weights='distance', n_jobs=-1)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")


# ─── Run Optimization ─────────────────────────────────────────────────────

def select_best_models(results_a, results_b, top_n=2):
    """
    Chọn top_n mô hình tốt nhất trên CẢ HAI baselines.
    Tiêu chí: RMSE trung bình thấp nhất giữa A và B.
    Chỉ xét các mô hình ML (không phải trivial 0a/0b/0c).
    """
    # Filter out trivial baselines
    ml_models_a = results_a[~results_a['model'].str.startswith('0')].copy()
    ml_models_b = results_b[~results_b['model'].str.startswith('0')].copy()

    merged = ml_models_a[['model', 'test_RMSE']].merge(
        ml_models_b[['model', 'test_RMSE']],
        on='model', suffixes=('_A', '_B')
    )
    merged['avg_RMSE'] = (merged['test_RMSE_A'] + merged['test_RMSE_B']) / 2
    merged = merged.sort_values('avg_RMSE').head(top_n)

    selected = merged['model'].tolist()
    print(f"🏆 Top {top_n} mô hình tốt nhất trên CẢ HAI baselines:")
    for i, row in merged.iterrows():
        print(f"   {row['model']}: A_RMSE={row['test_RMSE_A']:.4f}, "
              f"B_RMSE={row['test_RMSE_B']:.4f}, Avg={row['avg_RMSE']:.4f}")
    return selected


def run_optimization(model_name, X_train, y_train, X_valid, y_valid,
                     pop_size=200, n_gen=100, algorithm='nsga2',
                     random_state=42, verbose=True):
    """
    Chạy tối ưu hóa đa mục tiêu cho 1 mô hình.

    Args:
        model_name: Tên mô hình (key trong SEARCH_SPACES)
        pop_size: Kích thước quần thể (200-400)
        n_gen: Số thế hệ (100-200)
        algorithm: 'nsga2' hoặc 'nsga3'

    Returns:
        result: Pymoo Result object
        pareto_df: DataFrame chứa Pareto front
    """
    if model_name not in SEARCH_SPACES:
        raise ValueError(f"Model '{model_name}' không có search space. "
                         f"Hỗ trợ: {list(SEARCH_SPACES.keys())}")

    from sklearn.ensemble import RandomForestRegressor
    model_class = None  # sẽ build trong Problem

    problem = HyperparamOptProblem(
        model_name=model_name,
        model_class=model_class,
        X_train=X_train, y_train=y_train,
        X_valid=X_valid, y_valid=y_valid,
        random_state=random_state,
    )

    # Chọn thuật toán
    if algorithm == 'nsga3':
        n_obj = 3
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
        algo = NSGA3(
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
        )
    else:  # nsga2
        algo = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

    print(f"\n{'='*60}")
    print(f"🔍 TỐI ƯU HÓA {model_name}")
    print(f"   Thuật toán: {algorithm.upper()}")
    print(f"   Quần thể: {pop_size} | Thế hệ: {n_gen}")
    print(f"   Biến quyết định: {len(problem.param_defs)}")
    print(f"   Mục tiêu: RMSE ↓, Train time ↓, Complexity ↓")
    print(f"{'='*60}")

    t_start = time.time()
    result = pymoo_minimize(
        problem, algo,
        ('n_gen', n_gen),
        seed=random_state,
        verbose=verbose,
    )
    total_time = time.time() - t_start

    # Xử lý Pareto front
    pareto_F = result.F  # (n_pareto, 3)
    pareto_X = result.X  # (n_pareto, n_vars)

    pareto_records = []
    for i in range(len(pareto_F)):
        params, _ = problem._decode_params(pareto_X[i])
        pareto_records.append({
            **params,
            'RMSE': round(pareto_F[i, 0], 4),
            'train_time_s': round(pareto_F[i, 1], 4),
            'complexity': round(pareto_F[i, 2], 2),
        })
    pareto_df = pd.DataFrame(pareto_records)
    pareto_df = pareto_df.sort_values('RMSE').reset_index(drop=True)

    print(f"\n✅ Tối ưu hóa hoàn tất trong {total_time:.1f}s")
    print(f"   Số nghiệm Pareto: {len(pareto_df)}")
    print(f"   RMSE range: [{pareto_df['RMSE'].min():.4f}, {pareto_df['RMSE'].max():.4f}]")
    print(f"   Time range: [{pareto_df['train_time_s'].min():.4f}s, "
          f"{pareto_df['train_time_s'].max():.4f}s]")

    return result, pareto_df, problem


# ─── Visualization ────────────────────────────────────────────────────────

def plot_pareto_front_3d(pareto_df, model_name, save_dir=None):
    """Vẽ Pareto front 3D: RMSE × Time × Complexity."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        pareto_df['RMSE'],
        pareto_df['train_time_s'],
        pareto_df['complexity'],
        c=pareto_df['RMSE'], cmap='viridis', s=50, alpha=0.8
    )

    ax.set_xlabel('RMSE ↓', fontsize=11)
    ax.set_ylabel('Train Time (s) ↓', fontsize=11)
    ax.set_zlabel('Complexity ↓', fontsize=11)
    ax.set_title(f'Pareto Front — {model_name}', fontsize=14)
    fig.colorbar(sc, label='RMSE', shrink=0.6)

    plt.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{save_dir}/pareto_3d_{model_name}.png",
                    bbox_inches='tight', dpi=150)
    plt.show()


def plot_pareto_2d_pairs(pareto_df, model_name, save_dir=None):
    """Vẽ 3 cặp scatter plots 2D cho Pareto front."""
    pairs = [
        ('RMSE', 'train_time_s', 'RMSE vs Train Time'),
        ('RMSE', 'complexity', 'RMSE vs Complexity'),
        ('train_time_s', 'complexity', 'Train Time vs Complexity'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (x_col, y_col, title) in zip(axes, pairs):
        ax.scatter(pareto_df[x_col], pareto_df[y_col],
                   c=pareto_df['RMSE'], cmap='viridis', s=40, alpha=0.8)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Phân tích đánh đổi Pareto — {model_name}', fontsize=14)
    plt.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{save_dir}/pareto_2d_{model_name}.png",
                    bbox_inches='tight', dpi=150)
    plt.show()


def analyze_tradeoffs(pareto_df, model_name):
    """Phân tích sự đánh đổi giữa 3 mục tiêu."""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"📊 PHÂN TÍCH ĐÁNH ĐỔI — {model_name}")
    lines.append(f"{'='*60}")
    lines.append(f"   Số nghiệm Pareto: {len(pareto_df)}")

    # Nghiệm tốt nhất theo từng tiêu chí
    best_rmse = pareto_df.loc[pareto_df['RMSE'].idxmin()]
    best_time = pareto_df.loc[pareto_df['train_time_s'].idxmin()]
    best_complex = pareto_df.loc[pareto_df['complexity'].idxmin()]

    lines.append(f"\n🎯 Nghiệm tốt nhất theo RMSE:")
    lines.append(f"   RMSE={best_rmse['RMSE']:.4f}, "
                 f"Time={best_rmse['train_time_s']:.3f}s, "
                 f"Complexity={best_rmse['complexity']:.0f}")

    lines.append(f"\n⚡ Nghiệm nhanh nhất:")
    lines.append(f"   RMSE={best_time['RMSE']:.4f}, "
                 f"Time={best_time['train_time_s']:.3f}s, "
                 f"Complexity={best_time['complexity']:.0f}")

    lines.append(f"\n🧩 Nghiệm đơn giản nhất:")
    lines.append(f"   RMSE={best_complex['RMSE']:.4f}, "
                 f"Time={best_complex['train_time_s']:.3f}s, "
                 f"Complexity={best_complex['complexity']:.0f}")

    # Phân tích trade-off
    rmse_range = pareto_df['RMSE'].max() - pareto_df['RMSE'].min()
    time_range = pareto_df['train_time_s'].max() - pareto_df['train_time_s'].min()
    complex_range = pareto_df['complexity'].max() - pareto_df['complexity'].min()

    lines.append(f"\n📈 Phạm vi đánh đổi:")
    lines.append(f"   RMSE: {pareto_df['RMSE'].min():.4f} → {pareto_df['RMSE'].max():.4f} "
                 f"(range={rmse_range:.4f})")
    lines.append(f"   Time: {pareto_df['train_time_s'].min():.3f}s → "
                 f"{pareto_df['train_time_s'].max():.3f}s "
                 f"(range={time_range:.3f}s)")
    lines.append(f"   Complexity: {pareto_df['complexity'].min():.0f} → "
                 f"{pareto_df['complexity'].max():.0f} "
                 f"(range={complex_range:.0f})")

    # Trade-off insight
    if rmse_range > 0 and time_range > 0:
        ratio = time_range / rmse_range
        lines.append(f"\n💡 Insight đánh đổi:")
        lines.append(f"   Giảm 1 đơn vị RMSE cần ~{ratio:.1f}s thêm thời gian huấn luyện.")

    # Nghiệm cân bằng (compromise) — TOPSIS-like
    lines.append(f"\n🔑 Nghiệm cân bằng (compromise solution):")
    norm_df = pareto_df[['RMSE', 'train_time_s', 'complexity']].copy()
    for col in norm_df.columns:
        r = norm_df[col].max() - norm_df[col].min()
        if r > 0:
            norm_df[col] = (norm_df[col] - norm_df[col].min()) / r
        else:
            norm_df[col] = 0
    # Distance to ideal (0,0,0)
    norm_df['dist'] = np.sqrt(norm_df['RMSE']**2 + norm_df['train_time_s']**2
                              + norm_df['complexity']**2)
    best_idx = norm_df['dist'].idxmin()
    best_compromise = pareto_df.loc[best_idx]
    lines.append(f"   RMSE={best_compromise['RMSE']:.4f}, "
                 f"Time={best_compromise['train_time_s']:.3f}s, "
                 f"Complexity={best_compromise['complexity']:.0f}")
    # Print params
    param_cols = [c for c in pareto_df.columns
                  if c not in ['RMSE', 'train_time_s', 'complexity']]
    if param_cols:
        lines.append(f"   Params: {dict(best_compromise[param_cols])}")

    conclusion = "\n".join(lines)
    print(conclusion)
    return conclusion, best_compromise
