"""Module Optimization — Tối ưu hóa đa mục tiêu siêu tham số bằng Pymoo.

Sử dụng NSGA-II/NSGA-III để tối ưu 3 mục tiêu:
    1. RMSE (minimize) — Tiêu chí chính đánh giá chất lượng dự báo
    2. Search time (minimize) — Thời gian huấn luyện/tìm kiếm tham số
    3. Model complexity (minimize) — Độ phức tạp của mô hình

Tối ưu hiệu năng:
    - Song song hóa đánh giá cá thể bằng joblib.Parallel
    - Subsampling dữ liệu train để giảm thời gian fit mỗi cá thể
    - Thu hẹp search space hợp lý cho RandomForest
"""
import numpy as np
import pandas as pd
import time
import os
import copy
from pathlib import Path

try:
    import dill as _serializer   # dill xử lý tốt lambda/closure
except ImportError:
    import pickle as _serializer  # fallback nếu chưa cài dill

from joblib import Parallel, delayed

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.callback import Callback

import matplotlib.pyplot as plt


from configs.pymooSearchSpaces import SEARCH_SPACES


# ─── Pymoo Problem ───────────────────────────────────────────────────────

class HyperparamOptProblem(Problem):
    """Bài toán tối ưu đa mục tiêu cho hyperparameter tuning.

    3 objectives (ALL minimize):
        f1 = RMSE trên validation set
        f2 = Thời gian huấn luyện (giây)
        f3 = Độ phức tạp mô hình (model-specific)

    Tối ưu hiệu năng:
        - train_subsample_ratio: tỷ lệ lấy mẫu dữ liệu train (0.0-1.0)
          Giá trị < 1.0 giúp mỗi lần fit nhanh hơn đáng kể.
        - n_parallel_jobs: số luồng song song cho evaluate (-1 = tất cả CPU)
    """
    def __init__(self, model_name, model_class, X_train, y_train,
                 X_valid, y_valid, random_state=42,
                 train_subsample_ratio=1.0, n_parallel_jobs=-1, **kwargs):
        self.model_name = model_name
        self.model_class = model_class
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.random_state = random_state
        self.train_subsample_ratio = train_subsample_ratio
        self.n_parallel_jobs = n_parallel_jobs

        space = SEARCH_SPACES[model_name]
        self.param_defs = space['params']
        self.complexity_fn = space['complexity_fn']

        n_var = len(self.param_defs)
        xl = np.array([p[1] for p in self.param_defs], dtype=float)
        xu = np.array([p[2] for p in self.param_defs], dtype=float)

        super().__init__(n_var=n_var, n_obj=3, n_constr=0,
                         xl=xl, xu=xu, **kwargs)

        self.eval_history = []  # lưu lại mỗi lần evaluate
        self._current_gen = 0   # thế hệ hiện tại (được cập nhật bởi callback)

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

    def _get_subsample(self):
        """Lấy mẫu con của dữ liệu train để tăng tốc evaluate."""
        if self.train_subsample_ratio >= 1.0:
            return self.X_train, self.y_train

        rng = np.random.RandomState(self.random_state + self._current_gen)
        n = len(self.X_train)
        n_sub = max(100, int(n * self.train_subsample_ratio))  # ít nhất 100 mẫu
        idx = rng.choice(n, size=n_sub, replace=False)
        return self.X_train[idx], self.y_train[idx]

    def _eval_single(self, x, X_sub, y_sub):
        """Đánh giá 1 cá thể (1 bộ siêu tham số). Dùng cho song song hóa."""
        params, decoded_vals = self._decode_params(x)

        try:
            model = self._build_model(params)

            t0 = time.time()
            model.fit(X_sub, y_sub)
            train_time = time.time() - t0

            y_pred = model.predict(self.X_valid)
            rmse = np.sqrt(np.mean((self.y_valid - y_pred) ** 2))
            complexity = self.complexity_fn(decoded_vals)

            return {
                'f': [rmse, train_time, complexity],
                'history': {
                    'params': params,
                    'rmse': rmse,
                    'train_time': train_time,
                    'complexity': complexity,
                    'n_gen': self._current_gen,
                },
            }
        except Exception:
            return {
                'f': [1e6, 1e6, 1e6],
                'history': None,
            }

    def _evaluate(self, X, out, *args, **kwargs):
        X_sub, y_sub = self._get_subsample()

        # Song song hóa: đánh giá nhiều cá thể cùng lúc
        # Lưu ý: RandomForest/XGBoost đã dùng n_jobs=-1 bên trong,
        # nên ở đây dùng n_jobs vừa phải (mặc định -1 → joblib tự điều chỉnh)
        results = Parallel(
            n_jobs=self.n_parallel_jobs,
            prefer='threads',   # threads chia sẻ bộ nhớ, tránh copy dữ liệu
            verbose=0,
        )(
            delayed(self._eval_single)(x, X_sub, y_sub)
            for x in X
        )

        F = np.zeros((len(results), 3))
        for i, res in enumerate(results):
            F[i] = res['f']
            if res['history'] is not None:
                self.eval_history.append(res['history'])

        out["F"] = F

    def _build_model(self, params):
        """Tạo model instance từ params dict.
        Lưu ý: n_jobs=1 cho mỗi model vì song song hóa đã ở cấp evaluate.
        """
        from sklearn.linear_model import Ridge
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor

        if self.model_name == '5_RandomForest':
            return RandomForestRegressor(
                **params, random_state=self.random_state, n_jobs=1
            )
        elif self.model_name == '6_XGBoost':
            from xgboost import XGBRegressor
            return XGBRegressor(
                **params, random_state=self.random_state,
                n_jobs=1, verbosity=0
            )
        elif self.model_name == '4_DecisionTree':
            return DecisionTreeRegressor(
                **params, random_state=self.random_state
            )
        elif self.model_name == '2_Ridge':
            return Ridge(**params)
        elif self.model_name == '3_KNN':
            return KNeighborsRegressor(**params, weights='distance', n_jobs=1)
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


# ─── Checkpoint helpers ───────────────────────────────────────────────────

def _get_checkpoint_path(checkpoint_dir, model_name):
    """Tạo đường dẫn file checkpoint cho 1 model."""
    safe_name = model_name.replace('/', '_').replace(' ', '_')
    return Path(checkpoint_dir) / f"optim_checkpoint_{safe_name}.pkl"


def _save_checkpoint(path, algorithm, eval_history, elapsed_time):
    """Lưu trạng thái tối ưu hóa ra file."""
    checkpoint = {
        'algorithm': algorithm,
        'eval_history': eval_history,
        'elapsed_time': elapsed_time,
        'n_gen_completed': algorithm.n_gen,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ghi vào file tạm trước rồi rename — tránh corrupt nếu bị ngắt giữa chừng
    tmp_path = path.with_suffix('.tmp')
    with open(tmp_path, 'wb') as f:
        _serializer.dump(checkpoint, f)
    # Atomic rename (Windows: cần xóa file cũ trước)
    if path.exists():
        path.unlink()
    tmp_path.rename(path)


def _load_checkpoint(path):
    """Tải trạng thái tối ưu hóa từ file. Trả về None nếu không có."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, 'rb') as f:
            return _serializer.load(f)
    except Exception as e:
        print(f"⚠️  Không thể đọc checkpoint ({e}), bắt đầu lại từ đầu.")
        return None


# ─── Run Optimization (với checkpoint/resume) ─────────────────────────────

def run_optimization(model_name, X_train, y_train, X_valid, y_valid,
                     pop_size=200, n_gen=100, algorithm='nsga2',
                     random_state=42, verbose=True,
                     train_subsample_ratio=0.5, n_parallel_jobs=-1,
                     checkpoint_dir='pymooCheckpoint', checkpoint_interval=1):
    """Chạy tối ưu hóa đa mục tiêu cho 1 mô hình, hỗ trợ lưu/khôi phục tiến trình.

    Args:
        model_name: Tên mô hình (key trong SEARCH_SPACES)
        pop_size: Kích thước quần thể (200-400)
        n_gen: Số thế hệ tổng cộng mong muốn
        algorithm: 'nsga2' hoặc 'nsga3'
        train_subsample_ratio: Tỷ lệ lấy mẫu dữ liệu train (0.0-1.0).
        n_parallel_jobs: Số luồng song song (-1 = tự động theo CPU).
        checkpoint_dir: Thư mục lưu checkpoint (mặc định: 'pymooCheckpoint').
            Nếu đã có checkpoint, tự động resume từ thế hệ đã lưu.
            Đặt None để tắt checkpoint.
        checkpoint_interval: Lưu checkpoint mỗi N thế hệ (mặc định: 1 = mỗi thế hệ).

    Returns:
        result: Pymoo Result object
        pareto_df: DataFrame chứa Pareto front
        problem: HyperparamOptProblem instance
    """
    if model_name not in SEARCH_SPACES:
        raise ValueError(f"Model '{model_name}' không có search space. "
                         f"Hỗ trợ: {list(SEARCH_SPACES.keys())}")

    from sklearn.ensemble import RandomForestRegressor
    model_class = None  # sẽ build trong Problem

    # Tự động xác định số jobs nếu -1
    if n_parallel_jobs == -1:
        n_parallel_jobs = max(1, os.cpu_count() - 1)

    # ── Thử resume từ checkpoint ──
    resumed = False
    resumed_gen = 0
    elapsed_before = 0.0
    checkpoint_path = None
    algo = None
    problem = None

    if checkpoint_dir is not None:
        checkpoint_path = _get_checkpoint_path(checkpoint_dir, model_name)
        ckpt = _load_checkpoint(checkpoint_path)
        if ckpt is not None:
            resumed_gen = ckpt['n_gen_completed']
            if resumed_gen >= n_gen:
                print(f"\n✅ Checkpoint cho {model_name} đã hoàn tất "
                      f"{resumed_gen}/{n_gen} thế hệ. Không cần chạy lại.")
                # Trả về kết quả từ algorithm đã lưu
                algo_saved = ckpt['algorithm']
                problem_saved = algo_saved.problem
                problem_saved.eval_history = ckpt['eval_history']
                # Xây dựng pareto_df từ kết quả cuối
                return _build_result_from_algorithm(
                    algo_saved, problem_saved, ckpt['elapsed_time']
                )
            else:
                print(f"\n🔄 RESUME từ checkpoint: {resumed_gen}/{n_gen} thế hệ đã xong.")
                algo = ckpt['algorithm']
                problem = algo.problem
                problem.eval_history = ckpt['eval_history']
                elapsed_before = ckpt['elapsed_time']
                # Cập nhật dữ liệu train/valid (có thể khác nếu user đổi data)
                problem.X_train = X_train
                problem.y_train = y_train
                problem.X_valid = X_valid
                problem.y_valid = y_valid
                problem.n_parallel_jobs = n_parallel_jobs
                problem.train_subsample_ratio = train_subsample_ratio
                resumed = True

    # ── Tạo mới nếu không resume ──
    if not resumed:
        problem = HyperparamOptProblem(
            model_name=model_name,
            model_class=model_class,
            X_train=X_train, y_train=y_train,
            X_valid=X_valid, y_valid=y_valid,
            random_state=random_state,
            train_subsample_ratio=train_subsample_ratio,
            n_parallel_jobs=n_parallel_jobs,
        )

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

    # ── Callback: theo dõi gen + lưu checkpoint ──
    class CheckpointCallback(Callback):
        def __init__(self, ckpt_path, interval, elapsed_before):
            super().__init__()
            self._ckpt_path = ckpt_path
            self._interval = interval
            self._t_start = time.time()
            self._elapsed_before = elapsed_before

        def notify(self, algorithm):
            # Cập nhật gen hiện tại trên problem
            algorithm.problem._current_gen = algorithm.n_gen

            # Lưu checkpoint theo interval
            if self._ckpt_path is not None and algorithm.n_gen % self._interval == 0:
                elapsed = self._elapsed_before + (time.time() - self._t_start)
                _save_checkpoint(
                    self._ckpt_path,
                    algorithm,
                    algorithm.problem.eval_history,
                    elapsed,
                )
                print(f"   💾 Checkpoint saved: gen {algorithm.n_gen}/{n_gen} "
                      f"({elapsed:.0f}s tổng)")


    # ── In thông tin ──
    print(f"\n{'='*60}")
    print(f"🔍 TỐI ƯU HÓA {model_name}")
    print(f"   Thuật toán: {algorithm.upper() if isinstance(algorithm, str) else type(algo).__name__}")
    print(f"   Quần thể: {pop_size} | Thế hệ: {n_gen} "
          f"({'resume ' + str(resumed_gen) + '→' + str(n_gen) if resumed else 'từ đầu'})")
    print(f"   Biến quyết định: {len(problem.param_defs)}")
    print(f"   Mục tiêu: RMSE ↓, Train time ↓, Complexity ↓")
    print(f"   Subsample ratio: {train_subsample_ratio:.0%} "
          f"({int(len(X_train)*train_subsample_ratio)}/{len(X_train)} mẫu)")
    print(f"   Song song: {n_parallel_jobs} jobs")
    if checkpoint_path:
        print(f"   Checkpoint: {checkpoint_path} (mỗi {checkpoint_interval} gen)")
    print(f"{'='*60}")

    # ── Chạy tối ưu ──
    # Lưu ý quan trọng: khi resume, algorithm đã có n_gen = resumed_gen.
    # pymoo_minimize với ('n_gen', N) sẽ chạy cho đến khi algorithm.n_gen >= N.
    # Vì vậy luôn truyền n_gen TỔNG CỘNG, không phải remaining.
    t_start = time.time()
    result = pymoo_minimize(
        problem, algo,
        ('n_gen', n_gen),
        seed=random_state,
        verbose=verbose,
        callback=CheckpointCallback(checkpoint_path, checkpoint_interval, elapsed_before),
    )
    total_time = elapsed_before + (time.time() - t_start)

    # Lưu checkpoint cuối cùng
    if checkpoint_path is not None:
        _save_checkpoint(checkpoint_path, result.algorithm, problem.eval_history, total_time)
        print(f"   💾 Checkpoint cuối cùng đã lưu ({total_time:.0f}s tổng).")

    return _build_result_from_algorithm(result.algorithm, problem, total_time)


def _build_result_from_algorithm(algo, problem, total_time):
    """Xây dựng pareto_df từ algorithm đã chạy xong."""
    # Lấy kết quả tốt nhất từ quần thể cuối
    pop = algo.pop if hasattr(algo, 'pop') else algo.result().pop
    # Dùng opt (non-dominated) nếu có
    opt = algo.opt if hasattr(algo, 'opt') else pop

    pareto_F = np.array([ind.F for ind in opt])
    pareto_X = np.array([ind.X for ind in opt])

    def _find_gen_for_solution(params_dict, eval_history):
        best_gen = 0
        for entry in eval_history:
            if entry['params'] == params_dict:
                best_gen = entry['n_gen']
        return best_gen

    pareto_records = []
    for i in range(len(pareto_F)):
        params, _ = problem._decode_params(pareto_X[i])
        gen = _find_gen_for_solution(params, problem.eval_history)
        pareto_records.append({
            'n_gen': gen,
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

    # Tạo result-like object đơn giản
    class _SimpleResult:
        def __init__(self, F, X, algorithm):
            self.F = F
            self.X = X
            self.algorithm = algorithm
    result = _SimpleResult(pareto_F, pareto_X, algo)

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

    # Các cột siêu tham số (loại bỏ cột mục tiêu và n_gen)
    objective_cols = {'RMSE', 'train_time_s', 'complexity', 'n_gen'}
    param_cols = [c for c in pareto_df.columns if c not in objective_cols]

    # Nghiệm tốt nhất theo từng tiêu chí
    best_rmse = pareto_df.loc[pareto_df['RMSE'].idxmin()]
    best_time = pareto_df.loc[pareto_df['train_time_s'].idxmin()]
    best_complex = pareto_df.loc[pareto_df['complexity'].idxmin()]

    # --- Helper: format thông tin nghiệm kèm siêu tham số ---
    def _format_solution(label, solution):
        """Trả về danh sách các dòng mô tả nghiệm kèm siêu tham số."""
        result_lines = []
        result_lines.append(label)
        # Mục tiêu
        gen_info = f", n_gen={int(solution['n_gen'])}" if 'n_gen' in solution.index else ""
        result_lines.append(
            f"   RMSE={solution['RMSE']:.4f}, "
            f"Time={solution['train_time_s']:.3f}s, "
            f"Complexity={solution['complexity']:.0f}{gen_info}"
        )
        # Siêu tham số
        if param_cols:
            params_str = ", ".join(
                f"{c}={solution[c]:.6g}" if isinstance(solution[c], float)
                else f"{c}={int(solution[c])}"
                for c in param_cols
            )
            result_lines.append(f"   Params: {{{params_str}}}")
        return result_lines

    lines.extend(_format_solution(f"\n🎯 Nghiệm tốt nhất theo RMSE:", best_rmse))
    lines.extend(_format_solution(f"\n⚡ Nghiệm nhanh nhất:", best_time))
    lines.extend(_format_solution(f"\n🧩 Nghiệm đơn giản nhất:", best_complex))

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
    lines.extend(_format_solution(f"\n🔑 Nghiệm cân bằng (compromise solution):", best_compromise))

    conclusion = "\n".join(lines)
    print(conclusion)
    return conclusion, best_compromise
