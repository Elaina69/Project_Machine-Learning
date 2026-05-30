CONFIG = {
    # ── Đường dẫn ──
    'data_path': '__datasets-raw/SV16_PeMSD3_sample_8sensors.csv',
    'clean_dir': '__datasets-clean',
    'models_dir': 'models',
    'images_dir': 'resultImages',

    # ── Baselines ──
    'baseline_a_sensors': ['PEMSD3_007', 'PEMSD3_008', 'PEMSD3_009', 'PEMSD3_010'],
    'baseline_b_sensors': ['PEMSD3_011', 'PEMSD3_012', 'PEMSD3_013', 'PEMSD3_014'],

    # ── Chia dữ liệu (Hold-out Split theo thời gian) ──
    'train_ratio': 0.70,
    'valid_ratio': 0.15,
    # test_ratio = 1 - train_ratio - valid_ratio = 0.15

    # ── Feature Engineering ──
    'lag_steps': [1, 2, 3, 6, 12],    # t-5min, t-10min, t-15min, t-30min, t-1h
    'lag_columns': ['flow', 'speed', 'occupancy'],
    'rolling_windows': [3, 6, 12],    # 15min, 30min, 1h
    'rolling_columns': ['flow', 'speed', 'occupancy'],
    'target_horizon': 3,              # Dự báo flow sau 3 bước = 15 phút

    # ── Modeling ──
    'random_state': 42,
    'model_params': {
        'linear_regression': {},
        'ridge': {'alpha': 1.0},
        'knn': {'n_neighbors': 10, 'weights': 'distance'},
        'decision_tree': {'max_depth': 15},
        'random_forest': {'n_estimators': 100, 'max_depth': 15},
        'xgboost': {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1},
        'lstm': {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'lr': 0.001,
            'epochs': 100,
            'batch_size': 256,
            'patience': 8,
        },
    },
}

OPTIM_CONFIG = {
    'pop_size': 300,        # Kích thước quần thể khuyến nghị: 200-400
    'n_gen': 100,           # Số thế hệ khuyến nghị: 100-200
    'algorithm': 'nsga2',   # 'nsga2' hoặc 'nsga3'
    'random_state': 69,
}

STABILITY_CONFIG = {
    # Monte Carlo: giữ nguyên split thời gian, chỉ đổi random_state thuật toán
    'monte_carlo_runs': 30,
    'monte_carlo_seed_start': 1000,

    # Time sliding validation: Train 60%, Test 10%, bước trượt 5%, 5 fold
    'sliding_train_ratio': 0.60,
    'sliding_test_ratio': 0.10,
    'sliding_step_ratio': 0.05,
    'sliding_n_folds': 5,
    'sliding_random_state': 42,
    'stable_cv_threshold': 10.0,
}
