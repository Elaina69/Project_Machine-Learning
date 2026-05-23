SEARCH_SPACES = {
    '5_RandomForest': {
        'params': [
            ('n_estimators', 50, 200, int),       # Số cây (giảm từ 500→200 để tăng tốc)
            ('max_depth', 3, 15, int),             # Độ sâu tối đa (giảm từ 30→15)
            ('min_samples_split', 2, 20, int),     # Mẫu tối thiểu để chia nút
            ('min_samples_leaf', 1, 10, int),      # Mẫu tối thiểu ở lá
        ],
        'complexity_fn': lambda params: params[0] * params[1],  # n_trees × depth
    },

    '6_XGBoost': {
        'params': [
            ('n_estimators', 50, 200, int),        # Số boosting rounds
            ('max_depth', 3, 15, int),             # Độ sâu tối đa mỗi cây
            ('learning_rate', 0.001, 0.3, float),  # Tốc độ học
            ('subsample', 0.5, 1.0, float),        # Tỷ lệ mẫu cho mỗi cây
            ('colsample_bytree', 0.5, 1.0, float), # Tỷ lệ feature cho mỗi cây
            ('reg_alpha', 0.0, 10.0, float),       # L1 regularization
            ('reg_lambda', 0.0, 10.0, float),      # L2 regularization
        ],
        'complexity_fn': lambda params: params[0] * params[1],  # n_trees × depth
    },

    '4_DecisionTree': {
        'params': [
            ('max_depth', 3, 30, int),             # Độ sâu tối đa
            ('min_samples_split', 2, 30, int),     # Mẫu tối thiểu để chia
            ('min_samples_leaf', 1, 20, int),      # Mẫu tối thiểu ở lá
        ],
        'complexity_fn': lambda params: params[0],  # depth
    },

    '2_Ridge': {
        'params': [
            ('alpha', 0.001, 100.0, float),        # Hệ số regularization
        ],
        'complexity_fn': lambda params: 1,  # Ridge luôn O(n_features)
    },

    '3_KNN': {
        'params': [
            ('n_neighbors', 3, 50, int),           # Số láng giềng
        ],
        'complexity_fn': lambda params: params[0],  # k
    },
}
