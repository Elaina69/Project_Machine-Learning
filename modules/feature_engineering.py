"""
Module Feature Engineering cho dữ liệu PeMSD3 time series.
Tạo lag features, rolling features, time features, và target variable.
"""
import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trích xuất features từ timestamp: hour, weekday, is_weekend, time_slot."""
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.dayofweek  # 0=Mon, 6=Sun
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)

    # Time slot: chia ngày thành các khung giờ
    conditions = [
        (df['hour'] >= 0) & (df['hour'] < 6),
        (df['hour'] >= 6) & (df['hour'] < 10),
        (df['hour'] >= 10) & (df['hour'] < 15),
        (df['hour'] >= 15) & (df['hour'] < 19),
        (df['hour'] >= 19) & (df['hour'] < 24),
    ]
    choices = [0, 1, 2, 3, 4]  # đêm, sáng, trưa, chiều, tối
    df['time_slot'] = np.select(conditions, choices, default=0)

    print("✅ Đã thêm time features: hour, weekday, is_weekend, time_slot")
    return df


def add_lag_features(df: pd.DataFrame, lag_steps: list = None,
                     columns: list = None) -> pd.DataFrame:
    """
    Tạo lag features, group theo sensor_id (bắt buộc).

    Args:
        lag_steps: [1, 2, 3] tương ứng t-5min, t-10min, t-15min
        columns: ['flow', 'speed', 'occupancy']
    """
    if lag_steps is None:
        lag_steps = [1, 2, 3]
    if columns is None:
        columns = ['flow', 'speed', 'occupancy']

    df = df.copy()
    for col in columns:
        for lag in lag_steps:
            df[f"{col}_lag_{lag}"] = df.groupby('sensor_id')[col].shift(lag)

    n = len(columns) * len(lag_steps)
    print(f"✅ Đã thêm {n} lag features (lags={lag_steps}, cols={columns})")
    return df


def add_rolling_features(df: pd.DataFrame, windows: list = None,
                         columns: list = None) -> pd.DataFrame:
    """
    Tạo rolling mean & std features, group theo sensor_id.

    Args:
        windows: [3, 6, 12] tương ứng 15min, 30min, 1h
        columns: ['flow', 'speed']
    """
    if windows is None:
        windows = [3, 6, 12]
    if columns is None:
        columns = ['flow', 'speed']

    df = df.copy()
    for col in columns:
        for w in windows:
            df[f"{col}_roll_mean_{w}"] = (
                df.groupby('sensor_id')[col]
                .transform(lambda x: x.rolling(window=w, min_periods=1).mean())
            )
            df[f"{col}_roll_std_{w}"] = (
                df.groupby('sensor_id')[col]
                .transform(lambda x: x.rolling(window=w, min_periods=1).std())
            )

    n = len(columns) * len(windows) * 2
    print(f"✅ Đã thêm {n} rolling features (windows={windows})")
    return df


def add_target(df: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """Tạo target variable: flow tại t+horizon (mặc định 15 phút)."""
    df = df.copy()
    df['flow_target'] = df.groupby('sensor_id')['flow'].shift(-horizon)
    print(f"✅ Đã tạo target: flow_target (t+{horizon} = {horizon*5} phút)")
    return df


def prepare_all_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Pipeline đầy đủ: time features → lag → rolling → target → drop NaN.

    Args:
        config: dict chứa lag_steps, rolling_windows, rolling_columns,
                lag_columns, target_horizon
    """
    df = add_time_features(df)
    df = add_lag_features(
        df,
        lag_steps=config.get('lag_steps', [1, 2, 3]),
        columns=config.get('lag_columns', ['flow', 'speed', 'occupancy']),
    )
    df = add_rolling_features(
        df,
        windows=config.get('rolling_windows', [3, 6, 12]),
        columns=config.get('rolling_columns', ['flow', 'speed']),
    )
    df = add_target(df, horizon=config.get('target_horizon', 3))

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    print(f"✅ Drop NaN: {before:,} → {after:,} dòng (mất {before - after:,})")
    return df


def get_feature_columns(config: dict) -> list:
    """Trả về danh sách tên cột feature dùng cho modeling."""
    lag_steps = config.get('lag_steps', [1, 2, 3])
    lag_cols = config.get('lag_columns', ['flow', 'speed', 'occupancy'])
    roll_windows = config.get('rolling_windows', [3, 6, 12])
    roll_cols = config.get('rolling_columns', ['flow', 'speed'])

    features = []
    # Lag features
    for col in lag_cols:
        for lag in lag_steps:
            features.append(f"{col}_lag_{lag}")
    # Rolling features
    for col in roll_cols:
        for w in roll_windows:
            features.append(f"{col}_roll_mean_{w}")
            features.append(f"{col}_roll_std_{w}")
    # Time features
    features += ['hour', 'weekday', 'is_weekend']

    return features


def holdout_split(df: pd.DataFrame, train_ratio: float = 0.70,
                  valid_ratio: float = 0.15):
    """
    Hold-out split theo thời gian (KHÔNG shuffle) cho mỗi sensor.

    Returns:
        (df_train, df_valid, df_test)
    """
    dfs_train, dfs_valid, dfs_test = [], [], []

    for sid, group in df.groupby('sensor_id'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        n = len(group)
        train_end = int(n * train_ratio)
        valid_end = int(n * (train_ratio + valid_ratio))

        dfs_train.append(group.iloc[:train_end])
        dfs_valid.append(group.iloc[train_end:valid_end])
        dfs_test.append(group.iloc[valid_end:])

    df_train = pd.concat(dfs_train, ignore_index=True)
    df_valid = pd.concat(dfs_valid, ignore_index=True)
    df_test = pd.concat(dfs_test, ignore_index=True)

    print(f"✅ Hold-out Split (theo thời gian, per sensor):")
    print(f"   Train : {len(df_train):,} dòng ({train_ratio*100:.0f}%)")
    print(f"   Valid : {len(df_valid):,} dòng ({valid_ratio*100:.0f}%)")
    print(f"   Test  : {len(df_test):,} dòng ({(1-train_ratio-valid_ratio)*100:.0f}%)")
    return df_train, df_valid, df_test
