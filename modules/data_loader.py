"""
Module tải và xử lý dữ liệu PeMSD3.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load CSV gốc, parse timestamp, sắp xếp theo sensor + thời gian."""
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['sensor_id', 'timestamp']).reset_index(drop=True)
    print(f"✅ Đã load {len(df):,} dòng từ {Path(filepath).name}")
    print(f"   Sensors: {sorted(df['sensor_id'].unique())}")
    print(f"   Thời gian: {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


def get_data_info(df: pd.DataFrame) -> dict:
    """Trả về dict thông tin tổng quan dataset."""
    return {
        'total_rows': len(df),
        'columns': list(df.columns),
        'n_sensors': df['sensor_id'].nunique(),
        'sensors': sorted(df['sensor_id'].unique().tolist()),
        'date_range': (str(df['timestamp'].min()), str(df['timestamp'].max())),
        'rows_per_sensor': df.groupby('sensor_id').size().to_dict(),
        'missing_per_column': df.isnull().sum().to_dict(),
        'missing_total': int(df.isnull().sum().sum()),
    }


def print_data_info(info: dict):
    """In thông tin dataset ra console."""
    print("=" * 60)
    print("📊 THÔNG TIN DATASET")
    print("=" * 60)
    print(f"  Tổng số dòng   : {info['total_rows']:,}")
    print(f"  Số cột          : {len(info['columns'])}")
    print(f"  Số sensor       : {info['n_sensors']}")
    print(f"  Khoảng thời gian: {info['date_range'][0]} → {info['date_range'][1]}")
    print(f"  Missing values  : {info['missing_total']}")
    print()
    print("  Số dòng mỗi sensor:")
    for sid, cnt in sorted(info['rows_per_sensor'].items()):
        print(f"    {sid}: {cnt:,} dòng")
    print("=" * 60)


def split_baselines(df: pd.DataFrame, sensors_a: list, sensors_b: list):
    """Chia dữ liệu thành 2 baselines theo danh sách sensor."""
    df_a = df[df['sensor_id'].isin(sensors_a)].copy().reset_index(drop=True)
    df_b = df[df['sensor_id'].isin(sensors_b)].copy().reset_index(drop=True)
    print(f"✅ Chia Baselines:")
    print(f"   Baseline A: {sensors_a} → {len(df_a):,} dòng")
    print(f"   Baseline B: {sensors_b} → {len(df_b):,} dòng")
    return df_a, df_b
