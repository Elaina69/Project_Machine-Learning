"""Phan tich loi du bao cho bai toan flow regression.

Module nay duoc goi sau khi da co ket qua hold-out cua cac baseline, truoc
buoc Pymoo. Muc tieu la chi ra cac truong hop du bao sai, phan nhom loi theo
sensor/hour/flow regime, va goi y nguyen nhan cung huong cai thien.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def build_error_dataframe(
        test_df: pd.DataFrame,
        y_true,
        y_pred,
        baseline_name: str,
        model_name: str,
    ) -> pd.DataFrame:
    """Tao bang residual-level cho tung mau test."""
    df = test_df.reset_index(drop=True).copy()
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(df) != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError('test_df, y_true va y_pred phai co cung so dong.')

    df['baseline'] = baseline_name
    df['model'] = model_name
    df['actual'] = y_true
    df['predicted'] = y_pred
    df['error'] = df['actual'] - df['predicted']
    df['abs_error'] = np.abs(df['error'])
    df['squared_error'] = df['error'] ** 2
    df['ape_%'] = np.where(
        df['actual'] != 0,
        df['abs_error'] / np.abs(df['actual']) * 100,
        np.nan,
    )
    df['bias_type'] = np.where(
        df['error'] > 0,
        'under_predict',
        np.where(df['error'] < 0, 'over_predict', 'exact'),
    )
    if 'flow_lag_1' in df.columns:
        df['delta_from_lag1'] = df['actual'] - df['flow_lag_1']
        df['abs_delta_from_lag1'] = np.abs(df['delta_from_lag1'])

    labels = ['low_flow', 'medium_flow', 'high_flow']
    try:
        df['flow_regime'] = pd.qcut(
            df['actual'],
            q=3,
            labels=labels,
            duplicates='drop',
        )
    except ValueError:
        df['flow_regime'] = 'unknown'

    return df


def combine_error_dataframes(error_frames: list) -> pd.DataFrame:
    """Ghep cac bang loi cua nhieu baseline/model."""
    return pd.concat(error_frames, ignore_index=True)


def summarize_error_segments(
        error_df: pd.DataFrame,
        group_cols=('baseline', 'model', 'sensor_id'),
    ) -> pd.DataFrame:
    """Tinh MAE/RMSE/MAPE/bias theo nhom."""
    group_cols = [col for col in group_cols if col in error_df.columns]
    if not group_cols:
        raise ValueError('Khong co cot group hop le de tong hop loi.')

    rows = []
    for keys, group in error_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        rec.update({
            'n': len(group),
            'MAE': group['abs_error'].mean(),
            'RMSE': np.sqrt(group['squared_error'].mean()),
            'MAPE_%': group['ape_%'].mean(),
            'bias_mean': group['error'].mean(),
            'actual_mean': group['actual'].mean(),
            'predicted_mean': group['predicted'].mean(),
            'under_predict_%': (group['bias_type'].eq('under_predict').mean() * 100),
        })
        rows.append(rec)

    result = pd.DataFrame(rows)
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].round(4)
    return result.sort_values('RMSE', ascending=False).reset_index(drop=True)


def get_worst_predictions(error_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Lay top mau co abs_error lon nhat."""
    keep_cols = [
        'baseline', 'model', 'timestamp', 'sensor_id', 'hour', 'weekday',
        'flow_regime', 'actual', 'predicted', 'error', 'abs_error', 'ape_%',
        'bias_type',
    ]
    keep_cols = [col for col in keep_cols if col in error_df.columns]
    result = error_df.sort_values('abs_error', ascending=False).head(top_n)
    result = result[keep_cols].reset_index(drop=True)
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].round(4)
    return result


def infer_error_causes(error_df: pd.DataFrame) -> pd.DataFrame:
    """Tao bang nhan dinh nguyen nhan co the va huong cai thien."""
    rows = []
    q90 = error_df['actual'].quantile(0.90)
    q10 = error_df['actual'].quantile(0.10)
    ae90 = error_df['abs_error'].quantile(0.90)

    high_under = error_df[
        (error_df['actual'] >= q90)
        & (error_df['bias_type'] == 'under_predict')
    ]
    if len(high_under) > 0:
        rows.append({
            'case': 'High-flow underestimated',
            'n_cases': len(high_under),
            'MAE': high_under['abs_error'].mean(),
            'possible_cause': (
                'Giai doan flow cao/cao diem co bien dong manh, mo hinh hoc '
                'xu huong trung binh nen du bao thap hon thuc te.'
            ),
            'improvement': (
                'Tang trong so mau high-flow, them lag dai hon/cac feature '
                'cao diem, hoac train rieng model cho congestion regime.'
            ),
        })

    low_over = error_df[
        (error_df['actual'] <= q10)
        & (error_df['bias_type'] == 'over_predict')
    ]
    if len(low_over) > 0:
        rows.append({
            'case': 'Low-flow overestimated',
            'n_cases': len(low_over),
            'MAE': low_over['abs_error'].mean(),
            'possible_cause': (
                'Dem/khoang giao thong thap co pattern khac voi gio binh '
                'thuong, lag gan nhat lam mo hinh du bao cao hon.'
            ),
            'improvement': (
                'Them interaction hour x lag, tach weekday/weekend, hoac '
                'dung loss/weight rieng cho khoang flow thap.'
            ),
        })

    if 'abs_delta_from_lag1' in error_df.columns:
        delta90 = error_df['abs_delta_from_lag1'].quantile(0.90)
        abrupt = error_df[
            (error_df['abs_delta_from_lag1'] >= delta90)
            & (error_df['abs_error'] >= ae90)
        ]
        if len(abrupt) > 0:
            rows.append({
                'case': 'Abrupt flow change',
                'n_cases': len(abrupt),
                'MAE': abrupt['abs_error'].mean(),
                'possible_cause': (
                    'Flow thay doi dot ngot so voi lag gan nhat, cac feature '
                    'lag/rolling hien tai chua bat kip diem chuyen pha.'
                ),
                'improvement': (
                    'Them delta/acceleration features, rolling trend, hoac '
                    'du bao nhieu horizon voi mo hinh sequence manh hon.'
                ),
            })

    if 'hour' in error_df.columns:
        hour_summary = summarize_error_segments(
            error_df,
            group_cols=('baseline', 'model', 'hour'),
        )
        worst_hour = hour_summary.head(1)
        if len(worst_hour) > 0:
            rows.append({
                'case': f"Worst hour segment: hour={int(worst_hour.iloc[0]['hour'])}",
                'n_cases': int(worst_hour.iloc[0]['n']),
                'MAE': worst_hour.iloc[0]['MAE'],
                'possible_cause': (
                    'Mot khung gio co pattern flow/speed rieng, co the lien '
                    'quan den cao diem hoac chuyen tiep ngay-dem.'
                ),
                'improvement': (
                    'Kiem tra SHAP theo hour, them cyclic encoding '
                    'sin/cos hour va interaction voi flow_lag.'
                ),
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result['MAE'] = result['MAE'].round(4)
    return result


def plot_error_by_hour(error_df: pd.DataFrame, save_path: str = None):
    """Boxplot abs_error theo hour."""
    if 'hour' not in error_df.columns:
        raise KeyError("error_df khong co cot 'hour'.")
    df = error_df.copy()
    df['case'] = df['baseline'] + ' | ' + df['model']
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.lineplot(
        data=df,
        x='hour',
        y='abs_error',
        hue='case',
        estimator='mean',
        marker='o',
        ax=ax,
    )
    ax.set_title('Mean absolute error by hour')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Mean absolute error')
    ax.set_xticks(sorted(df['hour'].dropna().unique()))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def plot_error_by_flow_regime(error_df: pd.DataFrame, save_path: str = None):
    """Bar chart RMSE theo flow regime."""
    if 'flow_regime' not in error_df.columns:
        raise KeyError("error_df khong co cot 'flow_regime'.")

    summary = summarize_error_segments(
        error_df,
        group_cols=('baseline', 'model', 'flow_regime'),
    )
    summary['case'] = summary['baseline'] + ' | ' + summary['model']

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=summary, x='flow_regime', y='RMSE', hue='case', ax=ax)
    ax.set_title('RMSE by flow regime')
    ax.set_xlabel('Flow regime')
    ax.set_ylabel('RMSE')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
