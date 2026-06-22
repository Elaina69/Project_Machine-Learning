"""SHAP explainability cho cac mo hinh sau Pymoo va sau GAN.

Pipeline chinh chi dung SHAP. SHAP summary duoc gom thanh 14 feature quan
trong nhat va mot dong ``other`` cho cac feature con lai de tranh qua tai hinh.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules.stability import build_model


def fit_optimized_model(
        model_name: str,
        params: dict,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        feature_cols: list,
        target_col: str,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
    """Fit lai model toi uu tren train + valid de giai thich."""
    fit_df = pd.concat([train_df, valid_df], ignore_index=True)
    model = build_model(
        model_name,
        params,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(fit_df[feature_cols], fit_df[target_col].values)
    return model, fit_df


def _sample_df(df: pd.DataFrame, n: int = 2000, random_state: int = 42):
    if n is None or len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)


def compute_shap_values(
        model,
        data_df: pd.DataFrame,
        feature_cols: list,
        sample_size: int = 1500,
        random_state: int = 42,
    ):
    """Tinh SHAP values cho tree-based model.

    Returns:
        dict co shap_values, X_sample, mean_abs_shap.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            'Chua cai SHAP. Chay: pip install shap hoac cai requirements.txt.'
        ) from exc

    X_sample = _sample_df(
        data_df[feature_cols],
        n=sample_size,
        random_state=random_state,
    )
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if hasattr(shap_values, 'values'):
        shap_values = shap_values.values
    shap_values = np.asarray(shap_values)

    mean_abs = (
        pd.DataFrame({
            'feature': feature_cols,
            'mean_abs_shap': np.abs(shap_values).mean(axis=0),
        })
        .sort_values('mean_abs_shap', ascending=False)
        .reset_index(drop=True)
    )
    mean_abs['rank'] = np.arange(1, len(mean_abs) + 1)
    return {
        'shap_values': shap_values,
        'X_sample': X_sample,
        'mean_abs_shap': mean_abs,
    }


def aggregate_shap_top_other(
        shap_result: dict,
        top_features: int = 14,
        other_label: str = 'other',
    ) -> dict:
    """Gom SHAP thanh top feature + ``other``.

    ``other`` duoc tinh sao cho mean_abs_shap cua no bang tong mean_abs_shap
    cua cac feature khong nam trong top_features.
    """
    shap_values = np.asarray(shap_result['shap_values'])
    X_sample = shap_result['X_sample'].reset_index(drop=True)
    if shap_values.ndim != 2:
        raise ValueError('shap_values phai la ma tran 2 chieu.')
    if shap_values.shape[1] != X_sample.shape[1]:
        raise ValueError('So cot SHAP khong khop voi X_sample.')

    feature_names = list(X_sample.columns)
    mean_abs = np.abs(shap_values).mean(axis=0)
    n_top = min(int(top_features), len(feature_names))
    top_idx = np.argsort(-mean_abs)[:n_top]
    other_idx = np.array(
        [idx for idx in range(len(feature_names)) if idx not in set(top_idx)]
    )

    agg_values = shap_values[:, top_idx]
    agg_X = X_sample.iloc[:, top_idx].copy()
    agg_feature_names = [feature_names[idx] for idx in top_idx]

    rows = [
        {
            'feature': feature_names[idx],
            'mean_abs_shap': float(mean_abs[idx]),
            'is_other': False,
        }
        for idx in top_idx
    ]

    if len(other_idx) > 0:
        other_abs_per_sample = np.abs(shap_values[:, other_idx]).sum(axis=1)
        other_signed_sum = shap_values[:, other_idx].sum(axis=1)
        other_sign = np.sign(other_signed_sum)
        other_sign[other_sign == 0] = 1
        other_values = (other_abs_per_sample * other_sign).reshape(-1, 1)
        agg_values = np.hstack([agg_values, other_values])
        agg_X[other_label] = other_abs_per_sample
        agg_feature_names.append(other_label)
        rows.append({
            'feature': other_label,
            'mean_abs_shap': float(mean_abs[other_idx].sum()),
            'is_other': True,
        })

    mean_abs_table = (
        pd.DataFrame(rows)
        .sort_values('mean_abs_shap', ascending=False)
        .reset_index(drop=True)
    )
    mean_abs_table['rank'] = np.arange(1, len(mean_abs_table) + 1)

    return {
        'shap_values': agg_values,
        'X_sample': agg_X,
        'mean_abs_shap': mean_abs_table,
        'feature_names': agg_feature_names,
        'top_features': n_top,
        'other_label': other_label,
    }


def plot_shap_summary(
        shap_result: dict,
        save_dir: str = None,
        prefix: str = 'shap',
        max_display: int = 15,
        top_features: int = None,
        other_label: str = 'other',
    ):
    """Ve SHAP beeswarm neu co shap package."""
    import shap

    plot_result = shap_result
    if top_features is not None:
        plot_result = aggregate_shap_top_other(
            shap_result,
            top_features=top_features,
            other_label=other_label,
        )

    shap_values = plot_result['shap_values']
    X_sample = plot_result['X_sample']
    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=max_display,
        show=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(9, 6)
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / f'{prefix}_shap_summary.png',
                    bbox_inches='tight', dpi=150)
    plt.show()
    return plot_result


def shap_top_other_table(
        shap_result: dict,
        case: str = None,
        baseline: str = None,
        model: str = None,
        stage: str = None,
        top_features: int = 14,
        other_label: str = 'other',
    ) -> pd.DataFrame:
    """Tra ve bang mean_abs_shap top feature + other kem metadata."""
    aggregated = aggregate_shap_top_other(
        shap_result,
        top_features=top_features,
        other_label=other_label,
    )
    table = aggregated['mean_abs_shap'].copy()
    if case is not None:
        table.insert(0, 'case', case)
    if model is not None:
        table.insert(0, 'model', model)
    if baseline is not None:
        table.insert(0, 'baseline', baseline)
    if stage is not None:
        table.insert(0, 'stage', stage)
    return table


def plot_shap_dependence_flow_hour(
        shap_result: dict,
        flow_feature: str = 'flow_lag_1',
        hour_feature: str = 'hour',
        save_dir: str = None,
        prefix: str = 'shap',
    ):
    """Ve SHAP dependence cho flow_lag_1 va hour."""
    X_sample = shap_result['X_sample']
    shap_values = shap_result['shap_values']
    features = [flow_feature, hour_feature]
    missing = [feature for feature in features if feature not in X_sample.columns]
    if missing:
        raise KeyError(f'Thieu feature de ve SHAP dependence: {missing}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, feature in zip(axes, features):
        idx = X_sample.columns.get_loc(feature)
        ax.scatter(
            X_sample[feature],
            shap_values[:, idx],
            s=8,
            alpha=0.35,
        )
        ax.axhline(0, color='black', linewidth=1, linestyle='--')
        ax.set_title(f'SHAP dependence: {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('SHAP value')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / f'{prefix}_shap_dependence_flow_hour.png',
                    bbox_inches='tight', dpi=150)
    plt.show()


def compare_flow_hour_shap(
        shap_tables: dict,
        flow_feature: str = 'flow_lag_1',
        hour_feature: str = 'hour',
    ) -> pd.DataFrame:
    """So sanh SHAP importance cua flow/hour giua cac baseline/model."""
    rows = []
    for case_name, shap_result in shap_tables.items():
        table = shap_result['mean_abs_shap']
        for feature in [flow_feature, hour_feature]:
            matched = table[table['feature'] == feature]
            if len(matched) == 0:
                continue
            rows.append({
                'case': case_name,
                'feature': feature,
                'rank': int(matched.iloc[0]['rank']),
                'mean_abs_shap': float(matched.iloc[0]['mean_abs_shap']),
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        result['mean_abs_shap'] = result['mean_abs_shap'].round(6)
    return result.sort_values(['feature', 'mean_abs_shap'], ascending=[True, False])
