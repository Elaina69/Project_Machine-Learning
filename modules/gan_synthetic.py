"""Feature-space GAN cho dữ liệu giao thông đã feature engineering.

Module này không cố tạo một trục thời gian giả để ghép vào chuỗi thật. Thay
vào đó, GAN học phân phối của các mẫu supervised độc lập:
    [lag features, rolling features, time features] -> flow_target

Cách làm này khớp với chiến lược "gộp theo không gian mẫu" trong GAN.md:
synthetic samples có thể dùng cho Train Real/Test Fake, Train Fake/Test Real,
hoặc tăng cường train set mà không phá vỡ thứ tự thời gian thật.
"""
from pathlib import Path
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from modules.models import evaluate
from modules.stability import build_model

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class FeatureGenerator(nn.Module):
    """MLP generator: noise -> scaled feature vector."""
    def __init__(self, noise_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class FeatureDiscriminator(nn.Module):
    """MLP discriminator: scaled feature vector -> real/fake logit."""
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _require_torch():
    if not HAS_TORCH:
        raise ImportError('PyTorch chưa được cài. Cài bằng: pip install torch')


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_feature_gan_data(
        df: pd.DataFrame,
        feature_cols: list,
        target_col: str = 'flow_target',
        sample_size: int = None,
        random_state: int = 42,
        clip_quantiles=(0.01, 0.99),
    ) -> dict:
    """Chuẩn hóa dữ liệu để huấn luyện GAN trong không gian feature.

    Returns dict gồm:
        matrix: dữ liệu đã scale để đưa vào GAN.
        scaler: StandardScaler để inverse_transform.
        columns: danh sách cột GAN học.
        limits: ngưỡng clip theo quantile để giữ dữ liệu ảo thực tế hơn.
    """
    columns = list(feature_cols) + [target_col]
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f'Thiếu cột cho GAN: {missing}')

    clean_df = (
        df[columns]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )
    if sample_size is not None and sample_size < len(clean_df):
        clean_df = clean_df.sample(
            n=sample_size,
            random_state=random_state,
        ).reset_index(drop=True)

    scaler = StandardScaler()
    matrix = scaler.fit_transform(clean_df.values).astype(np.float32)
    q_low, q_high = clip_quantiles
    limits = clean_df.quantile([q_low, q_high]).T
    limits.columns = ['clip_low', 'clip_high']

    return {
        'matrix': matrix,
        'scaler': scaler,
        'columns': columns,
        'limits': limits,
        'n_samples': len(clean_df),
        'clip_quantiles': clip_quantiles,
    }


def train_feature_gan(
        df: pd.DataFrame,
        feature_cols: list,
        target_col: str = 'flow_target',
        sample_size: int = None,
        noise_dim: int = 32,
        hidden_dim: int = 128,
        epochs: int = 300,
        batch_size: int = 256,
        lr: float = 2e-4,
        random_state: int = 42,
        device: str = None,
        log_interval: int = 50,
    ) -> dict:
    """Huấn luyện vanilla GAN trên bảng feature đã xử lý.

    Với notebook đồ án, nên bắt đầu bằng sample_size nhỏ (ví dụ 5000-10000)
    để kiểm tra nhanh, sau đó tăng dần nếu loss và đánh giá real/fake ổn.
    """
    _require_torch()
    _set_seed(random_state)

    gan_data = prepare_feature_gan_data(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        sample_size=sample_size,
        random_state=random_state,
    )
    matrix = gan_data['matrix']
    data_dim = matrix.shape[1]
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = TensorDataset(torch.tensor(matrix, dtype=torch.float32))
    if len(dataset) < 2:
        raise ValueError('Cần ít nhất 2 mẫu để huấn luyện GAN.')
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    generator = FeatureGenerator(noise_dim, data_dim, hidden_dim).to(device)
    discriminator = FeatureDiscriminator(data_dim, hidden_dim).to(device)

    criterion = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    history = []
    print(f"\n{'='*60}")
    print('HUẤN LUYỆN FEATURE-SPACE GAN')
    print(f"   Samples: {len(matrix):,} | Data dim: {data_dim} | Device: {device}")
    print(f"   Epochs: {epochs} | Batch: {batch_size} | Noise dim: {noise_dim}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []
        for (real_batch,) in loader:
            real_batch = real_batch.to(device)
            batch_n = real_batch.size(0)
            if batch_n < 2:
                continue

            # Train discriminator.
            z = torch.randn(batch_n, noise_dim, device=device)
            fake_batch = generator(z).detach()
            real_labels = torch.full((batch_n,), 0.9, device=device)
            fake_labels = torch.zeros(batch_n, device=device)

            opt_d.zero_grad()
            d_real = criterion(discriminator(real_batch), real_labels)
            d_fake = criterion(discriminator(fake_batch), fake_labels)
            d_loss = d_real + d_fake
            d_loss.backward()
            opt_d.step()

            # Train generator.
            z = torch.randn(batch_n, noise_dim, device=device)
            opt_g.zero_grad()
            generated = generator(z)
            g_loss = criterion(discriminator(generated), torch.ones(batch_n, device=device))
            g_loss.backward()
            opt_g.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        if not d_losses:
            raise ValueError('Không có batch hợp lệ để huấn luyện GAN.')

        row = {
            'epoch': epoch,
            'd_loss': float(np.mean(d_losses)),
            'g_loss': float(np.mean(g_losses)),
        }
        history.append(row)

        if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
            print(
                f"   Epoch {epoch:03d}/{epochs} | "
                f"D_loss={row['d_loss']:.4f} | G_loss={row['g_loss']:.4f}"
            )

    generator.eval()
    return {
        **gan_data,
        'generator': generator,
        'noise_dim': noise_dim,
        'hidden_dim': hidden_dim,
        'device': device,
        'history': pd.DataFrame(history),
    }


def _postprocess_synthetic(df: pd.DataFrame, limits: pd.DataFrame = None) -> pd.DataFrame:
    out = df.copy()
    if limits is not None:
        for col in out.columns:
            if col in limits.index:
                out[col] = out[col].clip(
                    lower=limits.loc[col, 'clip_low'],
                    upper=limits.loc[col, 'clip_high'],
                )

    for col in out.columns:
        lower_col = col.lower()
        if lower_col in {'hour'}:
            out[col] = out[col].round().clip(0, 23)
        elif lower_col in {'weekday'}:
            out[col] = out[col].round().clip(0, 6)
        elif lower_col in {'is_weekend'}:
            out[col] = out[col].round().clip(0, 1)
        elif 'occupancy' in lower_col:
            out[col] = out[col].clip(0, 1)
        elif 'flow' in lower_col or 'speed' in lower_col:
            out[col] = out[col].clip(lower=0)

    return out


def generate_synthetic_data(
        gan_artifacts: dict,
        n_samples: int,
        reference_df: pd.DataFrame = None,
        random_state: int = 42,
        include_sensor_id: bool = True,
    ) -> pd.DataFrame:
    """Sinh dữ liệu ảo từ GAN và inverse về scale gốc."""
    _require_torch()
    _set_seed(random_state)

    generator = gan_artifacts['generator']
    device = gan_artifacts.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
    noise_dim = gan_artifacts['noise_dim']
    scaler = gan_artifacts['scaler']
    columns = gan_artifacts['columns']
    limits = gan_artifacts.get('limits')

    generator.to(device)
    generator.eval()
    chunks = []
    remaining = n_samples
    with torch.no_grad():
        while remaining > 0:
            batch_n = min(remaining, 4096)
            z = torch.randn(batch_n, noise_dim, device=device)
            fake_scaled = generator(z).cpu().numpy()
            chunks.append(fake_scaled)
            remaining -= batch_n

    fake_matrix = np.vstack(chunks)
    fake_values = scaler.inverse_transform(fake_matrix)
    synthetic_df = pd.DataFrame(fake_values, columns=columns)
    synthetic_df = _postprocess_synthetic(synthetic_df, limits)

    if include_sensor_id and reference_df is not None and 'sensor_id' in reference_df.columns:
        sampled_sensor = reference_df['sensor_id'].sample(
            n=n_samples,
            replace=True,
            random_state=random_state,
        ).reset_index(drop=True)
        synthetic_df.insert(0, 'sensor_id', sampled_sensor)

    synthetic_df['source'] = 'synthetic_gan'
    return synthetic_df.reset_index(drop=True)


def real_fake_cross_evaluation(
        model_name: str,
        params: dict,
        real_df: pd.DataFrame,
        fake_df: pd.DataFrame,
        feature_cols: list,
        target_col: str = 'flow_target',
        random_state: int = 42,
        n_jobs: int = -1,
        baseline_name: str = None,
        synthetic_ratio: float = None,
    ) -> pd.DataFrame:
    """Đánh giá Train Real/Test Fake và Train Fake/Test Real."""
    if fake_df is None or len(fake_df) == 0:
        raise ValueError('fake_df phải có dữ liệu để chạy Real/Fake cross-evaluation.')

    X_real = real_df[feature_cols].values
    y_real = real_df[target_col].values
    X_fake = fake_df[feature_cols].values
    y_fake = fake_df[target_col].values

    records = []
    scenarios = [
        ('Train Real / Test Fake', X_real, y_real, X_fake, y_fake),
        ('Train Fake / Test Real', X_fake, y_fake, X_real, y_real),
    ]
    for scenario, X_train, y_train, X_test, y_test in scenarios:
        model = build_model(
            model_name,
            params,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate(y_test, y_pred, y_train)
        record = {
            'scenario': scenario,
            'model': model_name,
            'n_train': len(X_train),
            'n_test': len(X_test),
            **metrics,
        }
        if baseline_name is not None:
            record['baseline'] = baseline_name
        if synthetic_ratio is not None:
            record['synthetic_ratio'] = synthetic_ratio
        records.append(record)
    return pd.DataFrame(records)


def synthetic_augmentation_experiment(
        model_name: str,
        params: dict,
        real_train_df: pd.DataFrame,
        real_test_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        feature_cols: list,
        target_col: str = 'flow_target',
        ratios=(0.5, 1.0, 2.0),
        random_state: int = 42,
        n_jobs: int = -1,
        baseline_name: str = None,
    ) -> pd.DataFrame:
    """Thử các tỷ lệ 100% real + X% synthetic để tìm điểm bão hòa."""
    rng = np.random.RandomState(random_state)
    X_test = real_test_df[feature_cols].values
    y_test = real_test_df[target_col].values

    records = []
    for ratio in ratios:
        n_fake = int(round(len(real_train_df) * ratio))
        if n_fake == 0:
            source_df = synthetic_df if synthetic_df is not None else real_train_df
            fake_part = source_df.iloc[0:0].copy()
        else:
            if synthetic_df is None or len(synthetic_df) == 0:
                raise ValueError('synthetic_df rỗng nhưng ratio yêu cầu dữ liệu synthetic.')
            fake_part = synthetic_df.sample(
                n=n_fake,
                replace=n_fake > len(synthetic_df),
                random_state=int(rng.randint(0, 1_000_000)),
            )
        aug_train = pd.concat([real_train_df, fake_part], ignore_index=True)

        model = build_model(
            model_name,
            params,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        model.fit(aug_train[feature_cols].values, aug_train[target_col].values)
        y_pred = model.predict(X_test)
        metrics = evaluate(y_test, y_pred, aug_train[target_col].values)

        record = {
            'scenario': f'100% Real + {int(ratio * 100)}% Synthetic',
            'model': model_name,
            'synthetic_ratio': ratio,
            'n_real_train': len(real_train_df),
            'n_synthetic_train': n_fake,
            'n_test': len(real_test_df),
            **metrics,
        }
        if baseline_name is not None:
            record['baseline'] = baseline_name
        records.append(record)
    return pd.DataFrame(records).sort_values('synthetic_ratio').reset_index(drop=True)


def save_synthetic_data(df: pd.DataFrame, path: str):
    """Lưu synthetic data ra CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f'Đã lưu synthetic data: {path} ({len(df):,} dòng)')
