"""
Module Models — Định nghĩa, huấn luyện và đánh giá 10 mô hình.

3 Trivial Baselines:
    0a. Seasonal Naive
    0b. Drift Method
    0c. Simple Moving Average (SMA)

7 mô hình ML:
    1. Linear Regression
    2. Ridge Regression
    3. K-Nearest Neighbors (KNN)
    4. Decision Tree Regressor
    5. Random Forest Regressor
    6. XGBoost Regressor
    7. LSTM (PyTorch)
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost chưa cài. Dùng: pip install xgboost")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️ PyTorch chưa cài. Dùng: pip install torch")


# ─── Metrics ──────────────────────────────────────────────────────────────

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error, bỏ qua y_true == 0."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(y_true, y_pred) -> dict:
    """Tính 4 metrics: MAE, RMSE, MAPE, R²."""
    return {
        'MAE': round(mean_absolute_error(y_true, y_pred), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        'MAPE': round(mape(y_true, y_pred), 4),
        'R2': round(r2_score(y_true, y_pred), 4),
    }


# ─── LSTM Model (PyTorch) ────────────────────────────────────────────────

class LSTMNet(nn.Module):
    """Mạng LSTM cho time series regression."""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # lấy output timestep cuối
        return self.fc(out).squeeze(-1)


class LSTMWrapper:
    """
    Wrapper cho LSTM, API tương thích sklearn (fit/predict).
    Dữ liệu tabular sẽ được reshape thành (batch, 1, n_features).
    """
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.2,
                 lr=0.001, epochs=50, batch_size=256, patience=8,
                 device=None):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.model = None
        self.train_losses = []

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        """Huấn luyện LSTM với early stopping trên validation."""
        X_tr = self.scaler.fit_transform(X_train)
        y_tr = np.array(y_train, dtype=np.float32)

        # Reshape: (n_samples, 1, n_features) — mỗi sample là 1 timestep
        X_tr = X_tr.reshape(X_tr.shape[0], 1, X_tr.shape[1])
        X_tensor = torch.FloatTensor(X_tr).to(self.device)
        y_tensor = torch.FloatTensor(y_tr).to(self.device)
        train_ds = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        # Validation data
        has_valid = X_valid is not None and y_valid is not None
        if has_valid:
            X_val = self.scaler.transform(X_valid).reshape(-1, 1, X_train.shape[1])
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.FloatTensor(np.array(y_valid, dtype=np.float32)).to(self.device)

        n_features = X_train.shape[1]
        self.model = LSTMNet(n_features, self.hidden_size,
                             self.num_layers, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(train_ds)
            self.train_losses.append(epoch_loss)

            # Validation
            if has_valid:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val_t)
                    val_loss = criterion(val_pred, y_val_t).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"   ⏹ Early stopping tại epoch {epoch+1}")
                        break

            if (epoch + 1) % 10 == 0:
                msg = f"   Epoch {epoch+1}/{self.epochs} — train_loss={epoch_loss:.4f}"
                if has_valid:
                    msg += f" — val_loss={val_loss:.4f}"
                print(msg)

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict(self, X):
        """Dự đoán trên dữ liệu mới."""
        X_sc = self.scaler.transform(X).reshape(-1, 1, X.shape[1])
        X_tensor = torch.FloatTensor(X_sc).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()
        return preds


# ─── Trivial Baselines ────────────────────────────────────────────────────

class SeasonalNaive:
    """
    Seasonal Naive: dự báo = giá trị cùng thời điểm ngày hôm trước.
    Sử dụng flow_lag_288 nếu có (288 bước × 5min = 1 ngày),
    fallback về flow_lag_12 (1 giờ) nếu không.
    Với tabular features, sử dụng cột lag xa nhất có sẵn.
    """
    def __init__(self, season_lag_idx=None):
        self.season_lag_idx = season_lag_idx  # index cột lag xa nhất
        self.fallback_mean = 0

    def fit(self, X, y):
        self.fallback_mean = np.mean(y)
        # Tìm cột lag xa nhất (index cuối cùng trong nhóm lag đầu tiên)
        # Mặc định dùng cột cuối trong 5 flow_lag features (flow_lag_12 = index 4)
        if self.season_lag_idx is None:
            # flow_lag columns nằm ở đầu, cột thứ 5 (index 4) = flow_lag_12
            self.season_lag_idx = min(4, X.shape[1] - 1)
        return self

    def predict(self, X):
        preds = X[:, self.season_lag_idx].copy()
        mask = np.isnan(preds)
        preds[mask] = self.fallback_mean
        return preds


class DriftMethod:
    """
    Drift Method: dự báo = giá trị gần nhất + trend trung bình.
    Sử dụng flow_lag_1 và flow_lag_3 để ước lượng drift.
    """
    def __init__(self):
        self.avg_drift = 0

    def fit(self, X, y):
        # flow_lag_1 = index 0, flow_lag_3 = index 2
        lag1 = X[:, 0]  # flow_lag_1 = flow tại t-1
        lag3 = X[:, 2]  # flow_lag_3 = flow tại t-3
        drifts = (lag1 - lag3) / 2  # trung bình thay đổi mỗi bước
        self.avg_drift = np.nanmean(drifts)
        return self

    def predict(self, X):
        lag1 = X[:, 0]  # flow_lag_1
        # Dự báo = lag1 + drift × horizon (horizon=3 bước)
        return lag1 + self.avg_drift * 3


class SimpleMovingAverage:
    """
    Simple Moving Average: dự báo = trung bình rolling mean.
    Sử dụng cột flow_roll_mean có sẵn trong features.
    """
    def __init__(self, roll_mean_idx=None):
        self.roll_mean_idx = roll_mean_idx
        self.fallback_mean = 0

    def fit(self, X, y):
        self.fallback_mean = np.mean(y)
        if self.roll_mean_idx is None:
            # flow_roll_mean_12 nằm ở sau các lag features
            # Với lag_steps=[1,2,3,6,12], lag_cols=3 → 15 lag features
            # rolling bắt đầu từ index 15, flow_roll_mean_3=15, mean_6=17, mean_12=19
            self.roll_mean_idx = min(19, X.shape[1] - 1)
        return self

    def predict(self, X):
        preds = X[:, self.roll_mean_idx].copy()
        mask = np.isnan(preds)
        preds[mask] = self.fallback_mean
        return preds


# ─── Model Factory ────────────────────────────────────────────────────────

def get_models(config: dict = None) -> dict:
    """
    Trả về dict {tên: model_instance} cho 10 mô hình (3 trivial + 7 ML).

    config có thể chứa hyperparameters tùy chỉnh:
        config['model_params']['random_forest'] = {'n_estimators': 200}
    """
    if config is None:
        config = {}
    params = config.get('model_params', {})
    rs = config.get('random_state', 42)

    models = {
        '0a_SeasonalNaive': SeasonalNaive(),
        '0b_DriftMethod': DriftMethod(),
        '0c_SMA': SimpleMovingAverage(),
        '1_LinearRegression': LinearRegression(
            **params.get('linear_regression', {})
        ),
        '2_Ridge': Ridge(
            alpha=params.get('ridge', {}).get('alpha', 1.0),
            **{k: v for k, v in params.get('ridge', {}).items() if k != 'alpha'}
        ),
        '3_KNN': KNeighborsRegressor(
            n_neighbors=params.get('knn', {}).get('n_neighbors', 10),
            weights=params.get('knn', {}).get('weights', 'distance'),
            n_jobs=-1,
        ),
        '4_DecisionTree': DecisionTreeRegressor(
            max_depth=params.get('decision_tree', {}).get('max_depth', 15),
            random_state=rs,
        ),
        '5_RandomForest': RandomForestRegressor(
            n_estimators=params.get('random_forest', {}).get('n_estimators', 100),
            max_depth=params.get('random_forest', {}).get('max_depth', 15),
            random_state=rs, n_jobs=-1,
        ),
    }

    if HAS_XGBOOST:
        models['6_XGBoost'] = XGBRegressor(
            n_estimators=params.get('xgboost', {}).get('n_estimators', 200),
            max_depth=params.get('xgboost', {}).get('max_depth', 8),
            learning_rate=params.get('xgboost', {}).get('learning_rate', 0.1),
            random_state=rs, n_jobs=-1, verbosity=0,
        )

    if HAS_TORCH:
        lstm_params = params.get('lstm', {})
        models['7_LSTM'] = LSTMWrapper(
            hidden_size=lstm_params.get('hidden_size', 64),
            num_layers=lstm_params.get('num_layers', 2),
            dropout=lstm_params.get('dropout', 0.2),
            lr=lstm_params.get('lr', 0.001),
            epochs=lstm_params.get('epochs', 50),
            batch_size=lstm_params.get('batch_size', 256),
            patience=lstm_params.get('patience', 8),
        )

    return models


# ─── Train & Evaluate ─────────────────────────────────────────────────────

def train_and_evaluate(models: dict, X_train, y_train,
                       X_valid, y_valid, X_test, y_test,
                       baseline_name: str = "Baseline") -> pd.DataFrame:
    """
    Huấn luyện tất cả mô hình, đánh giá trên valid + test.

    Returns:
        DataFrame với columns: model, MAE, RMSE, MAPE, R2 (trên test set)
    """
    results = []
    trained_models = {}

    print(f"\n{'='*60}")
    print(f"🚀 HUẤN LUYỆN {baseline_name} — {len(models)} mô hình")
    print(f"{'='*60}")

    for name, model in models.items():
        print(f"\n--- {name} ---")
        try:
            if isinstance(model, LSTMWrapper):
                model.fit(X_train, y_train, X_valid, y_valid)
            else:
                model.fit(X_train, y_train)

            y_pred_test = model.predict(X_test)
            y_pred_valid = model.predict(X_valid)

            test_metrics = evaluate(y_test, y_pred_test)
            valid_metrics = evaluate(y_valid, y_pred_valid)

            print(f"   Valid — MAE={valid_metrics['MAE']:.2f}  "
                  f"RMSE={valid_metrics['RMSE']:.2f}  R²={valid_metrics['R2']:.4f}")
            print(f"   Test  — MAE={test_metrics['MAE']:.2f}  "
                  f"RMSE={test_metrics['RMSE']:.2f}  R²={test_metrics['R2']:.4f}")

            results.append({
                'model': name,
                'baseline': baseline_name,
                **{f'test_{k}': v for k, v in test_metrics.items()},
                **{f'valid_{k}': v for k, v in valid_metrics.items()},
            })
            trained_models[name] = {
                'instance': model,
                'y_pred_test': y_pred_test,
                'y_pred_valid': y_pred_valid,
                'test_metrics': test_metrics,
                'valid_metrics': valid_metrics,
            }
        except Exception as e:
            print(f"   ❌ Lỗi: {e}")
            results.append({'model': name, 'baseline': baseline_name,
                            'test_MAE': np.nan, 'test_RMSE': np.nan,
                            'test_MAPE': np.nan, 'test_R2': np.nan})

    results_df = pd.DataFrame(results)
    print(f"\n✅ {baseline_name} hoàn tất!")
    return results_df, trained_models


# ─── Save / Load ──────────────────────────────────────────────────────────

def save_models(trained_models: dict, save_dir: str, baseline_name: str):
    """Lưu tất cả mô hình đã train vào thư mục."""
    save_path = Path(save_dir) / baseline_name
    save_path.mkdir(parents=True, exist_ok=True)
    for name, data in trained_models.items():
        model = data['instance']
        if isinstance(model, LSTMWrapper):
            if model.model is not None:
                torch.save({
                    'model_state': model.model.state_dict(),
                    'scaler': model.scaler,
                    'config': {
                        'hidden_size': model.hidden_size,
                        'num_layers': model.num_layers,
                        'dropout': model.dropout,
                        'input_size': model.model.lstm.input_size,
                    }
                }, save_path / f"{name}.pth")
        else:
            joblib.dump(model, save_path / f"{name}.pkl")
    print(f"✅ Đã lưu {len(trained_models)} mô hình vào {save_path}")


def load_sklearn_model(filepath: str):
    """Load mô hình sklearn từ file .pkl."""
    return joblib.load(filepath)
