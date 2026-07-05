from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from configs.configs import CONFIG
from modules import feature_engineering as fe
from modules.models import evaluate


ROOT_DIR = Path(__file__).resolve().parents[2]
CLEAN_DIR = ROOT_DIR / CONFIG["clean_dir"]
RAW_DATA_PATH = ROOT_DIR / CONFIG["data_path"]
FRONTEND_DIST = ROOT_DIR / "dashboard" / "frontend" / "dist"

EXPECTED_BASELINES = ["Baseline_A", "Baseline_B"]
EXPECTED_MODELS = ["5_RandomForest", "6_XGBoost"]
TARGET_COL = "flow_target"
METRIC_COLS = ["MAE", "RMSE", "MAPE", "R2", "MASE"]
RAW_METRICS = ["flow", "speed", "occupancy"]


def _baseline_tag(baseline: str) -> str:
    return "A" if baseline == "Baseline_A" else "B"


def _dataset_name(baseline: str, split: str) -> str:
    return f"baseline{_baseline_tag(baseline)}_{split}"


def _safe_float(value: Any):
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if pd.isna(value):
        return None
    return value


def _json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    clean = df.replace([np.inf, -np.inf], np.nan).copy()
    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = clean[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    clean = clean.where(pd.notna(clean), None)
    return [
        {key: _safe_float(value) for key, value in row.items()}
        for row in clean.to_dict(orient="records")
    ]


def _metric_dict(metrics: dict[str, Any]) -> dict[str, float | None]:
    return {key: _safe_float(metrics.get(key)) for key in METRIC_COLS}


def _median_step_minutes(df: pd.DataFrame) -> float:
    if "timestamp" not in df.columns or len(df) < 2:
        return 5.0
    diffs = (
        df["timestamp"]
        .sort_values()
        .diff()
        .dropna()
        .dt.total_seconds()
        .div(60)
    )
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 5.0
    return float(diffs.median())


def _default_forecast_time(df: pd.DataFrame) -> pd.Timestamp | None:
    if "timestamp" not in df.columns or df.empty:
        return None
    timestamps = df["timestamp"].dropna().sort_values().drop_duplicates().reset_index(drop=True)
    if timestamps.empty:
        return None
    step_minutes = _median_step_minutes(pd.DataFrame({"timestamp": timestamps}))
    max_horizon_steps = max(1, int(round(1440 / step_minutes)))
    default_index = max(0, len(timestamps) - max_horizon_steps - 1)
    return pd.Timestamp(timestamps.iloc[default_index])


def _nearest_time_index(timeline: pd.DataFrame, requested_time: pd.Timestamp) -> int:
    timestamps = timeline["timestamp"].reset_index(drop=True)
    if timestamps.empty:
        return 0

    values = timestamps.to_numpy(dtype="datetime64[ns]")
    requested = np.datetime64(requested_time.to_datetime64(), "ns")
    insert_at = int(np.searchsorted(values, requested, side="left"))

    if insert_at <= 0:
        return 0
    if insert_at >= len(values):
        return len(values) - 1

    prev_diff = abs(requested - values[insert_at - 1])
    next_diff = abs(values[insert_at] - requested)
    return insert_at - 1 if prev_diff <= next_diff else insert_at


class DashboardStore:
    def __init__(self):
        self.feature_cols = fe.get_feature_columns(CONFIG)
        self.registry = self._load_registry()
        self.datasets = self._load_datasets()
        self.raw_df = self._load_raw_data()
        self.shap_df = self._load_optional_csv("optimized_shap_top14_other.csv")
        self.error_by_sensor = self._load_optional_csv("forecast_error_by_sensor.csv")
        self.error_by_hour = self._load_optional_csv("forecast_error_by_hour.csv")
        self.models, self.model_errors = self._load_models()
        self.prediction_cache: dict[tuple[str, str], pd.DataFrame] = {}

    def _load_registry(self) -> pd.DataFrame:
        path = CLEAN_DIR / "demo_model_registry.csv"
        if not path.exists():
            raise RuntimeError(f"Missing model registry: {path}")

        registry = pd.read_csv(path)
        registry = registry[
            (registry["stage"] == "optimized_original")
            & registry["baseline"].isin(EXPECTED_BASELINES)
            & registry["model"].isin(EXPECTED_MODELS)
        ].copy()

        if registry.empty:
            raise RuntimeError("Registry has no optimized baseline models.")

        keep_cols = [
            "baseline",
            "model",
            "case",
            "model_path",
            "n_train",
            "n_test",
            *METRIC_COLS,
        ]
        return registry[keep_cols].drop_duplicates(["baseline", "model"], keep="last")

    def _load_optional_csv(self, file_name: str) -> pd.DataFrame:
        path = CLEAN_DIR / file_name
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def _load_datasets(self) -> dict[str, pd.DataFrame]:
        datasets = {}
        for baseline in EXPECTED_BASELINES:
            for split in ["train", "valid", "test"]:
                name = _dataset_name(baseline, split)
                path = CLEAN_DIR / f"{name}.csv"
                if not path.exists():
                    continue
                df = pd.read_csv(path)
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                datasets[name] = df
        return datasets

    def _load_raw_data(self) -> pd.DataFrame:
        if not RAW_DATA_PATH.exists():
            return pd.DataFrame()
        df = pd.read_csv(RAW_DATA_PATH)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def _load_models(self):
        loaded = {}
        errors = []
        for row in self.registry.itertuples(index=False):
            model_path = ROOT_DIR / row.model_path
            key = (row.baseline, row.model)
            if not model_path.exists():
                errors.append({
                    "baseline": row.baseline,
                    "model": row.model,
                    "message": f"Missing model file: {model_path}",
                })
                continue
            try:
                loaded[key] = joblib.load(model_path)
            except Exception as exc:
                errors.append({
                    "baseline": row.baseline,
                    "model": row.model,
                    "message": str(exc),
                })
        return loaded, errors

    def validate_baseline(self, baseline: str):
        if baseline not in EXPECTED_BASELINES:
            raise HTTPException(status_code=400, detail=f"Unknown baseline: {baseline}")

    def validate_model(self, model: str):
        if model not in EXPECTED_MODELS:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    def test_df(self, baseline: str) -> pd.DataFrame:
        self.validate_baseline(baseline)
        name = _dataset_name(baseline, "test")
        if name not in self.datasets:
            raise HTTPException(status_code=500, detail=f"Missing test dataset: {name}")
        return self.datasets[name]

    def fit_target(self, baseline: str, sensor: str | None = None):
        frames = []
        for split in ["train", "valid"]:
            name = _dataset_name(baseline, split)
            if name in self.datasets:
                df = self.datasets[name]
                if sensor:
                    df = df[df["sensor_id"] == sensor]
                frames.append(df[TARGET_COL])
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True).values

    def registry_row(self, baseline: str, model: str) -> pd.Series:
        self.validate_baseline(baseline)
        self.validate_model(model)
        rows = self.registry[
            (self.registry["baseline"] == baseline)
            & (self.registry["model"] == model)
        ]
        if rows.empty:
            raise HTTPException(status_code=404, detail="Model is not registered.")
        return rows.iloc[0]

    def prediction_frame(self, baseline: str, model: str) -> pd.DataFrame:
        self.validate_baseline(baseline)
        self.validate_model(model)
        key = (baseline, model)
        if key in self.prediction_cache:
            return self.prediction_cache[key]

        if key not in self.models:
            raise HTTPException(status_code=404, detail="Model artifact is not loaded.")

        df = self.test_df(baseline).copy()
        missing = [col for col in self.feature_cols + [TARGET_COL] if col not in df.columns]
        if missing:
            raise HTTPException(status_code=500, detail=f"Missing feature columns: {missing}")

        pred = self.models[key].predict(df[self.feature_cols])
        out = df[
            ["timestamp", "sensor_id", "flow", "speed", "occupancy", "hour", TARGET_COL]
        ].copy()
        out = out.rename(columns={TARGET_COL: "actual"})
        out["predicted"] = pred
        out["error"] = out["actual"] - out["predicted"]
        out["abs_error"] = out["error"].abs()
        out["baseline"] = baseline
        out["model"] = model
        self.prediction_cache[key] = out
        return out

    def filtered_predictions(
        self,
        baseline: str,
        model: str,
        sensor: str | None = None,
        n_points: int = 300,
    ) -> pd.DataFrame:
        df = self.prediction_frame(baseline, model)
        if sensor:
            df = df[df["sensor_id"] == sensor]
        if df.empty:
            return df
        df = df.sort_values(["sensor_id", "timestamp"])
        return df.tail(max(1, min(n_points, 3000))).reset_index(drop=True)

    def computed_metrics(self, df: pd.DataFrame, baseline: str, sensor: str | None = None):
        if df.empty:
            return {key: None for key in METRIC_COLS}
        y_train = self.fit_target(baseline, sensor=sensor)
        return _metric_dict(evaluate(df["actual"].values, df["predicted"].values, y_train))

    def registry_metrics(self, baseline: str, model: str):
        row = self.registry_row(baseline, model)
        return _metric_dict(row.to_dict())

    def sensors_for_baseline(self, baseline: str) -> list[str]:
        df = self.test_df(baseline)
        return sorted(df["sensor_id"].dropna().astype(str).unique().tolist())


@lru_cache(maxsize=1)
def get_store() -> DashboardStore:
    return DashboardStore()


app = FastAPI(
    title="SV16 Traffic Forecast Dashboard API",
    version="1.0.0",
    description="Dashboard API for the four optimized baseline models.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    store = get_store()
    expected = len(EXPECTED_BASELINES) * len(EXPECTED_MODELS)
    loaded = len(store.models)
    missing_data = [
        _dataset_name(baseline, split)
        for baseline in EXPECTED_BASELINES
        for split in ["train", "valid", "test"]
        if _dataset_name(baseline, split) not in store.datasets
    ]
    return {
        "ok": loaded == expected and not missing_data and not store.model_errors,
        "expected_model_count": expected,
        "loaded_model_count": loaded,
        "registered_model_count": int(len(store.registry)),
        "missing_data": missing_data,
        "model_errors": store.model_errors,
        "raw_data_loaded": not store.raw_df.empty,
        "scope": "optimized_baseline_models_only",
    }


@app.get("/api/options")
def options():
    store = get_store()
    sensors = {baseline: store.sensors_for_baseline(baseline) for baseline in EXPECTED_BASELINES}
    ranges = {}
    for baseline in EXPECTED_BASELINES:
        df = store.test_df(baseline)
        default_time = _default_forecast_time(df)
        ranges[baseline] = {
            "start": df["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "end": df["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S"),
            "default_forecast_time": (
                default_time.strftime("%Y-%m-%d %H:%M:%S") if default_time is not None else None
            ),
        }
    default_baseline = "Baseline_A"
    return {
        "baselines": EXPECTED_BASELINES,
        "models": EXPECTED_MODELS,
        "raw_metrics": RAW_METRICS,
        "sensors": sensors,
        "time_ranges": ranges,
        "defaults": {
            "baseline": default_baseline,
            "model": "6_XGBoost",
            "sensor": sensors[default_baseline][0] if sensors[default_baseline] else None,
            "raw_metric": "flow",
            "n_points": 300,
            "forecast_base_time": ranges[default_baseline]["default_forecast_time"],
        },
    }


@app.get("/api/raw-series")
def raw_series(
    sensor: str | None = None,
    metric: str = Query("flow", pattern="^(flow|speed|occupancy)$"),
    n_points: int = Query(500, ge=20, le=5000),
):
    store = get_store()
    if store.raw_df.empty:
        raise HTTPException(status_code=404, detail="Raw dataset is missing.")

    df = store.raw_df.copy()
    if sensor is None:
        sensor = str(df["sensor_id"].dropna().astype(str).sort_values().iloc[0])
    df = df[df["sensor_id"] == sensor].sort_values("timestamp").tail(n_points)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No raw data for sensor: {sensor}")

    stats = {
        "mean": _safe_float(df[metric].mean()),
        "std": _safe_float(df[metric].std()),
        "min": _safe_float(df[metric].min()),
        "max": _safe_float(df[metric].max()),
        "count": int(len(df)),
    }
    return {
        "sensor": sensor,
        "metric": metric,
        "stats": stats,
        "series": _json_records(df[["timestamp", "sensor_id", metric]]),
    }


@app.get("/api/predictions")
def predictions(
    baseline: str = "Baseline_A",
    model: str = "6_XGBoost",
    sensor: str | None = None,
    n_points: int = Query(300, ge=20, le=3000),
):
    store = get_store()
    if sensor is None:
        sensors = store.sensors_for_baseline(baseline)
        sensor = sensors[0] if sensors else None
    df = store.filtered_predictions(baseline, model, sensor=sensor, n_points=n_points)
    return {
        "baseline": baseline,
        "model": model,
        "sensor": sensor,
        "metrics": store.computed_metrics(df, baseline, sensor=sensor),
        "series": _json_records(df),
    }


@app.get("/api/metrics")
def metrics(
    baseline: str = "Baseline_A",
    model: str = "6_XGBoost",
    sensor: str | None = None,
):
    store = get_store()
    registry = store.registry_metrics(baseline, model)
    computed = None
    if sensor:
        df = store.filtered_predictions(baseline, model, sensor=sensor, n_points=3000)
        computed = store.computed_metrics(df, baseline, sensor=sensor)
    return {
        "baseline": baseline,
        "model": model,
        "sensor": sensor,
        "registry_metrics": registry,
        "computed_metrics": computed,
    }


@app.get("/api/comparison")
def comparison(
    model: str = "6_XGBoost",
    n_points: int = Query(200, ge=20, le=1000),
):
    store = get_store()
    store.validate_model(model)
    rows = []
    series = {}
    for baseline in EXPECTED_BASELINES:
        sensors = store.sensors_for_baseline(baseline)
        sensor = sensors[0] if sensors else None
        pred_df = store.filtered_predictions(baseline, model, sensor=sensor, n_points=n_points)
        rows.append({
            "baseline": baseline,
            "model": model,
            "sensor": sensor,
            **store.registry_metrics(baseline, model),
        })
        series[baseline] = _json_records(pred_df)

    best = min(rows, key=lambda row: row["RMSE"] if row["RMSE"] is not None else float("inf"))
    return {
        "model": model,
        "rows": rows,
        "series": series,
        "best_baseline": best["baseline"],
    }


@app.get("/api/xai")
def xai(
    baseline: str = "Baseline_A",
    model: str = "6_XGBoost",
):
    store = get_store()
    store.validate_baseline(baseline)
    store.validate_model(model)
    if store.shap_df.empty:
        return {"baseline": baseline, "model": model, "features": []}

    df = store.shap_df[
        (store.shap_df["stage"] == "optimized_original")
        & (store.shap_df["baseline"] == baseline)
        & (store.shap_df["model"] == model)
    ].copy()
    if df.empty:
        return {"baseline": baseline, "model": model, "features": []}
    df = df.sort_values("rank").head(15)
    return {
        "baseline": baseline,
        "model": model,
        "features": _json_records(df[["feature", "rank", "mean_abs_shap", "is_other"]]),
    }


@app.get("/api/alerts")
def alerts(
    baseline: str = "Baseline_A",
    model: str = "6_XGBoost",
    sensor: str | None = None,
    forecast_minutes: Annotated[int, Query(ge=5, le=1440)] = 15,
    base_time: str | None = None,
):
    store = get_store()
    if sensor is None:
        sensors = store.sensors_for_baseline(baseline)
        sensor = sensors[0] if sensors else None

    pred_full = store.prediction_frame(baseline, model)
    pred_df = pred_full[pred_full["sensor_id"] == sensor] if sensor else pred_full
    if pred_df.empty:
        return {"baseline": baseline, "model": model, "sensor": sensor, "alerts": []}

    timeline = pred_df.sort_values("timestamp").reset_index(drop=True)
    step_minutes = _median_step_minutes(timeline)
    steps_ahead = max(1, int(round(forecast_minutes / step_minutes)))

    if base_time:
        try:
            requested_base_time = pd.to_datetime(base_time)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"base_time không hợp lệ: {base_time}") from exc
        current_index = _nearest_time_index(timeline, requested_base_time)
    else:
        max_horizon_steps = max(1, int(round(1440 / step_minutes)))
        current_index = max(0, len(timeline) - max_horizon_steps - 1)
        requested_base_time = pd.Timestamp(timeline.iloc[current_index]["timestamp"])

    forecast_index = min(current_index + steps_ahead, len(timeline) - 1)
    current_row = timeline.iloc[current_index]
    latest = timeline.iloc[forecast_index]
    effective_minutes = (
        (latest["timestamp"] - current_row["timestamp"]).total_seconds() / 60
        if latest["timestamp"] >= current_row["timestamp"]
        else steps_ahead * step_minutes
    )
    predicted_q25 = float(pred_full["predicted"].quantile(0.25))
    predicted_q75 = float(pred_full["predicted"].quantile(0.75))
    predicted_q90 = float(pred_full["predicted"].quantile(0.90))
    occupancy_q75 = float(store.test_df(baseline)["occupancy"].quantile(0.75))
    alerts_out = []

    latest_predicted = float(latest["predicted"])
    latest_actual = float(latest["actual"])
    latest_abs_error = abs(latest_actual - latest_predicted)
    if latest_predicted >= predicted_q90:
        flow_status = "Rất cao"
        flow_level = "risk"
        flow_recommendation = "Cần ưu tiên kiểm tra tuyến/sensor này vì dự báo lưu lượng đang ở vùng rất cao."
    elif latest_predicted >= predicted_q75:
        flow_status = "Cao"
        flow_level = "warning"
        flow_recommendation = "Nên theo dõi sát sensor này trong khung giờ gần nhất."
    elif latest_predicted <= predicted_q25:
        flow_status = "Thấp"
        flow_level = "info"
        flow_recommendation = "Lưu lượng dự báo thấp; tiếp tục theo dõi nếu có biến động bất thường."
    else:
        flow_status = "Bình thường"
        flow_level = "ok"
        flow_recommendation = "Lưu lượng dự báo nằm trong vùng vận hành thông thường."

    flow_forecast = {
        "requested_base_time": requested_base_time.strftime("%Y-%m-%d %H:%M:%S"),
        "demo_now": current_row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": latest["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
        "requested_minutes": int(forecast_minutes),
        "effective_minutes": _safe_float(effective_minutes),
        "predicted_flow": _safe_float(latest_predicted),
        "actual_flow": _safe_float(latest_actual),
        "abs_error": _safe_float(latest_abs_error),
        "status": flow_status,
        "level": flow_level,
        "q25": _safe_float(predicted_q25),
        "q75": _safe_float(predicted_q75),
        "q90": _safe_float(predicted_q90),
        "recommendation": flow_recommendation,
    }

    alerts_out.append({
        "level": flow_level,
        "title": "Cảnh báo lưu lượng dòng xe",
        "message": (
            f"Dự báo flow sau {forecast_minutes} phút là {latest_predicted:.2f} "
            f"tại {flow_forecast['timestamp']}; phân loại: {flow_status}."
        ),
        "recommendation": flow_recommendation,
    })

    if latest["predicted"] >= predicted_q75:
        alerts_out.append({
            "level": "warning",
            "title": "Lưu lượng dự báo cao",
            "message": f"Flow dự báo {latest['predicted']:.2f} cao hơn ngưỡng Q75 {predicted_q75:.2f}.",
            "recommendation": "Ưu tiên theo dõi sensor này trong khung giờ gần nhất.",
        })

    if latest["occupancy"] >= occupancy_q75:
        alerts_out.append({
            "level": "warning",
            "title": "Occupancy cao",
            "message": f"Occupancy hiện tại {latest['occupancy']:.3f} cao hơn ngưỡng Q75 {occupancy_q75:.3f}.",
            "recommendation": "Có thể xuất hiện mật độ phương tiện cao; cần kiểm tra tốc độ và flow đi kèm.",
        })

    sensor_risk = None
    if not store.error_by_sensor.empty and sensor:
        rows = store.error_by_sensor[
            (store.error_by_sensor["baseline"] == baseline)
            & (store.error_by_sensor["model"] == model)
            & (store.error_by_sensor["sensor_id"] == sensor)
        ]
        if not rows.empty:
            median_rmse = float(store.error_by_sensor["RMSE"].median())
            row = rows.iloc[0]
            if float(row["RMSE"]) > median_rmse:
                sensor_risk = row
                alerts_out.append({
                    "level": "risk",
                    "title": "Nguy cơ sai số cao theo sensor",
                    "message": f"RMSE sensor {sensor} là {row['RMSE']:.2f}, cao hơn median {median_rmse:.2f}.",
                    "recommendation": "Khi dùng dự báo cho sensor này, nên xem thêm sai số lịch sử và bias.",
                })

    if not store.error_by_hour.empty:
        rows = store.error_by_hour[
            (store.error_by_hour["baseline"] == baseline)
            & (store.error_by_hour["model"] == model)
            & (store.error_by_hour["hour"] == int(latest["hour"]))
        ]
        if not rows.empty:
            median_rmse = float(store.error_by_hour["RMSE"].median())
            row = rows.iloc[0]
            if float(row["RMSE"]) > median_rmse:
                alerts_out.append({
                    "level": "risk",
                    "title": "Nguy cơ sai số cao theo giờ",
                    "message": f"RMSE giờ {int(latest['hour'])} là {row['RMSE']:.2f}, cao hơn median {median_rmse:.2f}.",
                    "recommendation": "Cần thận trọng với các quyết định điều phối trong khung giờ này.",
                })

    registry_rows = store.registry[store.registry["baseline"] == baseline]
    best_row = registry_rows.sort_values("RMSE").iloc[0]
    if best_row["model"] != model:
        alerts_out.append({
            "level": "info",
            "title": f"Nên ưu tiên {best_row['model']}",
            "message": f"{best_row['model']} đang có RMSE thấp nhất trong {baseline}: {best_row['RMSE']:.2f}.",
            "recommendation": "Dùng model có RMSE thấp hơn làm lựa chọn mặc định cho baseline này.",
        })
    else:
        alerts_out.append({
            "level": "ok",
            "title": "Model đang chọn là lựa chọn tốt nhất",
            "message": f"{model} có RMSE thấp nhất trong {baseline}.",
            "recommendation": "Có thể dùng model này làm mốc demo chính cho baseline hiện tại.",
        })

    if not alerts_out:
        alerts_out.append({
            "level": "ok",
            "title": "Trạng thái ổn định",
            "message": "Không phát hiện dấu hiệu rủi ro nổi bật theo rule hiện tại.",
            "recommendation": "Tiếp tục theo dõi dự báo và sai số theo sensor.",
        })

    return {
        "baseline": baseline,
        "model": model,
        "sensor": sensor,
        "flow_forecast": flow_forecast,
        "latest": {
            "timestamp": latest["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "predicted": _safe_float(latest["predicted"]),
            "actual": _safe_float(latest["actual"]),
            "occupancy": _safe_float(latest["occupancy"]),
            "hour": int(latest["hour"]),
        },
        "sensor_risk": None if sensor_risk is None else _json_records(pd.DataFrame([sensor_risk]))[0],
        "alerts": alerts_out,
    }


@app.get("/", include_in_schema=False)
def index():
    index_path = FRONTEND_DIST / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse(
        """
        <h1>SV16 Dashboard API</h1>
        <p>React build chưa tồn tại. Chạy:</p>
        <pre>cd dashboard/frontend
npm install
npm run build</pre>
        <p>Sau đó mở lại trang này, hoặc chạy frontend dev server bằng <code>npm run dev</code>.</p>
        """
    )


assets_path = FRONTEND_DIST / "assets"
if assets_path.exists():
    app.mount("/assets", StaticFiles(directory=assets_path), name="assets")
