# Kế hoạch Đồ án Học máy — SV16
## **So sánh dự báo Flow giữa hai tập Sensor trong PeMSD3**

---

## 1. Thông tin chung

| Mục | Chi tiết |
|---|---|
| **Sinh viên** | SV16 |
| **Nhóm bài** | Nhóm 2 — Dữ liệu PeMSD (flow, speed, occupancy) |
| **Đề bài** | So sánh dự báo flow giữa **hai tập sensor** (2 baselines) trong PeMSD3 |
| **Dataset** | `SV16_PeMSD3_sample_8sensors.csv` |
| **Yêu cầu mô hình** | So sánh 2 baselines × **7 mô hình** học máy (tối thiểu 5, mở rộng 7–10) |
| **Bài toán** | Time-series Regression — dự báo flow tương lai |
| **Sản phẩm cuối** | `main.ipynb` (phân tích + huấn luyện) + `demo.ipynb` (Dashboard trực quan) |

---

## 2. Phân tích Dataset gốc

### 2.1 Tổng quan dữ liệu

| Thuộc tính | Giá trị |
|---|---|
| **Số dòng** | 48,385 (không tính header) |
| **Số cột** | 6 (`timestamp`, `sensor_id`, `flow`, `speed`, `occupancy`, `district`) |
| **Số sensor** | 8 (PEMSD3_007 → PEMSD3_014) |
| **Mỗi sensor** | ~6,048 dòng |
| **Khoảng lấy mẫu** | Mỗi 5 phút |
| **Khoảng thời gian** | 2024-02-01 00:00 → 2024-02-21 23:55 (21 ngày) |
| **District** | D3 (tất cả đều thuộc District 3) |

### 2.2 Mô tả các cột

| Cột | Ý nghĩa | Kiểu dữ liệu |
|---|---|---|
| `timestamp` | Mốc thời gian ghi nhận (5 phút/lần) | datetime |
| `sensor_id` | Mã cảm biến giao thông (PEMSD3_007..014) | categorical |
| `flow` | Số xe đi qua trong khoảng 5 phút quan sát | float (biến mục tiêu) |
| `speed` | Tốc độ trung bình tại sensor (mph) | float |
| `occupancy` | Mức chiếm dụng detector (0–1), liên quan mật độ | float |
| `district` | Khu vực quận (luôn là D3) | categorical |

### 2.3 Đặc điểm quan trọng

- **Flow cao + speed thấp** → Tình trạng **tải cao** hoặc **tắc nghẽn**
- **Flow thấp + speed cao** → Giao thông **thông thoáng**
- Giá trị `occupancy` capped ở **0.92** → có thể là giá trị bão hòa sensor
- Có pattern rõ theo **giờ trong ngày** (đêm thấp, giờ cao điểm cao)
- Cần phân biệt **ngày trong tuần vs cuối tuần** (weekday effect)

---

## 3. Định nghĩa 2 Baselines (yêu cầu đề bài)

> **QUAN TRỌNG:** Đề bài yêu cầu "So sánh **ít nhất 02 baseline** với tối thiểu 05 mô hình học máy; khuyến khích mở rộng lên 7-10 mô hình."
> → Mỗi baseline là **1 cụm 4 sensor**, chạy cùng **7 mô hình** học máy, sau đó so sánh kết quả giữa 2 baselines.

### Phương án chia 2 baselines:

| | Baseline A (4 sensors) | Baseline B (4 sensors) |
|---|---|---|
| **Sensors** | PEMSD3_007, PEMSD3_008, PEMSD3_009, PEMSD3_010 | PEMSD3_011, PEMSD3_012, PEMSD3_013, PEMSD3_014 |
| **Số dòng** | ~24,192 | ~24,192 |
| **Ý nghĩa** | Cụm sensor nhóm đầu | Cụm sensor nhóm sau |

> **Ghi chú:** Có thể điều chỉnh cách chia sau khi phân tích EDA (ví dụ: chia theo mức flow trung bình cao/thấp, hoặc theo vị trí sensor). Phương án trên là chia liên tiếp để đơn giản.

---

## 4. Chia dữ liệu — Hold-out Split theo thời gian

> **⚠️ QUAN TRỌNG:** Vì đây là bài toán **time series**, dữ liệu PHẢI được chia theo **trình tự thời gian** (Hold-out Split, KHÔNG shuffle) để tránh rò rỉ dữ liệu tương lai vào tập huấn luyện.

### Tỷ lệ chia: Train 70% / Validation 15% / Test 15%

Tổng thời gian: 21 ngày (2024-02-01 → 2024-02-21)

| Tập | Tỷ lệ | Khoảng thời gian | Số ngày |
|---|---|---|---|
| **Train** | 70% | 2024-02-01 → 2024-02-15 12:00 | ~14.7 ngày |
| **Validation** | 15% | 2024-02-15 12:00 → 2024-02-18 18:00 | ~3.15 ngày |
| **Test** | 15% | 2024-02-18 18:00 → 2024-02-21 23:55 | ~3.15 ngày |

> **Ghi chú:** Ranh giới chính xác sẽ được tính trong code dựa trên số dòng thực tế per sensor. Mỗi baseline (A, B) đều chia **riêng** với cùng tỷ lệ và cùng ranh giới thời gian.

### Mục đích từng tập:
- **Train**: Huấn luyện mô hình
- **Validation**: Tinh chỉnh hyperparameters, early stopping, chọn mô hình tốt nhất
- **Test**: Đánh giá cuối cùng (chỉ dùng 1 lần, không dùng để tune)

---

## 5. Pipeline xử lý dữ liệu (Data Processing)

### Phase 1: Khám phá dữ liệu (EDA)
```
Module: modules/eda.py → gọi trong main.ipynb mục 4.2
```
- [ ] Load CSV, kiểm tra shape, dtypes, missing values
- [ ] Thống kê mô tả (mean, std, min, max) cho flow, speed, occupancy **per sensor**
- [ ] Vẽ biểu đồ **time-series flow** cho từng sensor (8 biểu đồ)
- [ ] So sánh distribution flow giữa 8 sensors (boxplot / violin plot)
- [ ] Phân tích **correlation** giữa flow, speed, occupancy
- [ ] Phát hiện pattern: giờ cao điểm, ngày trong tuần, anomaly
- [ ] Kiểm tra **missing data** (timestamp gaps) cho từng sensor
- [ ] Báo cáo: mỗi sensor có bao nhiêu dòng, missing rate bao nhiêu

### Phase 2: Tiền xử lý & Feature Engineering
```
Module: modules/feature_engineering.py → gọi trong main.ipynb mục 4.3
Output:  __datasets-clean/
```
- [ ] Parse `timestamp` → datetime, trích xuất features thời gian:
  - `hour` (0–23)
  - `weekday` (0=Mon → 6=Sun)
  - `is_weekend` (0 hoặc 1)
  - `time_slot` (đêm/sáng/trưa/chiều/tối)
- [ ] Xử lý missing values (nếu có): forward fill hoặc interpolation
- [ ] **Tạo lag features** (bắt buộc group theo `sensor_id`):
  - `flow_lag_1` (t-5min), `flow_lag_2` (t-10min), `flow_lag_3` (t-15min)
  - `speed_lag_1`, `speed_lag_2`, `speed_lag_3`
  - `occupancy_lag_1`, `occupancy_lag_2`, `occupancy_lag_3`
- [ ] Tạo rolling features (window 3, 6, 12 → 15min, 30min, 1h):
  - `flow_rolling_mean_6`, `flow_rolling_std_6`
  - `speed_rolling_mean_6`
- [ ] **Target variable**: `flow_target` = flow tại **t+3** (15 phút sau, 3 bước × 5 phút)
- [ ] Drop rows có NaN do lag/rolling/target shift
- [ ] Chia dữ liệu thành **Baseline A** và **Baseline B** theo sensor groups
- [ ] Mỗi baseline chia train/valid/test theo thời gian (70/15/15)
- [ ] Lưu datasets đã xử lý vào `__datasets-clean/`

---

## 6. Xây dựng mô hình — 7 mô hình cho mỗi Baseline

```
Module: modules/models.py → gọi trong main.ipynb mục 5
```

### 6.1 Features & Target

**Features đầu vào (X):**
```
flow_lag_1, flow_lag_2, flow_lag_3,
speed_lag_1, speed_lag_2, speed_lag_3,
occupancy_lag_1, occupancy_lag_2, occupancy_lag_3,
flow_rolling_mean_6, flow_rolling_std_6, speed_rolling_mean_6,
hour, weekday, is_weekend
```

**Target (y):** `flow_target` (flow sau 15 phút)

### 6.2 Danh sách 7 mô hình

> Tất cả 7 mô hình đều phù hợp cho bài toán **time-series regression** dự báo flow giao thông.

| # | Mô hình | Loại | Lý do phù hợp cho Time Series |
|---|---|---|---|
| 1 | **Linear Regression** | Baseline thống kê | Đơn giản, dễ giải thích, làm baseline so sánh |
| 2 | **Ridge Regression** | Regularized linear | Xử lý tốt multicollinearity giữa lag features |
| 3 | **K-Nearest Neighbors (KNN)** | Instance-based | Tận dụng pattern lặp lại theo giờ/ngày |
| 4 | **Decision Tree Regressor** | Tree-based | Bắt được non-linearity, feature importance |
| 5 | **Random Forest Regressor** | Ensemble (Bagging) | Giảm overfitting, ổn định hơn single tree |
| 6 | **Gradient Boosting (XGBoost)** | Ensemble (Boosting) | SOTA cho tabular data, xử lý tốt lag features |
| 7 | **LSTM (Long Short-Term Memory)** | Deep Learning / RNN | Thiết kế chuyên cho sequence/time series data |

### 6.3 Quy trình cho MỖI Baseline (A và B)

```
Tổng: 2 baselines × 7 mô hình = 14 lần huấn luyện + đánh giá
```

- [ ] Huấn luyện 7 mô hình trên tập **Train**
- [ ] Tune hyperparameters trên tập **Validation**
- [ ] Đánh giá cuối cùng trên tập **Test**
- [ ] Lưu mô hình đã train vào `models/`
- [ ] Ghi nhận metrics cho từng mô hình

### 6.4 Metrics đánh giá

| Metric | Công thức / Mô tả |
|---|---|
| **MAE** | Mean Absolute Error |
| **RMSE** | Root Mean Squared Error |
| **MAPE** | Mean Absolute Percentage Error |
| **R²** | Coefficient of Determination |

---

## 7. So sánh 2 Baselines (Trọng tâm đề bài)

```
Module: modules/comparison.py → gọi trong main.ipynb mục 6
```

> **ĐÂY LÀ PHẦN CỐT LÕI** của đề bài SV16 — so sánh hiệu năng dự báo flow giữa Baseline A và Baseline B trên cùng 7 mô hình.

### Bảng so sánh tổng hợp (ví dụ):

| Mô hình | Baseline A MAE | Baseline A RMSE | Baseline B MAE | Baseline B RMSE | Kết luận |
|---|---|---|---|---|---|
| Linear Regression | ? | ? | ? | ? | A/B tốt hơn |
| Ridge Regression | ? | ? | ? | ? | ... |
| KNN | ? | ? | ? | ? | ... |
| Decision Tree | ? | ? | ? | ? | ... |
| Random Forest | ? | ? | ? | ? | ... |
| XGBoost | ? | ? | ? | ? | ... |
| LSTM | ? | ? | ? | ? | ... |

### Nội dung so sánh:
- [ ] **Bảng tổng hợp metrics** (MAE, RMSE, MAPE, R²) cho mỗi baseline × mỗi mô hình
- [ ] **Biểu đồ so sánh**: grouped bar chart / radar chart so sánh metrics giữa 2 baselines
- [ ] **Biểu đồ Actual vs Predicted** cho mô hình tốt nhất của mỗi baseline
- [ ] **Phân tích sai số theo thời gian**: baseline nào predict tốt hơn vào giờ cao điểm?
- [ ] **Nhận xét & kết luận**:
  - Baseline nào cho kết quả dự báo tốt hơn? Tại sao?
  - Mô hình nào phù hợp nhất cho từng baseline?
  - Yếu tố nào ảnh hưởng đến chất lượng dự báo giữa hai baselines?

---

## 8. Cấu trúc chạy trong `main.ipynb`

> **Toàn bộ logic** nằm trong `modules/`, notebook `main.ipynb` chỉ import và gọi hàm.

```
main.ipynb
│
├── 1. Cài đặt và nhập thư viện
│   ├── 1.1. Cài đặt thư viện (!pip install ...)
│   └── 1.2. Nhập thư viện (import modules/...)
│
├── 2. Kiểm tra GPU
│   └── Kiểm tra CUDA/GPU availability cho LSTM
│
├── 3. Cấu hình (CONFIG)
│   └── Toàn bộ thông số có thể chỉnh sửa:
│       ├── DATA_PATH, OUTPUT_DIR
│       ├── BASELINE_A_SENSORS, BASELINE_B_SENSORS
│       ├── TRAIN_RATIO=0.70, VALID_RATIO=0.15, TEST_RATIO=0.15
│       ├── LAG_STEPS=[1,2,3], ROLLING_WINDOWS=[3,6,12]
│       ├── TARGET_HORIZON=3 (15 phút)
│       ├── RANDOM_STATE=42
│       └── MODEL_PARAMS (hyperparameters cho từng mô hình)
│
├── 4. Xử lý dữ liệu
│   ├── 4.1. Nhập dữ liệu (load CSV, basic info)
│   ├── 4.2. Thống kê mô tả & EDA
│   │   ├── Thống kê per sensor (mean, std, min, max)
│   │   ├── Missing data report
│   │   ├── Time-series plots, distribution plots
│   │   ├── Correlation analysis
│   │   └── Pattern analysis (giờ cao điểm, weekday vs weekend)
│   ├── 4.3. Feature Engineering
│   │   ├── Time features (hour, weekday, is_weekend)
│   │   ├── Lag features (group by sensor_id)
│   │   ├── Rolling features
│   │   └── Target variable (flow_target = t+15min)
│   ├── 4.4. Chia Baseline A & Baseline B
│   └── 4.5. Hold-out Split (Train 70% / Valid 15% / Test 15%)
│
├── 5. Xây dựng mô hình Baseline
│   ├── 5.1. Huấn luyện 7 mô hình cho Baseline A
│   │   ├── Linear Regression
│   │   ├── Ridge Regression
│   │   ├── KNN Regressor
│   │   ├── Decision Tree Regressor
│   │   ├── Random Forest Regressor
│   │   ├── XGBoost Regressor
│   │   └── LSTM
│   ├── 5.2. Huấn luyện 7 mô hình cho Baseline B
│   │   └── (cùng 7 mô hình như trên)
│   └── 5.3. Lưu mô hình vào models/
│
└── 6. So sánh 2 Baseline
    ├── 6.1. Bảng tổng hợp metrics (2 baselines × 7 mô hình)
    ├── 6.2. Biểu đồ so sánh (bar chart, radar chart)
    ├── 6.3. Actual vs Predicted cho mô hình tốt nhất
    ├── 6.4. Phân tích sai số theo thời gian
    └── 6.5. Nhận xét & Kết luận
```

---

## 9. Dashboard — `demo.ipynb`

```
File: demo.ipynb (chạy riêng, load mô hình đã train từ models/)
```

> **MẸO:** Dashboard là "màn hình sản phẩm" — người vận hành chỉ có 30 giây, cần hiển thị info quan trọng nhất trước.

### Yêu cầu Dashboard:
- [ ] **Bộ lọc**: chọn sensor, chọn khoảng thời gian, chọn baseline (A/B)
- [ ] **Biểu đồ dữ liệu gốc**: time-series flow, speed, occupancy cho sensor được chọn
- [ ] **Biểu đồ Actual vs Predicted**: overlay flow thực tế và flow dự báo
- [ ] **Bảng metrics**: RMSE, MAE, MAPE, R² hiển thị rõ ràng
- [ ] **Kết luận tự động**: 
  - 🟢 Bình thường
  - 🟡 Cần theo dõi
  - 🔴 Cảnh báo (tải cao bất thường)
- [ ] **So sánh 2 baselines**: hiển thị song song kết quả Baseline A vs Baseline B

---

## 10. Cấu trúc thư mục dự kiến

```
DoAn_MachineLearning_UTT/
├── .docs/
│   ├── Nhiem-vu-Do-an-Hoc-may.pdf    # Phiếu giao nhiệm vụ
│   ├── plan.md                        # File kế hoạch này
│   ├── Screenshots...                 # Ảnh hướng dẫn
│   └── structure.md                   # Cấu trúc project
├── __datasets-raw/
│   └── SV16_PeMSD3_sample_8sensors.csv  # Dataset gốc (48,385 dòng)
├── __datasets-clean/
│   ├── baselineA_train.csv            # Baseline A - train (70%)
│   ├── baselineA_valid.csv            # Baseline A - valid (15%)
│   ├── baselineA_test.csv             # Baseline A - test  (15%)
│   ├── baselineB_train.csv            # Baseline B - train (70%)
│   ├── baselineB_valid.csv            # Baseline B - valid (15%)
│   └── baselineB_test.csv             # Baseline B - test  (15%)
├── modules/
│   ├── __init__.py                    # Package init
│   ├── data_loader.py                 # Load & parse CSV
│   ├── eda.py                         # Thống kê mô tả, EDA, visualizations
│   ├── feature_engineering.py         # Tạo lag, rolling, time features
│   ├── models.py                      # Định nghĩa, huấn luyện & đánh giá 7 mô hình
│   ├── visualization.py              # Vẽ biểu đồ, charts
│   └── comparison.py                 # So sánh 2 baselines
├── models/                            # Lưu mô hình đã train (.pkl, .joblib, .pth)
├── resultImages/                      # Hình ảnh kết quả (biểu đồ, charts)
├── main.ipynb                         # Notebook chính (phân tích + huấn luyện)
├── demo.ipynb                         # Dashboard / Demo trực quan
└── requirements.txt
```

---

## 11. Thư viện cần sử dụng

```txt
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
torch>=2.0               # Cho LSTM
matplotlib>=3.7
seaborn>=0.12
plotly>=5.15
ipywidgets>=8.0           # Cho dashboard trong notebook
joblib                    # Lưu mô hình
```

---

## 12. Timeline thực hiện (gợi ý)

| Tuần | Công việc | Modules liên quan |
|---|---|---|
| **Tuần 1** | EDA + Feature Engineering + Chia dữ liệu | `data_loader.py`, `eda.py`, `feature_engineering.py` |
| **Tuần 2** | Huấn luyện 7 mô hình cho Baseline A & B | `models.py` |
| **Tuần 3** | So sánh 2 baselines + Biểu đồ phân tích | `comparison.py`, `visualization.py` |
| **Tuần 4** | Dashboard (`demo.ipynb`) + Hoàn thiện báo cáo | `demo.ipynb`, `main.ipynb` |

---

## 13. Checklist tổng hợp

### Phân tích dữ liệu
- [ ] EDA đầy đủ cho 8 sensors
- [ ] Xác định cách chia 2 baselines (có giải thích lý do)
- [ ] Báo cáo missing data rate cho mỗi sensor

### Xây dựng mô hình
- [ ] 7 mô hình cho Baseline A (LR, Ridge, KNN, DT, RF, XGB, LSTM)
- [ ] 7 mô hình cho Baseline B (cùng 7 mô hình)
- [ ] Feature engineering đầy đủ (lag + time + rolling)
- [ ] Hold-out Split đúng: Train 70% / Valid 15% / Test 15% (theo thời gian)

### Đánh giá & So sánh
- [ ] Bảng tổng hợp metrics cho 2 baselines × 7 mô hình (= 14 dòng)
- [ ] Biểu đồ so sánh trực quan
- [ ] Nhận xét, kết luận chi tiết về sự khác biệt giữa 2 baselines

### Sản phẩm
- [ ] `main.ipynb` chạy được end-to-end
- [ ] `demo.ipynb` Dashboard hoạt động (chọn sensor, xem kết quả)
- [ ] Biểu đồ Actual vs Predicted
- [ ] Hệ thống cảnh báo (bình thường / theo dõi / cảnh báo)

---

> **⚠️ LƯU Ý QUAN TRỌNG TỪ HƯỚNG DẪN:**
> - Khi tạo lag features, **bắt buộc** group theo `sensor_id` (không lag chéo sensor)
> - Chia dữ liệu **theo thời gian** (Hold-out Split, KHÔNG shuffle) → Train 70% / Valid 15% / Test 15%
> - Cần báo cáo rõ: mỗi sensor có bao nhiêu dòng, missing rate bao nhiêu
> - Dashboard (`demo.ipynb`) cần hiển thị thông tin quan trọng **ngay đầu tiên** (người vận hành chỉ có 30 giây)
> - Toàn bộ logic nằm trong `modules/`, notebook chỉ import và gọi hàm
> - Mọi thông số cấu hình đều nằm tập trung ở **mục 3. Cấu hình** trong `main.ipynb`
