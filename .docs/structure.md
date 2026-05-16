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