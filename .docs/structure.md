```
DoAn_MachineLearning_UTT/
├── .docs/
│   ├── Nhiem-vu-Do-an-Hoc-may.pdf    # Phiếu giao nhiệm vụ
│   ├── plan.md                        # File kế hoạch
│   ├── workflow.md                    # Luồng xử lý chi tiết
│   ├── structure.md                   # Cấu trúc project (file này)
│   ├── Buoc11.jpg                     # Hướng dẫn Monte Carlo stability
│   ├── Buoc12-1.jpg                   # Hướng dẫn time sliding stability
│   ├── Buoc12-2.jpg                   # Sơ đồ train 60%, test 10%, shift 5%
│   ├── Buoc12-3.jpg                   # Ví dụ đồ thị RMSE qua 5 fold
│   └── Screenshots...                 # Ảnh hướng dẫn ban đầu
├── __datasets-raw/
│   └── SV16_PeMSD3_sample_8sensors.csv  # Dataset gốc (48,385 dòng)
├── __datasets-clean/
│   ├── baselineA_train.csv            # Baseline A - train (70%)
│   ├── baselineA_valid.csv            # Baseline A - valid (15%)
│   ├── baselineA_test.csv             # Baseline A - test  (15%)
│   ├── baselineB_train.csv            # Baseline B - train (70%)
│   ├── baselineB_valid.csv            # Baseline B - valid (15%)
│   └── baselineB_test.csv             # Baseline B - test  (15%)
├── configs/
│   ├── configs.py                     # CONFIG, OPTIM_CONFIG, STABILITY_CONFIG
│   └── pymooSearchSpaces.py           # Search spaces cho Pymoo (chứa lambda → dùng .py)
├── modules/
│   ├── __init__.py                    # Package init
│   ├── data_loader.py                 # Load & parse CSV
│   ├── eda.py                         # Thống kê mô tả, EDA, visualizations
│   ├── feature_engineering.py         # Tạo lag, rolling, time features
│   ├── models.py                      # Định nghĩa, huấn luyện & đánh giá 10 mô hình (3 trivial + 7 ML)
│   ├── visualization.py              # Vẽ biểu đồ, charts
│   ├── comparison.py                 # So sánh 2 baselines
│   ├── optimization.py               # Tối ưu hóa đa mục tiêu (Pymoo NSGA-II/III)
│   ├── stability.py                  # Monte Carlo + trượt thời gian + chọn mô hình cuối
│   └── gan_synthetic.py              # Sinh dữ liệu ảo bằng GAN trong không gian feature
├── pymooCheckpoint/                   # Checkpoint lưu tiến trình tối ưu Pymoo (tự tạo)
│   ├── optim_checkpoint_A_5_RandomForest.pkl
│   ├── optim_checkpoint_A_6_XGBoost.pkl
│   ├── optim_checkpoint_B_5_RandomForest.pkl
│   └── optim_checkpoint_B_6_XGBoost.pkl
├── models/                            # Lưu mô hình đã train (.pkl, .joblib, .pth)
├── resultImages/                      # Hình ảnh kết quả (EDA, compare, Pareto, stability)
│   ├── monte_carlo_rmse_boxplot.png   # Boxplot RMSE Monte Carlo
│   ├── monte_carlo_rmse_kde.png       # Phân phối tần suất + KDE
│   ├── monte_carlo_rmse_ci95.png      # Khoảng tin cậy 95%
│   ├── time_sliding_rmse.png          # RMSE qua 5 cửa sổ trượt thời gian
│   └── gan_synthetic_saturation.png   # Điểm bão hòa khi tăng synthetic data
├── main.ipynb                         # Notebook chính (phân tích + huấn luyện)
├── demo.ipynb                         # Dashboard / Demo trực quan
└── requirements.txt
```
