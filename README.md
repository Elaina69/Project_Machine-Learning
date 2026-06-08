# Đồ án Học máy — SV16

Bài toán: so sánh dự báo `flow` giữa hai cụm sensor PeMSD3 bằng 10 mô hình cho mỗi baseline, tối ưu siêu tham số bằng Pymoo và kiểm định độ ổn định sau tối ưu.

Luồng chính nằm trong `main.ipynb`:

1. EDA và feature engineering cho dữ liệu time series.
2. Chia Baseline A/B theo sensor và hold-out split theo thời gian.
3. Huấn luyện 3 trivial baselines + 7 mô hình ML.
4. So sánh hai baseline bằng MAE, RMSE, MAPE, R², MASE.
5. Tối ưu 2 mô hình tốt nhất bằng Pymoo với 3 mục tiêu: RMSE, thời gian tìm kiếm, độ phức tạp.
6. Chạy Monte Carlo cho 4 mô hình tối ưu, giữ split thời gian cố định.
7. Kiểm định ổn định dữ liệu bằng 5 cửa sổ trượt thời gian: train 60%, test 10%, shift 5%.
8. Chọn mô hình cuối bằng scorecard tổng hợp và mở rộng sinh dữ liệu bằng feature-space GAN.

Tài liệu chi tiết nằm trong `.docs/plan.md`, `.docs/workflow.md` và `.docs/structure.md`.
