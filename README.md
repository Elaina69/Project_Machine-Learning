# Đồ án Học máy — SV16

Bài toán: so sánh dự báo `flow` giữa hai cụm sensor PeMSD3 bằng 10 mô hình cho mỗi baseline, tối ưu siêu tham số bằng Pymoo và kiểm định độ ổn định sau tối ưu.

Luồng chính nằm trong `main.ipynb`:

1. EDA và feature engineering cho dữ liệu time series.
2. Chia Baseline A/B theo sensor và hold-out split theo thời gian.
3. Huấn luyện 3 trivial baselines + 7 mô hình ML.
4. So sánh hai baseline bằng MAE, RMSE, MAPE, R², MASE.
5. Phân tích lỗi dự báo: worst cases, lỗi theo sensor/hour/flow regime, nguyên nhân và hướng cải thiện.
6. Tối ưu 2 mô hình tốt nhất bằng Pymoo với 3 mục tiêu: RMSE, thời gian tìm kiếm, độ phức tạp.
7. Chạy Monte Carlo cho 4 mô hình tối ưu, giữ split thời gian cố định.
8. Kiểm định ổn định dữ liệu bằng 5 cửa sổ trượt thời gian: train 60%, test 10%, shift 5%.
9. Chọn mô hình cuối bằng scorecard tổng hợp.
10. Dùng SHAP top 14 + `other` để giải thích độ nhạy cho 4 mô hình tối ưu.
11. Mở rộng sinh dữ liệu bằng feature-space GAN cho 4 mô hình tối ưu và so sánh XAI sau GAN.

Dashboard/demo mới nằm trong `dashboard/`:

- Backend: FastAPI, chỉ load 4 mô hình baseline tối ưu ở `models/Optimized`.
- Frontend: React single-page app, hiển thị dữ liệu gốc, dự báo, metrics, SHAP và cảnh báo/khuyến nghị.
- Chạy nhanh: xem `dashboard/README.md`.

Tài liệu chi tiết nằm trong `.docs/plan.md`, `.docs/workflow.md` và `.docs/structure.md`.
