# Promt 1:
- Đọc nhiệm vụ của sinh viên 16 (SV16) trong "Nhiem-vu-Do-an-Hoc-may.pdf", các hướng dẫn dể làm đồ án trong "Nhiem-vu-Do-an-Hoc-may.pdf" "Screenshot 2026-05-09 143141.png" "Screenshot 2026-05-09 144230.png" "Screenshot 2026-05-09 150310.png"
- Sau đó phân tích data gốc trong file "SV16_PeMSD3_sample_8sensors.csv" để hiểu mục đích của Đồ án Học máy này.
- Sau đó hãy xây dựng 1 kế hoạch, file kế hoạch "plan.md" nằm trong folder .docs về việc xây dựng bài toán "So sánh dự báo flow giữa hai tập sensor trong PeMSD3" phục vụ cho đồ án học máy

# Promt 2: 
chỉnh sửa kế hoạch:
- các file .ipynb trong folder notebook sẽ chuyển thành các modules và nhập vào trong main.ipynb để chạy
- về demo/dashboard, chạy trong file demo.ipynb
- trong nhiệm vụ đồ án: "So sánh ít nhất 02 baseline với tối thiểu 05 mô hình học máy; khuyến khích mở rộng lên 7-10 mô hình.", chỉnh sửa lại kế hoạch để so sánh 2 baseline, mỗi baseline là 1 cụm 4 sensor chạy với 7 mô hình học máy phù hợp với bài toán time series. Ghi rõ 7 mô hình sẽ dùng cho 2 baseline là gì
- vì đây là 1 bài toán time series, cần phải chia dữ liệu theo trình tự thời gian (Hold-out Split), train 70%, valid 15%, test 15% để tránh rò rỉ dữ liệu
- Các bước chạy trong main.ipynb:

1. Cài đặt và nhập thư viện
1.1. Cài đặt thư viện
1.2. Nhập thư viện
2. Kiểm tra GPU
3. Cấu hình (toàn bộ thông số có thể chỉnh sửa được trong bài đều năm ở mục này)
4. Xử lý dữ liệu
4.1. Nhập dữ liệu
4.2. Thống kê mô tả
4.3. ...
5. Xây dựng mô hình baseline
6. So sánh 2 mô hình baseline

# Promt 3:
- Dựa vào file "plan.md" đã tạo, hãy tiến hành làm đồ án này
- Giải thích chi tiết workflow của code hiện tại vào file "workflow.md"

# Promt 4:
- Cập nhật hàm "update_prediction_plot()", cho phép hiện avg_flow để dễ dàng theo dõi dòng xe

# Promt 5:
Cập nhật code "main.ipynb" và các tài liệu "workflow.md" "structure.md" "plan.md"
- Tạo thêm đặc trưng 'lag_6', 'lag_12', 'occupancy_roll_mean_3', 'occupancy_roll_std_3', 'occupancy_roll_mean_6', 'occupancy_roll_std_6', 'occupancy_roll_mean_12', 'occupancy_roll_std_12'
- Huấn luyện thêm 3 mô hình trivial baseline Seasonal Naive, Drift Method, Simple Moving Average

# Promt 6:
Cập nhật code "main.ipynb" và các tài liệu "workflow.md" "structure.md" "plan.md"
- thêm tiêu chí đánh giá MASE 
- Chọn tiêu chí đánh giá RMSE là tiêu chí chính
- Sắp xếp bảng tổng hợp sau khi huấn luyện baseline từ cao đến thấp theo tiêu chí RMSE
- Trong bảng tổng hợp sau khi huấn luyện baseline , phải hiển thị đầy đủ thông tin toàn bộ các tiêu chí của train, valid, test
- Sau khi hoàn thành so sánh 2 baseline (Mục 6), chọn ra 2 mô hình tốt nhất ở cả 2 baseline (mô hình này bắt buộc phải có điểm đánh giá tốt trên cả 2 baseline) để tối ưu siêu tham số, thiết lập bài toán tối ưu hóa đa mục tiêu (Pymoo problem) với 3 tiêu chí mục tiêu:
    + Mục tiêu 1: RMSE
    + Mục tiêu 2: Search time (thời gian huấn luyện/tìm kiếm tham số)
    + Mục tiêu 3: Độ phức tạp của mô hình (Model complexity)
- Để bao phủ tốt không gian 3 mục tiêu trên, đề xuất:
    + Thuật toán: NSGA-II hoặc NSGA-III tùy theo độ phức tạp
    + Kích thước quần thể từ 200 đến 400
    + Số thế hệ (n_gen) từ 100 đến 200
- Sau đó thực thi tối ưu hóa và tìm tập Pareto, chạy thuật toán và tìm ra các tập hợp các nghiệm tối ưu. Đồng thời phân tích sự đánh đổi (Ví dụ: muốn giảm thêm RMSE thì cần phải đánh đổi những gì)

# Promt 7:
Đọc lại toàn bộ code `main.ipynb`, `modules/` và tài liệu `.docs/`; cập nhật code dựa vào hướng dẫn `Buoc11.jpg`, `Buoc12-1.jpg`, `Buoc12-2.jpg`, `Buoc12-3.jpg`:
- Thêm mô phỏng Monte Carlo cho 4 mô hình tối ưu sau Pymoo (2 mô hình tốt nhất × Baseline A/B).
- Monte Carlo giữ nguyên split time series, chỉ thay đổi `random_state` của thuật toán/model.
- Thêm trực quan hóa boxplot, histogram/KDE và khoảng tin cậy 95%.
- Chứng minh độ ổn định bộ dữ liệu bằng kỹ thuật trượt thời gian: train 60%, test 10%, shift 5%, 5 fold.
- Cập nhật lại toàn bộ tài liệu để phù hợp với code hiện tại.
