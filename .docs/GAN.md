# Nội dung trình chiếu: Áp dụng GAN và Đào Tạo Mô Hình Tổng Quát

## Các giai đoạn chính

**1. Sinh Dữ Liệu**
Huấn luyện mạng GAN trên tập dữ liệu giao thông gốc để sinh ra tập dữ liệu mô phỏng có chung phân phối thống kê.

**2. Đánh Giá Chéo**
Kiểm chứng độ tin cậy thông qua kịch bản Train Real/Test Fake và ngược lại để đo lường độ lệch (bias).

**3. Tích Hợp Mô Hình**
Đào tạo mô hình tổng quát hóa dựa trên kỹ thuật gộp không gian mẫu để tránh xung đột trục thời gian.

---

## Bước 1: Huấn luyện & Sinh Dữ Liệu

* **Sử dụng dữ liệu gốc:** Lấy bộ dữ liệu đã được làm sạch và xử lý đặc trưng (biến trễ, ngoại sinh) từ Bài toán 1.
* **Cấu hình mô hình GAN:** Ứng dụng các kiến trúc phù hợp như Time-series GAN (TGAN/TimeGAN).
* **Quá trình hội tụ:** Đào tạo Mạng phân biệt và Mạng sinh cho đến khi Mạng phân biệt không phân biệt được thật/giả.
* **Sinh dữ liệu ảo:** Khởi tạo tập dữ liệu tổng hợp (synthetic data) duy trì được các đặc trưng tự tương quan.

### Chiến Lược Xác Định Kích Thước dữ liệu ảo

**1. Tỉ lệ 1:1 (Bằng gốc)**
Tối ưu cho bước đánh giá chéo. Việc có dung lượng mẫu bằng nhau giúp so sánh sai số giữa Train Real/Test Fake và Train Fake/Test Real công bằng, không bị nhiễu do chênh lệch.

**2. Tỉ lệ 1:N (Mở rộng)**
Phục vụ tăng cường dữ liệu. Sinh lượng dữ liệu khổng lồ (gấp 2, 5, 10 lần) giúp mô hình tổng quát học được nhiều kịch bản đa dạng hơn, giảm thiểu nguy cơ Overfitting hiệu quả.

**3. Sinh có chủ đích**
Giải quyết mất cân bằng dữ liệu. Chỉ tập trung sinh thêm dữ liệu cho các trường hợp hiếm (ví dụ: ùn tắc nghiêm trọng) để cân bằng phân phối, không cần sinh lại toàn bộ chuỗi.

---

## Bước 2: Đánh Giá Chéo Chất Lượng Dữ Liệu

**Kịch Bản A: Train Real / Test Fake**
* **Mục tiêu:** Xác nhận dữ liệu ảo tuân theo đúng quy luật vật lý của dữ liệu thực.
* **Thực thi:** Đào tạo mô hình bằng bộ dữ liệu gốc, sau đó dùng nó để dự báo trên bộ dữ liệu ảo do GAN sinh ra. Sai số thấp chứng tỏ phân phối đã được sao chép tốt.

**Kịch Bản B: Train Fake / Test Real**
* **Mục tiêu:** Khẳng định dữ liệu ảo mang lại giá trị học tập thực tiễn (utility).
* **Thực thi:** Đào tạo mô hình hoàn toàn bằng bộ dữ liệu ảo, sau đó kiểm thử trên tập dữ liệu gốc. Phân tích chênh lệch hiệu suất giữa hai kịch bản để đánh giá toàn diện.

---

## Bước 3: Đào Tạo Mô Hình Tổng Quát

Thách thức lớn nhất trong Time Series là việc "gộp" dữ liệu không được làm phá vỡ trục thời gian thực. Cần áp dụng kỹ thuật gộp theo Không gian mẫu (Sample Space).

### 3 Chiến Lược Gộp Dữ Liệu Chuỗi Thời Gian

**1. Cửa Sổ Trượt (Sliding Window)**
Biến đổi chuỗi thành các mẫu (samples) học giám sát độc lập mang đặc trưng trễ trước khi concat (ghép dòng).

**2. Dữ Liệu Đa Chuỗi (Panel Data)**
Xử lý dữ liệu đầu vào dưới dạng Tensor 3D cho mạng nơ-ron, coi mỗi chuỗi GAN sinh ra là một kịch bản giao thông riêng biệt.

**3. Học Chuyển Giao (Transfer Learning)**
An toàn nhất: Pre-train mô hình bằng lượng lớn dữ liệu ảo từ GAN, sau đó Fine-tune lại bằng tập dữ liệu gốc để giữ độ chính xác.

---

## Bước 4: Tinh Chỉnh Siêu Tham Số

**Tối Ưu Hóa Nâng Cao**
* Kế thừa thuật toán tối ưu tiến hóa đa mục tiêu (như Pymoo) từ bài toán 1 để áp dụng cho mô hình tổng quát mới.
* Do cấu trúc không gian dữ liệu đã mở rộng (có thêm dữ liệu ảo), bộ siêu tham số (hyperparameters) cũ có thể không còn phù hợp. Việc tìm kiếm không gian tham số mới là bắt buộc để mô hình đạt hiệu năng hội tụ tối đa.

---

## Bước 5: Giải Thích & Độ Nhạy (XAI)

* **Minh bạch hóa AI (Explainable AI):** Sử dụng SHAP để bóc tách và định lượng mức độ đóng góp (feature importance) của từng biến đầu vào.
* **Phân tích độ nhạy bằng SHAP:** Trực quan hóa top feature ảnh hưởng mạnh nhất và gộp các feature còn lại vào nhóm `other`.
* **Phân tích đối chiếu:** So sánh sự thay đổi về tầm quan trọng của các đặc trưng giữa mô hình gốc (Bài toán 1) và mô hình tổng quát (Bài toán 2).

---

## Yêu Cầu Mở Rộng (Không bắt buộc)

Để bài báo cáo đạt điểm tối đa, sinh viên cần thực hiện phân tích Điểm Bão Hòa của dữ liệu ảo:
* **Kịch bản 1:** Huấn luyện với 100% Gốc + 50% Ảo
* **Kịch bản 2:** Huấn luyện với 100% Gốc + 100% Ảo
* **Kịch bản 3:** Huấn luyện với 100% Gốc + 200% Ảo
* **Đánh giá:** Vẽ biểu đồ theo dõi hiệu năng (RMSE/MAE). Xác định ngưỡng nào mô hình tốt nhất, ngưỡng nào sinh thêm dữ liệu không còn tác dụng.
