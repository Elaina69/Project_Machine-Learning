# SV16 Traffic Dashboard

Dashboard web app một trang cho 4 mô hình baseline tối ưu:

- `Baseline_A | 5_RandomForest`
- `Baseline_A | 6_XGBoost`
- `Baseline_B | 5_RandomForest`
- `Baseline_B | 6_XGBoost`

Dashboard không load các model GAN và không huấn luyện lại mô hình. Backend chỉ đọc artifact đã sinh bởi `main.ipynb`.

## Chạy Dev

Từ thư mục gốc dự án:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\dashboard\dev.ps1
```

Script này sẽ bật backend trước, đợi `/api/health` sẵn sàng, sau đó mới bật frontend Vite. Nếu `5173` đang bận bởi app khác, Vite sẽ tự chọn cổng kế tiếp; hãy mở đúng URL mà Vite in ra trong terminal. Đây là cách khuyến nghị để tránh lỗi frontend proxy gọi API khi backend chưa chạy.

Nếu muốn chạy thủ công 2 terminal, mở terminal 1:

```powershell
.\.venv\Scripts\python.exe -m dashboard.backend.run --strict-port
```

Mở terminal 2:

```powershell
cd dashboard\frontend
npm install
npm run dev
```

Truy cập `http://127.0.0.1:5173`.

Nếu mở `127.0.0.1:5173` mà thấy giao diện khác, nghĩa là cổng này đang thuộc một Vite app khác. Khi đó hãy dùng URL mới mà `.\dashboard\dev.ps1` hoặc `npm run dev` in ra, ví dụ `http://127.0.0.1:5175`.

Backend mặc định dùng `http://127.0.0.1:8010`. Không dùng cổng `8000` vì trên Windows cổng này dễ bị process khác hoặc dải cổng bị hệ thống giữ quyền, gây lỗi `WinError 10013`. Nếu `8010` cũng bận, runner sẽ tự chọn cổng kế tiếp và in ra biến cần đặt cho frontend:

```powershell
$env:VITE_BACKEND_URL="http://127.0.0.1:<PORT_DUOC_IN_RA>"
npm run dev
```

Nếu vẫn muốn chạy trực tiếp bằng uvicorn, hãy chỉ định cổng:

```powershell
.\.venv\Scripts\python.exe -m uvicorn dashboard.backend.app:app --reload --host 127.0.0.1 --port 8010
```

Nếu Vite báo `connect ECONNREFUSED 127.0.0.1:8010`, nghĩa là frontend đang chạy nhưng backend chưa lắng nghe tại cổng `8010`. Hãy chạy backend trước hoặc dừng Vite rồi chạy lại bằng `.\dashboard\dev.ps1`. Sau khi sửa `vite.config.js`, cần restart Vite để proxy mới có hiệu lực.

## Chạy Một URL

Build frontend rồi để FastAPI serve lại:

```powershell
cd dashboard\frontend
npm install
npm run build
cd ..\..
.\.venv\Scripts\python.exe -m dashboard.backend.run
```

Truy cập URL backend được in ra, mặc định là `http://127.0.0.1:8010`.
