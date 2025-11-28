
# Plant Disease Detection (Apple and Corn)

Dự án này sử dụng học máy để phát hiện bệnh trên cây táo và ngô dựa trên hình ảnh của lá cây. Mô hình học sâu (deep learning) được huấn luyện với các loại bệnh phổ biến trên táo và ngô, sau đó ứng dụng vào một ứng dụng web để nhận dạng bệnh qua ảnh.

## Mô tả

Dự án này bao gồm một ứng dụng Flask đơn giản với các chức năng sau:

- Nhận ảnh của cây táo hoặc ngô từ người dùng.
- Dự đoán bệnh cây dựa trên ảnh.
- Cung cấp kết quả dự đoán cùng với độ tin cậy của mô hình.

Ứng dụng sử dụng một mô hình học sâu (deep learning) được huấn luyện sẵn, có thể nhận dạng các bệnh sau:

- **Táo**: Apple Scab, Healthy
- **Ngô**: Common Rust, Healthy

## Cài đặt

Để cài đặt và chạy ứng dụng này trên máy của bạn, làm theo các bước sau:

### Bước 1: Clone repository

```bash
git clone https://github.com/Erwinpro23/Plant-Disease-Detection-apple-and-Corn-
cd Plant-Disease-Detection-apple-and-Corn-
```

### Bước 2: Cài đặt các thư viện yêu cầu

Cài đặt các thư viện cần thiết từ file `requirements.txt`:

```bash
pip install -r requirements.txt
```

Nếu file `requirements.txt` chưa có, bạn có thể cài đặt các thư viện cần thiết thủ công:

```bash
pip install Flask tensorflow pillow opencv-python
```

### Bước 3: Đảm bảo rằng bạn có mô hình

Dự án này yêu cầu mô hình học máy được lưu trong file `best_b6_model.h5`. Bạn cần tải mô hình này từ nguồn đã cho hoặc huấn luyện mô hình của riêng bạn.

- Nếu bạn chưa có mô hình, bạn có thể huấn luyện lại mô hình bằng cách sử dụng **`Model.ipynb`**. Đảm bảo rằng notebook này chạy đúng và tạo ra mô hình `.h5` cuối cùng.

### Bước 4: Chạy ứng dụng Flask

```bash
python app.py
```

Ứng dụng sẽ chạy trên `http://localhost:5000/`.

## Cách sử dụng

1. Truy cập ứng dụng web tại `http://localhost:5000/`.
2. Tải lên một hình ảnh của lá cây táo hoặc ngô.
3. Ứng dụng sẽ trả về dự đoán bệnh cùng với độ tin cậy.

### Dự đoán

- **prediction**: Tên bệnh hoặc trạng thái của cây (ví dụ: "Apple__Apple_scab", "Corn__healthy").
- **confidence**: Độ tin cậy của dự đoán (ví dụ: 0.85).
- **all_predictions**: Dự đoán cho tất cả các lớp (bệnh và trạng thái cây).

## Cấu trúc thư mục

```
Plant-Disease-Detection-apple-and-Corn/
│
├── app.py                # File ứng dụng Flask
├── Model.ipynb           # Jupyter Notebook để huấn luyện mô hình
├── templates/            # Thư mục chứa file HTML (index.html)
│   └── index.html        # Template chính của ứng dụng
├── best_b6_model.h5      # Mô hình học máy (nếu có sẵn)
└── requirements.txt      # Các thư viện yêu cầu
```

## Lưu ý

- Nếu không có mô hình `best_b6_model.h5`, ứng dụng sẽ trả về dự đoán ngẫu nhiên. Đảm bảo rằng mô hình này được đặt đúng vị trí trong thư mục gốc của dự án.
- Nếu bạn sử dụng **`Model.ipynb`** để huấn luyện mô hình, hãy đảm bảo rằng notebook này chạy đúng và tạo ra file mô hình `.h5`.
- Để đảm bảo môi trường Python phù hợp, bạn có thể tạo một môi trường ảo (`virtualenv`) và cài đặt các thư viện từ `requirements.txt`.
- Ứng dụng yêu cầu một số thư viện như Flask, TensorFlow, Pillow, OpenCV để hoạt động chính xác.

## License
### Những điểm bổ sung trong bản README này:
1. **Hướng dẫn huấn luyện mô hình**: Thêm chi tiết về việc sử dụng `Model.ipynb` để huấn luyện mô hình và tạo file `.h5`.
2. **Các thư viện yêu cầu**: Đảm bảo người dùng biết cần cài đặt gì thông qua `requirements.txt` hoặc danh sách thủ công.
3. **Lưu ý**: Đảm bảo rằng mô hình `.h5` có sẵn hoặc người dùng có thể tự huấn luyện nếu cần.
4. **Cấu trúc thư mục**: Cung cấp rõ ràng cấu trúc thư mục và nơi chứa các file quan trọng.
