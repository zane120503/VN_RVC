# 📁 HƯỚNG DẪN ĐẶT FILE VÀO DATASET

## Cách đơn giản nhất (Không cần checkbox):

### Bước 1: Tạo thư mục cho dataset

1. Mở File Explorer
2. Đi đến thư mục: `D:\Vietnamese-RVC\dataset\`
3. Tạo thư mục mới với tên mô hình của bạn (ví dụ: `my_voice`)
   - Tên thư mục = Tên mô hình bạn sẽ nhập trong giao diện
   - Ví dụ: Nếu bạn muốn đặt tên mô hình là `my_voice`, tạo thư mục `dataset\my_voice\`

### Bước 2: Copy file âm thanh vào thư mục

1. Copy TẤT CẢ file âm thanh của bạn vào thư mục vừa tạo
   - Ví dụ: Copy vào `D:\Vietnamese-RVC\dataset\my_voice\`
2. File có thể là: wav, mp3, flac, m4a, v.v.

### Bước 3: Sử dụng trong giao diện

1. Mở giao diện Vietnamese-RVC
2. Vào tab **"Huấn Luyện Mô Hình"** → **"Huấn Luyện Mô Hình"**
3. Nhập **Tên của mô hình** = Tên thư mục bạn vừa tạo (ví dụ: `my_voice`)
4. Chọn các cài đặt:
   - Tỉ lệ lấy mẫu: `48k`
   - Phiên bản: `v2`
   - ✅ Bật "Huấn luyện cao độ"
5. Nhấn **"1. Xử lí dữ liệu"**

### Lưu ý:
- Tên thư mục phải KHỚP với tên mô hình bạn nhập
- Ví dụ: Thư mục `dataset\my_voice\` → Tên mô hình: `my_voice`
- Không dùng ký tự đặc biệt hay dấu cách trong tên

---

## Cách 2: Sử dụng checkbox "Tùy chỉnh dataset" (Nếu muốn dùng thư mục khác)

1. Trong giao diện, mở phần **"Cài đặt chung"** (accordion bên phải)
2. ✅ Bật checkbox **"Tùy chỉnh dataset"**
3. Nhập đường dẫn thư mục chứa file (ví dụ: `dataset\my_voice`)
4. Tiếp tục các bước như trên

---

## Ví dụ cụ thể:

**Giả sử bạn có các file:**
- `voice1.wav`
- `voice2.wav`
- `voice3.mp3`

**Các bước:**
1. Tạo thư mục: `D:\Vietnamese-RVC\dataset\my_voice\`
2. Copy 3 file vào: 
   - `D:\Vietnamese-RVC\dataset\my_voice\voice1.wav`
   - `D:\Vietnamese-RVC\dataset\my_voice\voice2.wav`
   - `D:\Vietnamese-RVC\dataset\my_voice\voice3.mp3`
3. Trong giao diện:
   - Tên mô hình: `my_voice`
   - Nhấn "1. Xử lí dữ liệu"

Done! ✅

