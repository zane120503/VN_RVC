# 🚀 HƯỚNG DẪN CHẠY VIETNAMESE-RVC TRÊN GOOGLE COLAB

Hướng dẫn chi tiết để chạy Vietnamese-RVC trên Google Colab, tận dụng GPU miễn phí của Google.

---

## 📋 MỤC LỤC

1. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
2. [Bước 1: Tạo Notebook mới trên Colab](#bước-1-tạo-notebook-mới-trên-colab)
3. [Bước 2: Cài đặt môi trường](#bước-2-cài-đặt-môi-trường)
4. [Bước 3: Clone repository](#bước-3-clone-repository)
5. [Bước 4: Cài đặt dependencies](#bước-4-cài-đặt-dependencies)
6. [Bước 5: Khởi động ứng dụng](#bước-5-khởi-động-ứng-dụng)
7. [Bước 6: Truy cập giao diện](#bước-6-truy-cập-giao-diện)
8. [Lưu ý quan trọng](#lưu-ý-quan-trọng)
9. [Xử lý lỗi thường gặp](#xử-lý-lỗi-thường-gặp)
10. [Tối ưu hóa cho Colab](#tối-ưu-hóa-cho-colab)

---

## 🎯 YÊU CẦU HỆ THỐNG

- Tài khoản Google (miễn phí)
- Kết nối internet ổn định
- Trình duyệt web (Chrome, Firefox, Edge, Safari)

**Lưu ý:** Google Colab cung cấp GPU miễn phí nhưng có giới hạn:
- Phiên làm việc có thể bị ngắt sau 12 giờ không hoạt động
- GPU miễn phí có thể bị giới hạn thời gian sử dụng
- Dữ liệu sẽ bị xóa khi phiên kết thúc (trừ khi lưu vào Google Drive)

---

## 📝 BƯỚC 1: TẠO NOTEBOOK MỚI TRÊN COLAB

1. Truy cập [Google Colab](https://colab.research.google.com/)
2. Đăng nhập bằng tài khoản Google của bạn
3. Tạo notebook mới:
   - Nhấn **"File"** → **"New notebook"**
   - Hoặc sử dụng notebook có sẵn: [Vietnamese-RVC Colab](https://colab.research.google.com/github/PhamHuynhAnh16/Vietnamese-RVC-ipynb/blob/main/Vietnamese-RVC.ipynb)

---

## ⚙️ BƯỚC 2: CÀI ĐẶT MÔI TRƯỜNG

### 2.1. Kích hoạt GPU

Trong Colab, chọn GPU để tăng tốc xử lý:

1. Nhấn **"Runtime"** → **"Change runtime type"**
2. Chọn:
   - **Hardware accelerator:** `GPU`
   - **GPU type:** `T4` (miễn phí) hoặc `A100` (nếu có Pro/Pro+)
3. Nhấn **"Save"**

### 2.2. Kiểm tra GPU

Chạy ô code sau để kiểm tra GPU:

```python
!nvidia-smi
```

Bạn sẽ thấy thông tin GPU như:
- **GPU:** Tesla T4, V100, hoặc A100
- **Memory:** VRAM khả dụng

---

## 📥 BƯỚC 3: CLONE REPOSITORY

Chạy lệnh sau để clone repository Vietnamese-RVC:

```python
# Clone repository
!git clone https://github.com/PhamHuynhAnh16/Vietnamese-RVC.git

# Di chuyển vào thư mục dự án
%cd Vietnamese-RVC
```

**Lưu ý:** 
- Nếu repository đã tồn tại, có thể cần xóa và clone lại:
```python
!rm -rf Vietnamese-RVC
!git clone https://github.com/PhamHuynhAnh16/Vietnamese-RVC.git
%cd Vietnamese-RVC
```

---

## 📦 BƯỚC 4: CÀI ĐẶT DEPENDENCIES

### 4.1. Cài đặt Python packages cơ bản

```python
# Cài đặt pip và các công cụ cơ bản
!python -m pip install --upgrade pip
!pip install wheel
```

### 4.2. Cài đặt PyTorch với CUDA

Colab thường có CUDA sẵn, cài đặt PyTorch tương thích:

```python
# Kiểm tra phiên bản CUDA
!nvcc --version

# Cài đặt PyTorch với CUDA (thường là cu118 hoặc cu121)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Hoặc nếu CUDA 12.1:
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4.3. Cài đặt các dependencies từ requirements.txt

```python
# Cài đặt tất cả dependencies
!pip install -r requirements.txt
```

**Lưu ý:** Quá trình này có thể mất 5-10 phút tùy tốc độ internet.

### 4.4. Cài đặt FFmpeg (nếu cần)

FFmpeg thường đã có sẵn trên Colab, nhưng nếu thiếu:

```python
# Kiểm tra FFmpeg
!ffmpeg -version

# Nếu thiếu, cài đặt:
# !apt-get update
# !apt-get install -y ffmpeg
```

---

## 🚀 BƯỚC 5: KHỞI ĐỘNG ỨNG DỤNG

### 5.1. Chạy ứng dụng với Gradio

```python
# Khởi động ứng dụng
!python main/app/app.py --share
```

**Giải thích các tham số:**
- `--share`: Tạo link công khai để truy cập từ bất kỳ đâu (khuyến nghị cho Colab)
- `--open`: Tự động mở trình duyệt (không hoạt động trên Colab)
- Mặc định chạy trên cổng `7860`

### 5.2. Chạy với cấu hình tùy chỉnh

```python
# Chạy với cổng tùy chỉnh và share
!python main/app/app.py --share --server_port 7860
```

---

## 🌐 BƯỚC 6: TRUY CẬP GIAO DIỆN

Sau khi chạy ứng dụng, bạn sẽ thấy output như sau:

```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live
```

### Cách truy cập:

1. **Link công khai (khuyến nghị):**
   - Copy link `https://xxxxx.gradio.live` từ output
   - Dán vào trình duyệt và truy cập
   - Link này hoạt động từ bất kỳ thiết bị nào

2. **Link local (chỉ trong Colab):**
   - Click vào link `http://127.0.0.1:7860` trong output
   - Hoặc sử dụng **"ngrok"** để tạo tunnel:
   ```python
   !pip install pyngrok
   from pyngrok import ngrok
   
   # Tạo tunnel
   public_url = ngrok.connect(7860)
   print(f"Public URL: {public_url}")
   ```

---

## ⚠️ LƯU Ý QUAN TRỌNG

### 1. **Thời gian phiên làm việc:**
- Colab miễn phí có thể ngắt kết nối sau 12 giờ không hoạt động
- GPU miễn phí có thể bị giới hạn thời gian sử dụng
- **Giải pháp:** Chạy code định kỳ để giữ phiên hoạt động:
  ```python
  import time
  while True:
      time.sleep(300)  # Chờ 5 phút
      print("Keeping session alive...")
  ```

### 2. **Lưu trữ dữ liệu:**
- Dữ liệu trong Colab sẽ **bị xóa** khi phiên kết thúc
- **Giải pháp:** Lưu vào Google Drive:
  ```python
  # Mount Google Drive
  from google.colab import drive
  drive.mount('/content/drive')
  
  # Copy dữ liệu quan trọng vào Drive
  !cp -r assets/weights /content/drive/MyDrive/
  !cp -r dataset /content/drive/MyDrive/
  ```

### 3. **Giới hạn bộ nhớ:**
- Colab miễn phí có giới hạn RAM và VRAM
- Nếu gặp lỗi "Out of Memory":
  - Giảm batch size khi training
  - Sử dụng CPU mode cho một số tác vụ
  - Xóa các biến không cần thiết:
  ```python
  import gc
  import torch
  gc.collect()
  torch.cuda.empty_cache()
  ```

### 4. **Tải file lên Colab:**
- Sử dụng giao diện web để upload file
- Hoặc upload lên Google Drive và mount:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  # File sẽ ở /content/drive/MyDrive/
  ```

---

## 🔧 XỬ LÝ LỖI THƯỜNG GẶP

### Lỗi 1: "CUDA out of memory"

**Nguyên nhân:** GPU hết bộ nhớ

**Giải pháp:**
```python
# Giảm batch size trong training
# Sử dụng CPU mode cho một số tác vụ
# Xóa cache GPU
import torch
torch.cuda.empty_cache()
```

### Lỗi 2: "Module not found"

**Nguyên nhân:** Thiếu package

**Giải pháp:**
```python
# Cài đặt lại package
!pip install [tên_package]

# Hoặc cài đặt lại tất cả
!pip install -r requirements.txt
```

### Lỗi 3: "Port already in use"

**Nguyên nhân:** Cổng 7860 đã được sử dụng

**Giải pháp:**
```python
# Sử dụng cổng khác
!python main/app/app.py --share --server_port 7861

# Hoặc kill process cũ
!fuser -k 7860/tcp
```

### Lỗi 4: "Connection timeout"

**Nguyên nhân:** Phiên Colab bị ngắt

**Giải pháp:**
- Chạy lại tất cả các ô code từ đầu
- Sử dụng `--share` để tạo link công khai ổn định hơn

### Lỗi 5: "FFmpeg not found"

**Giải pháp:**
```python
!apt-get update
!apt-get install -y ffmpeg
```

---

## 🎯 TỐI ƯU HÓA CHO COLAB

### 1. **Sử dụng Google Drive để lưu trữ:**

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Tạo symlink để lưu dữ liệu vào Drive
!mkdir -p /content/drive/MyDrive/Vietnamese-RVC
!ln -s /content/drive/MyDrive/Vietnamese-RVC/assets ./assets
!ln -s /content/drive/MyDrive/Vietnamese-RVC/dataset ./dataset
```

### 2. **Tự động tải pretrained models:**

```python
# Tải pretrained models vào thư mục assets
!mkdir -p assets/weights
!mkdir -p assets/indexes

# Tải từ HuggingFace hoặc các nguồn khác
# (Thêm code tải models nếu cần)
```

### 3. **Giữ phiên hoạt động:**

```python
# Chạy trong background để giữ phiên
import threading
import time

def keep_alive():
    while True:
        time.sleep(300)  # 5 phút
        print("Session alive")

thread = threading.Thread(target=keep_alive, daemon=True)
thread.start()
```

### 4. **Tối ưu bộ nhớ:**

```python
# Xóa cache định kỳ
import gc
import torch

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()
    print("Cache cleared")

# Gọi khi cần
clear_cache()
```

---

## 📚 VÍ DỤ NOTEBOOK HOÀN CHỈNH

Dưới đây là ví dụ notebook hoàn chỉnh để copy vào Colab:

```python
# ============================================
# CELL 1: Cài đặt môi trường
# ============================================
# Kích hoạt GPU: Runtime → Change runtime type → GPU

# Kiểm tra GPU
!nvidia-smi

# ============================================
# CELL 2: Clone repository
# ============================================
!git clone https://github.com/PhamHuynhAnh16/Vietnamese-RVC.git
%cd Vietnamese-RVC

# ============================================
# CELL 3: Cài đặt dependencies
# ============================================
!python -m pip install --upgrade pip
!pip install wheel

# Cài đặt PyTorch với CUDA
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cài đặt các dependencies khác
!pip install -r requirements.txt

# ============================================
# CELL 4: Mount Google Drive (tùy chọn)
# ============================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================
# CELL 5: Khởi động ứng dụng
# ============================================
!python main/app/app.py --share

# Sau khi chạy, copy link public URL và mở trong trình duyệt
```

---

## 🔗 LIÊN KẾT HỮU ÍCH

- **GitHub Repository:** https://github.com/PhamHuynhAnh16/Vietnamese-RVC
- **Colab Notebook chính thức:** https://colab.research.google.com/github/PhamHuynhAnh16/Vietnamese-RVC-ipynb/blob/main/Vietnamese-RVC.ipynb
- **HuggingFace Models:** https://huggingface.co/AnhP/Vietnamese-RVC-Project
- **Hướng dẫn sử dụng:** Xem file `HUONG_DAN_SU_DUNG.md`

---

## 💡 MẸO VÀ THỦ THUẬT

1. **Lưu checkpoint thường xuyên:**
   - Khi training, lưu checkpoint vào Google Drive
   - Tải checkpoint về local để tiếp tục training sau

2. **Sử dụng Colab Pro (nếu có):**
   - GPU tốt hơn (V100, A100)
   - Thời gian phiên làm việc lâu hơn
   - Bộ nhớ lớn hơn

3. **Tối ưu training:**
   - Giảm batch size nếu hết VRAM
   - Sử dụng mixed precision training
   - Tắt các tính năng không cần thiết

4. **Backup dữ liệu:**
   - Luôn backup models và datasets vào Google Drive
   - Sử dụng version control cho code

---

## 📞 HỖ TRỢ

Nếu gặp vấn đề:
- Tạo Issue trên GitHub: https://github.com/PhamHuynhAnh16/Vietnamese-RVC/issues
- Liên hệ Discord: **pham_huynh_anh**
- Xem file `HUONG_DAN_SU_DUNG.md` để biết cách sử dụng chi tiết

---

**Chúc bạn sử dụng Vietnamese-RVC trên Colab thành công! 🎉**
