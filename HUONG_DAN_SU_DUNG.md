# 📖 HƯỚNG DẪN SỬ DỤNG VIETNAMESE-RVC

## 🎯 HƯỚNG DẪN NHANH: ĐỔI GIỌNG CA SĨ BẰNG GIỌNG CỦA BẠN

### Mục tiêu: 
Bạn có giọng nói của mình và muốn thay thế giọng ca sĩ trong các bài hát bằng giọng của bạn.

### Quy trình 3 bước:

#### **BƯỚC 1: HUẤN LUYỆN MÔ HÌNH TỪ GIỌNG CỦA BẠN** ⭐ (Quan trọng nhất!)

1. **Chuẩn bị dữ liệu giọng nói:**
   - Thu âm giọng nói của bạn (10-30 phút, càng nhiều càng tốt)
   - Chất lượng tốt, rõ ràng, ít nhiễu
   - Định dạng: WAV, 44100Hz hoặc 48000Hz
   - Nên có nhiều đoạn ngắn (5-30 giây mỗi đoạn) thay vì 1 file dài

2. **Tạo Dataset:**

   **Cách 1: Từ YouTube (Nếu bạn có video trên YouTube)**
   - Vào tab **"Huấn Luyện Mô Hình"** → **"Tạo dữ liệu huấn luyện"**
   - Nhập link YouTube vào ô **"Đường dẫn liên kết đến âm thanh"**
     - Có thể nhập nhiều link, cách nhau bằng dấu phẩy: `link1, link2, link3`
   - ✅ Bật **"Tách nhạc"** nếu video có nhạc nền (để chỉ lấy giọng)
   - ✅ Bật **"Làm sạch âm thanh"** để loại bỏ nhiễu
   - Chọn **Tốc độ lấy mẫu:** `48000` (khuyến nghị) hoặc `44100`
   - Nhập tên dataset vào **"Đầu ra dữ liệu"** (ví dụ: `my_voice`)
   - Nhấn **"Tạo dữ liệu"**

   **Cách 2: Từ file local (Nếu bạn có file âm thanh trên máy)**
   - Đặt các file âm thanh của bạn vào thư mục `dataset\` trong dự án
   - Hoặc sử dụng tab **"Huấn Luyện Mô Hình"** → **"Huấn luyện mô hình"** → Bật **"Upload dataset"** để upload file trực tiếp
   - Sau đó xử lý tương tự như trên

3. **Trích xuất đặc trưng:**
   - Vào tab **"Huấn Luyện Mô Hình"** → **"Huấn luyện mô hình"**
   - Chọn dataset vừa tạo
   - Chọn **Mô hình Embedding:** `vietnamese_hubert_base` (tối ưu cho tiếng Việt)
   - Chọn **Phương thức F0:** `rmvpe` (nhanh và chính xác)
   - Nhấn **"Trích xuất đặc trưng"**

4. **Huấn luyện mô hình:**
   - Chọn dataset đã trích xuất
   - **Batch size:** 4-8 (tùy GPU của bạn)
   - **Epoch:** 50-200 (khuyến nghị: 100-150)
   - **Bộ mã hóa:** Chọn v2 (tốt hơn v1)
   - Nhấn **"Bắt đầu huấn luyện"**
   - ⏰ Đợi quá trình huấn luyện (có thể mất vài giờ tùy GPU)

5. **Tạo Index (Quan trọng!):**
   - Sau khi huấn luyện xong, tạo file index để cải thiện chất lượng
   - Mô hình và index sẽ được lưu trong `assets\weights\`

---

#### **BƯỚC 2: TÁCH GIỌNG CA SĨ TỪ BÀI HÁT**

1. Vào tab **"Suy Luận"** → **"Tách Nhạc"**
2. Upload file bài hát cần đổi giọng
3. Chọn mô hình tách nhạc:
   - **MDX-Net:** Nhanh, chất lượng tốt (khuyến nghị)
   - **Demucs:** Chất lượng rất tốt nhưng chậm và tốn RAM
   - **VR:** Chất lượng tốt
4. ✅ Bật **"Tách giọng nền"** nếu muốn tách cả giọng nền
5. Nhấn **"Tách Nhạc"**
6. Kết quả sẽ có:
   - **Giọng ca sĩ** (original_vocal) - File này bạn sẽ dùng để chuyển đổi
   - **Nhạc nền** (instruments) - File này để ghép lại sau

---

#### **BƯỚC 3: CHUYỂN ĐỔI GIỌNG CA SĨ BẰNG MÔ HÌNH CỦA BẠN**

1. Vào tab **"Suy Luận"** → **"Chuyển Đổi Âm Thanh"**

2. **Chọn mô hình:**
   - **Tệp mô hình:** Chọn mô hình bạn vừa huấn luyện (từ `assets\weights\`)
   - **Tệp chỉ mục:** Chọn file index tương ứng
   - Nhấn **"Tải lại"** nếu vừa tạo mô hình mới

3. **Upload giọng ca sĩ đã tách:**
   - Upload file **"original_vocal"** từ bước 2
   - Hoặc ✅ bật **"Sử dụng âm thanh vừa tách"** để tự động dùng file vừa tách

4. **Cài đặt:**
   - **Cao độ (Pitch):** 
     - Nếu giọng bạn và ca sĩ cùng giới tính: **0**
     - Nam → Nữ: **+12**
     - Nữ → Nam: **-12**
   - ✅ **Làm sạch âm thanh:** Bật để cải thiện chất lượng
   - **Độ mạnh chỉ mục:** 0.5-0.7 (cao hơn = giống giọng bạn hơn)

5. **Cài đặt nâng cao (tùy chọn):**
   - **Phương thức F0:** `rmvpe` (mặc định, tốt nhất)
   - **Mô hình Embedding:** `vietnamese_hubert_base`

6. **Ghép với nhạc nền (Tùy chọn):**
   - ✅ Bật **"Sử dụng âm thanh vừa tách"**
   - ✅ Bật **"Chuyển đổi giọng nền"** nếu muốn chuyển cả giọng nền
   - ✅ Bật **"Không ghép giọng nền"** nếu chỉ muốn giọng chính
   - ✅ Bật **"Ghép nhạc cụ"** để tự động ghép với nhạc nền

7. Nhấn **"Chuyển Đổi Âm Thanh"**

8. Kết quả:
   - **Giọng đã chuyển đổi** (main_convert) - Giọng bạn thay thế ca sĩ
   - **Bài hát hoàn chỉnh** (nếu đã ghép nhạc nền)

---

### 🎵 Kết quả cuối cùng:
Bạn sẽ có bài hát với giọng ca sĩ đã được thay thế bằng giọng của bạn!

---

### 💡 MẸO QUAN TRỌNG:

1. **Chất lượng dữ liệu huấn luyện:**
   - Càng nhiều dữ liệu giọng nói càng tốt (10-30 phút)
   - Giọng nói rõ ràng, không có tiếng ồn
   - Nhiều đoạn ngắn tốt hơn 1 đoạn dài

2. **Điều chỉnh cao độ:**
   - Nếu giọng bạn và ca sĩ khác giới tính, điều chỉnh pitch ±12
   - Nếu cùng giới tính, thử điều chỉnh nhỏ ±3-6 để phù hợp hơn

3. **Index Strength:**
   - 0.5-0.7: Giọng tự nhiên, giữ một số đặc điểm gốc
   - 0.7-1.0: Giọng giống mô hình hơn (giọng bạn) nhưng có thể mất tự nhiên

4. **Tách nhạc:**
   - MDX-Net thường đủ tốt cho hầu hết trường hợp
   - Nếu chất lượng không tốt, thử Demucs (chậm hơn)

5. **Huấn luyện:**
   - Kiểm tra loss trong Tensorboard
   - Dừng khi loss không giảm nữa (thường sau 100-150 epochs)
   - Không huấn luyện quá lâu (overfitting)

---

## 🚀 Khởi động ứng dụng

### Cách 1: Sử dụng file batch (Đơn giản nhất)
```bash
run_app.bat
```

### Cách 2: Chạy từ Command Prompt/Terminal
```bash
.\env\Scripts\activate
python main\app\app.py --open
```

Sau khi chạy, trình duyệt sẽ tự động mở giao diện web tại `http://127.0.0.1:7860` (hoặc cổng khác nếu 7860 đã được sử dụng).

---

## 📑 Tổng quan các tab chính

Giao diện Vietnamese-RVC có 6 tab chính:

1. **Suy Luận** (Inference) - Chuyển đổi giọng nói và tách nhạc
2. **Chỉnh Sửa** (Edit) - Chỉnh sửa âm thanh
3. **Thời gian thực** (Real-time) - Chuyển đổi giọng nói thời gian thực
4. **Huấn Luyện Mô Hình** (Train Model) - Huấn luyện mô hình RVC
5. **Tải Xuống** (Download) - Tải mô hình và tài nguyên
6. **Thêm** (Extra) - Các tính năng bổ sung

---

## 🎵 TAB 1: SUY LUẬN (INFERENCE)

Tab này có 4 chức năng con:

### 1.1. Tách Nhạc (Separate Music)

**Mục đích:** Tách giọng hát và nhạc nền từ file âm thanh.

**Cách sử dụng:**
1. Chọn phương thức tách nhạc: **MDX-Net**, **Demucs**, hoặc **VR**
2. Upload file âm thanh cần tách (hỗ trợ: wav, mp3, flac, m4a, v.v.)
3. Nhấn nút **"Tách Nhạc"**
4. Kết quả sẽ hiển thị:
   - Giọng hát đã tách
   - Nhạc nền đã tách
   - Có thể tải về từng phần

**Lưu ý:**
- Demucs có thể tốn nhiều bộ nhớ GPU, nếu gặp lỗi hãy chỉnh `demucs_cpu_mode` thành `true` trong `main\configs\config.json`

---

### 1.2. Chuyển Đổi Âm Thanh (Voice Conversion)

**Mục đích:** Chuyển đổi giọng nói từ file âm thanh sử dụng mô hình RVC đã được huấn luyện.

#### Bước 1: Chuẩn bị mô hình
- **Tệp mô hình (.pth):** Chọn mô hình RVC từ dropdown hoặc nhập đường dẫn
- **Tệp chỉ mục (.index):** Chọn file index (tùy chọn, nhưng khuyến khích để chất lượng tốt hơn)
- Nhấn **"Tải lại"** nếu vừa thêm mô hình mới

#### Bước 2: Upload âm thanh
- Kéo thả file âm thanh vào vùng **"Thả âm thanh vào đây"**
- Hoặc click để chọn file từ máy tính
- Hỗ trợ định dạng: wav, mp3, flac, m4a, v.v.

#### Bước 3: Cài đặt cơ bản

**Các checkbox:**
- ✅ **Làm sạch âm thanh:** Loại bỏ nhiễu, cải thiện chất lượng
- ✅ **Tự động điều chỉnh:** Tự động chỉnh cao độ
- ✅ **Sử dụng âm thanh vừa tách:** Sử dụng file đã tách từ tab "Tách Nhạc"
- ✅ **Sử dụng hiệu quả bộ nhớ:** Giảm sử dụng RAM (chậm hơn một chút)

**Cao độ (Pitch):**
- Thanh trượt từ **-20 đến +20**
- **Khuyến cáo:** 
  - Chuyển giọng nam → nữ: **+12**
  - Chuyển giọng nữ → nam: **-12**
  - Giữ nguyên: **0**

#### Bước 4: Cài đặt nâng cao (Mở rộng các Accordion)

**Cài đặt F0 (Cao độ):**
- **Phương thức F0:** Chọn phương thức trích xuất cao độ
  - `rmvpe` (khuyến nghị, nhanh và chính xác)
  - `harvest` (chậm hơn nhưng chính xác)
  - `dio` (nhanh nhưng kém chính xác)
  - `crepe` (các phiên bản: tiny, small, medium, large, full)
  - `hybrid` (kết hợp nhiều phương thức)
- **F0 ONNX Mode:** Bật để tăng tốc (nếu có)
- **Hop Length:** Độ dài bước nhảy (mặc định: 160)

**Mô hình Embedding (Hubert):**
- **Chế độ:** fairseq, onnx, transformers, spin, whisper
- **Mô hình:** 
  - `hubert_base` (mặc định)
  - `vietnamese_hubert_base` (tối ưu cho tiếng Việt)
  - `contentvec_base`
  - Các mô hình khác theo ngôn ngữ

**Cài đặt khác:**
- **Độ mạnh chỉ mục (Index Strength):** 0.0 - 1.0 (mặc định: 0.5)
  - Cao hơn = giọng giống mô hình hơn
  - Thấp hơn = giữ nguyên đặc điểm giọng gốc hơn
- **Bán kính lọc (Filter Radius):** 0-7 (mặc định: 3)
- **Tỷ lệ RMS (RMS Mix Rate):** 0.0 - 1.0
- **Bảo vệ (Protect):** 0.0 - 1.0 (bảo vệ các âm không phải giọng nói)

#### Bước 5: Chuyển đổi
1. Nhấn nút **"Chuyển Đổi Âm Thanh"** (màu xanh)
2. Đợi quá trình xử lý (có thể mất vài phút tùy độ dài file)
3. Kết quả sẽ hiển thị ở phần dưới, có thể nghe và tải về

**Lưu ý:**
- File đầu ra mặc định: `audios/output.wav`
- Có thể thay đổi đường dẫn đầu ra trong phần "Đầu vào, đầu ra âm thanh"

---

### 1.3. Chuyển Đổi Âm Thanh Với Whisper

**Mục đích:** Chuyển đổi giọng nói kết hợp với Whisper để cải thiện chất lượng.

**Cách sử dụng:**
1. Upload file âm thanh
2. Chọn mô hình Whisper (tiny, base, small, medium, large)
3. Chọn mô hình RVC và index
4. Cài đặt các thông số tương tự như "Chuyển Đổi Âm Thanh"
5. Nhấn **"Chuyển Đổi"**

---

### 1.4. Chuyển Đổi Văn Bản (Text-to-Speech)

**Mục đích:** Chuyển đổi văn bản thành giọng nói, sau đó chuyển đổi giọng nói bằng RVC.

**Cách sử dụng:**
1. Nhập văn bản cần chuyển đổi
2. Chọn giọng nói TTS (Edge-TTS hoặc các engine khác)
3. Chọn mô hình RVC và index
4. Cài đặt cao độ và các thông số
5. Nhấn **"Chuyển Đổi"**

---

## ✏️ TAB 2: CHỈNH SỬA (EDIT)

**Mục đích:** Chỉnh sửa và xử lý âm thanh với các hiệu ứng.

**Các tính năng:**
- Cắt, ghép âm thanh
- Thay đổi tốc độ phát
- Thay đổi cao độ
- Thêm hiệu ứng (reverb, echo, v.v.)
- Loại bỏ nhiễu
- Chuẩn hóa âm lượng

---

## 🎤 TAB 3: THỜI GIAN THỰC (REAL-TIME)

**Mục đích:** Chuyển đổi giọng nói thời gian thực từ microphone.

**Cách sử dụng:**
1. Chọn thiết bị microphone đầu vào
2. Chọn thiết bị loa đầu ra
3. Chọn mô hình RVC và index
4. Cài đặt các thông số (pitch, index strength, v.v.)
5. Nhấn **"Bắt đầu"** để bắt đầu chuyển đổi
6. Nói vào microphone, giọng nói sẽ được chuyển đổi và phát ra loa

**Lưu ý:**
- Cần có microphone và loa
- Độ trễ phụ thuộc vào cấu hình máy
- GPU sẽ giúp giảm độ trễ đáng kể

---

## 🎓 TAB 4: HUẤN LUYỆN MÔ HÌNH (TRAIN MODEL)

**Mục đích:** Huấn luyện mô hình RVC từ dữ liệu giọng nói của bạn.

Tab này có 3 chức năng con:

### 4.1. Tạo dữ liệu huấn luyện (Create Dataset)

**Mục đích:** Xử lý và tạo dataset từ file âm thanh hoặc link YouTube.

#### Cách sử dụng:

**A. Từ link YouTube:**
1. Nhập link YouTube vào ô **"Đường dẫn liên kết đến âm thanh"**
   - Có thể nhập nhiều link, cách nhau bằng dấu phẩy: `https://youtube.com/watch?v=..., https://youtube.com/watch?v=...`
2. Nhập tên dataset vào **"Đầu ra dữ liệu"** (ví dụ: `my_voice`)

**B. Từ file local:**
- Đặt file vào thư mục `dataset\` hoặc sử dụng tab "Huấn luyện mô hình" → Bật "Upload dataset"

#### Các tùy chọn xử lý:

**Checkbox:**
- ✅ **Tách Nhạc:** Bật nếu file có nhạc nền (chỉ lấy giọng nói)
  - Khi bật, sẽ hiện thêm các tùy chọn:
    - **Mô hình tách nhạc:** MDX-Net, Demucs, VR
    - **Overlap:** 0.25, 0.5, 0.75, 0.99 (mặc định: 0.25)
    - **Segments size:** 32-3072 (mặc định: 256)
    - **Shifts:** 1-20 (mặc định: 2)
- ✅ **Làm sạch âm thanh:** Loại bỏ nhiễu
  - **Độ mạnh làm sạch:** 0.0-1.0 (mặc định: 0.5)
- ✅ **Bỏ qua giây:** Bỏ qua phần đầu/cuối file
  - **Bỏ qua đầu:** Nhập số giây (ví dụ: `0,5,10` để bỏ 0s, 5s, 10s)
  - **Bỏ qua cuối:** Tương tự
- ✅ **Tách vang:** Loại bỏ reverb/echo (chỉ khi bật "Tách Nhạc")

**Tốc độ lấy mẫu (Sample Rate):**
- **Khuyến nghị:** `48000` Hz hoặc `44100` Hz
- **Lưu ý:** Một số định dạng không hỗ trợ trên 48000
- Các tùy chọn: 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 96000

**Thông tin:**
- Sau khi nhấn **"Tạo dữ liệu"**, thông tin tiến trình sẽ hiển thị ở ô **"Thông tin tạo dữ liệu"**

**Lưu ý:**
- Dataset sẽ được lưu trong thư mục `dataset\[tên_dataset]\`
- Quá trình có thể mất vài phút đến vài giờ tùy độ dài file
- Nên bật "Tách Nhạc" nếu file có nhạc nền để có chất lượng tốt hơn

---

### 4.2. Tạo tham chiếu huấn luyện (Create Reference)

**Mục đích:** Tạo file tham chiếu để cải thiện chất lượng huấn luyện.

**Cách sử dụng:**
1. Upload file âm thanh mẫu (giọng nói rõ ràng, chất lượng tốt)
2. Chọn phương thức F0 và mô hình embedding
3. Nhấn **"Tạo tham chiếu"**

---

### 4.3. Huấn luyện mô hình (Train Model)

**Mục đích:** Huấn luyện mô hình RVC từ dataset đã chuẩn bị.

#### 📁 SỬ DỤNG DỮ LIỆU LOCAL (Bạn đã có dữ liệu trên máy)

**Cách 1: Upload trực tiếp trong giao diện (Khuyến nghị)**

1. **Bật tùy chọn upload:**
   - ✅ Bật checkbox **"Tải lên dữ liệu huấn luyện"** (Upload training data)
   - Khi bật, sẽ hiện vùng kéo thả file

2. **Upload file:**
   - Kéo thả tất cả file âm thanh vào vùng **"Thả âm thanh vào đây"**
   - Hoặc click để chọn file từ máy tính
   - Hỗ trợ: wav, mp3, flac, m4a, v.v.
   - File sẽ tự động được di chuyển vào thư mục dataset

3. **Đặt tên mô hình:**
   - Nhập tên mô hình vào **"Tên của mô hình"**
   - ⚠️ **Lưu ý:** Không dùng ký tự đặc biệt hay dấu cách (ví dụ: `my_voice`, `voice_model_1`)

4. **Cài đặt cơ bản:**
   - **Tỉ lệ lấy mẫu:** `48k` (khuyến nghị) hoặc `40k`, `32k`
   - **Phiên bản mô hình:** `v2` (tốt hơn v1, khuyến nghị)
   - ✅ **Huấn luyện cao độ:** Bật (mặc định, quan trọng!)

5. **Cài đặt xử lý dữ liệu:**
   - ✅ **Làm sạch dữ liệu:** Bật nếu file có nhiễu
   - **Tùy chỉnh cắt âm thanh:** `Automatic` (mặc định, khuyến nghị)
   - **Chuẩn hóa âm lượng:** `none` (mặc định) hoặc `pre`, `post`

6. **Bước 1: Xử lý dữ liệu:**
   - Nhấn nút **"1. Xử lí dữ liệu"** (màu xanh)
   - Đợi quá trình xử lý (cắt file, chuẩn hóa, v.v.)
   - Thông tin tiến trình hiển thị ở **"Thông tin phần xử lí trước"**

**Cách 2: Đặt file vào thư mục dataset trước**

1. **Chuẩn bị thư mục:**
   - Tạo thư mục trong `dataset\` (ví dụ: `dataset\my_voice\`)
   - Copy tất cả file âm thanh vào thư mục đó

2. **Cấu hình dataset:**
   - ✅ Bật **"Tùy chỉnh dataset"** (Custom dataset)
   - Nhập đường dẫn: `dataset\my_voice` (hoặc tên thư mục bạn đã tạo)

3. **Tiếp tục từ bước 4** ở trên

---

#### 🔄 QUY TRÌNH HUẤN LUYỆN ĐẦY ĐỦ (5 bước):

**Bước 1: Xử lý dữ liệu (Preprocess)**
- Nhấn **"1. Xử lí dữ liệu"**
- Hệ thống sẽ:
  - Cắt file thành các đoạn ngắn
  - Chuẩn hóa âm lượng
  - Làm sạch (nếu bật)
  - Lưu vào thư mục dataset

**Bước 2: Trích xuất đặc trưng (Extract)**
- **Phương pháp trích xuất F0:** Chọn `rmvpe` (khuyến nghị)
- **Mô hình nhúng:** Chọn `vietnamese_hubert_base` (tối ưu cho tiếng Việt)
- Nhấn **"2. Trích xuất dữ liệu"**
- Hệ thống sẽ trích xuất:
  - Đặc trưng giọng nói (embedding)
  - Cao độ (F0) - nếu bật "Huấn luyện cao độ"

**Bước 3: Tạo chỉ mục (Create Index)**
- Nhấn **"3. Tạo chỉ mục"**
- Tạo file index để cải thiện chất lượng chuyển đổi
- File index sẽ được lưu cùng với mô hình

**Bước 4: Huấn luyện mô hình (Train)**
- **Tổng số kỷ nguyên (Epochs):** 100-300 (khuyến nghị: 200-300)
  - Ít hơn 100: Chất lượng chưa tốt
  - 200-300: Thường đủ tốt
  - Trên 500: Có thể bị overfitting
- **Tần suất lưu:** 50 (lưu mô hình mỗi 50 epochs)
- **Batch size:** 4-8 (tùy GPU, mặc định: 8)
  - GPU mạnh: 8-16
  - GPU yếu: 4-6
- Nhấn **"4. Huấn Luyện Mô Hình"**
- ⏰ Quá trình có thể mất vài giờ đến vài ngày tùy:
  - Số lượng dữ liệu
  - Số epochs
  - GPU của bạn
- Theo dõi tiến trình ở **"Thông tin phần huấn luyện"**

**Bước 5: Xuất mô hình (Export)**
- Sau khi huấn luyện xong, chọn mô hình và index
- Nhấn **"Xuất mô hình"** hoặc **"Zip mô hình"** để đóng gói

---

#### ⚙️ CÀI ĐẶT NÂNG CAO (Mở Accordion "Cài đặt chung"):

**Cài đặt GPU/CPU:**
- **Số GPU:** Mặc định tự động, có thể chỉ định (ví dụ: `0` cho GPU đầu tiên)
- **Số lõi CPU:** Mặc định = tất cả lõi
- **Batch size:** 4-16 (tùy GPU)

**Cài đặt huấn luyện:**
- ✅ **Cache trong GPU:** Bật để tăng tốc (nếu có đủ VRAM)
- ✅ **Lưu mọi trọng số:** Bật để lưu tất cả checkpoint
- ✅ **Chỉ lưu mới nhất:** Tắt nếu muốn giữ tất cả checkpoint
- **Optimizer:** `AdamW` (mặc định, tốt nhất)

**Cài đặt pretrained:**
- Mô hình sẽ tự động tải pretrained model
- Có thể tùy chỉnh trong phần "Custom pretrain"

**Vocoder (Bộ mã hóa):**
- **Default:** Mặc định, ổn định
- **MRF-HiFi-GAN:** Chất lượng tốt hơn (cần pretrained)
- **RefineGAN:** Chất lượng rất tốt (cần pretrained)

---

#### 🔄 TIẾP TỤC TRAINING TỪ CHECKPOINT (Resume Training):

**Khi nào cần resume?**
- Đã train trên Google Colab và muốn tiếp tục ở local
- Training bị gián đoạn (tắt máy, lỗi, v.v.)
- Muốn train thêm epochs từ checkpoint hiện có

**Cách tiếp tục training từ checkpoint:**

1. **Chuẩn bị file checkpoint:**
   - Bạn cần 2 file: `G_latest.pth` (Generator) và `D_latest.pth` (Discriminator)
   - Nếu file có tên khác (ví dụ: `G_50.pth`, `D_50.pth`), đổi tên thành `G_latest.pth` và `D_latest.pth`

2. **Đặt file vào đúng thư mục:**
   - Tạo hoặc tìm thư mục model: `assets\logs\{tên_mô_hình}\`
   - Ví dụ: Nếu tên mô hình là `my_voice`, đặt vào `assets\logs\my_voice\`
   - Copy 2 file vào:
     ```
     assets\logs\my_voice\G_latest.pth
     assets\logs\my_voice\D_latest.pth
     ```

3. **Đảm bảo có đầy đủ dữ liệu đã xử lý:**
   - ✅ Thư mục `sliced_audios\` (từ bước 1: Xử lý dữ liệu)
   - ✅ Thư mục `v2_extracted\` hoặc `v1_extracted\` (từ bước 2: Trích xuất)
   - ✅ File `config.json` trong thư mục model
   - ✅ File `filelist.txt` trong thư mục model

4. **Cài đặt training:**
   - **Tên mô hình:** Phải giống với tên thư mục chứa checkpoint
   - **Tỉ lệ lấy mẫu:** Phải giống với lúc train trước (48k, 40k, hoặc 32k)
   - **Phiên bản:** Phải giống (v1 hoặc v2)
   - **Huấn luyện cao độ:** Phải giống với lúc train trước
   - **Tổng số kỷ nguyên:** Đặt số epochs bạn muốn train thêm
     - Ví dụ: Đã train 50 epochs, muốn train thêm 50 → Đặt **100** (tổng cộng)
     - Hoặc: Đã train 50 epochs, muốn train thêm 50 → Đặt **50** (sẽ train từ epoch 51 đến 100)

5. **Bắt đầu training:**
   - ⚠️ **KHÔNG** chạy lại bước 1, 2, 3 (đã có sẵn)
   - Chỉ cần nhấn **"4. Huấn Luyện Mô Hình"**
   - Ứng dụng sẽ tự động:
     - Tìm file `G_latest.pth` và `D_latest.pth`
     - Load checkpoint và tiếp tục từ epoch đã lưu + 1
     - Ví dụ: Checkpoint ở epoch 50 → Sẽ tiếp tục từ epoch 51

**Lưu ý quan trọng:**
- ✅ Tên mô hình phải giống nhau
- ✅ Sample rate, version, pitch guidance phải giống nhau
- ✅ File checkpoint phải đặt đúng tên: `G_latest.pth` và `D_latest.pth`
- ✅ Phải có đầy đủ dữ liệu đã preprocess và extract
- ⚠️ Nếu không tìm thấy checkpoint, sẽ bắt đầu từ epoch 1 (train từ đầu)

**Ví dụ cụ thể:**
```
Đã train trên Colab: 50 epochs → Có file G_50.pth và D_50.pth
Tải về local:
1. Đổi tên: G_50.pth → G_latest.pth, D_50.pth → D_latest.pth
2. Đặt vào: assets\logs\my_voice\G_latest.pth và D_latest.pth
3. Đảm bảo có: sliced_audios\, v2_extracted\, config.json, filelist.txt
4. Vào tab Training, đặt:
   - Tên mô hình: my_voice
   - Sample rate: 48k (giống lúc train trên Colab)
   - Version: v2 (giống lúc train trên Colab)
   - Total epochs: 100 (để train thêm 50 epochs nữa)
5. Nhấn "4. Huấn Luyện Mô Hình"
→ Sẽ tự động tiếp tục từ epoch 51 đến 100
```

---

### Quy trình huấn luyện (Tóm tắt):

#### Bước 1: Chuẩn bị dữ liệu
- Thu thập file âm thanh giọng nói (khuyến nghị: 10-30 phút, chất lượng tốt)
- File nên là giọng nói rõ ràng, ít nhiễu
- Định dạng: wav, 44100Hz hoặc 48000Hz

#### Bước 2: Tạo dataset
- Vào tab **"Tạo dữ liệu huấn luyện"**
- Upload các file âm thanh
- Chọn phương thức tách nhạc (nếu cần)
- Nhấn **"Tạo Dataset"**

#### Bước 3: Trích xuất đặc trưng
- Chọn dataset đã tạo
- Chọn mô hình embedding (khuyến nghị: `vietnamese_hubert_base` cho tiếng Việt)
- Chọn phương thức F0
- Nhấn **"Trích xuất"**

#### Bước 4: Huấn luyện
- Chọn dataset đã trích xuất
- Cài đặt các thông số:
  - **Batch size:** 4-8 (tùy GPU)
  - **Epoch:** 50-200 (càng nhiều càng tốt nhưng mất thời gian)
  - **Learning rate:** Mặc định thường ổn
  - **Bộ mã hóa:** Chọn v1 hoặc v2
- Nhấn **"Bắt đầu huấn luyện"**

#### Bước 5: Kiểm tra tiến trình
- Mở Tensorboard: `tensorboard.bat` hoặc chạy `python main\app\run_tensorboard.py`
- Xem biểu đồ loss và các metric khác
- Dừng khi loss không giảm nữa

#### Bước 6: Xuất mô hình
- Sau khi huấn luyện xong, xuất mô hình
- Tạo index file để cải thiện chất lượng
- Mô hình sẽ được lưu trong `assets\weights\`

---

## 📥 TAB 5: TẢI XUỐNG (DOWNLOAD)

**Mục đích:** Tải mô hình, index, và các tài nguyên từ internet.

**Các nguồn:**
- HuggingFace
- AI Hub
- Voice-models.com
- Mediafire, Mega, Google Drive, v.v.

**Cách sử dụng:**
1. Nhập link hoặc tên mô hình
2. Chọn loại file (mô hình, index, v.v.)
3. Nhấn **"Tải xuống"**
4. File sẽ được lưu vào thư mục tương ứng

---

## ⚙️ TAB 6: THÊM (EXTRA)

**Các tính năng bổ sung:**
- Dung hợp mô hình (Merge models)
- Đọc thông tin mô hình
- Xuất mô hình sang ONNX
- Tạo tham chiếu huấn luyện
- Trích xuất cao độ
- Các công cụ tiện ích khác

---

## 💡 MẸO VÀ LƯU Ý

### Để có chất lượng tốt nhất:
1. ✅ Sử dụng file âm thanh chất lượng cao (44100Hz hoặc 48000Hz)
2. ✅ Luôn sử dụng file index (.index) khi chuyển đổi
3. ✅ Điều chỉnh Index Strength phù hợp (0.5-0.7 thường tốt)
4. ✅ Chọn mô hình embedding phù hợp với ngôn ngữ
5. ✅ Sử dụng phương thức F0 phù hợp:
   - `rmvpe`: Nhanh và chính xác (khuyến nghị)
   - `harvest`: Chậm nhưng rất chính xác
   - `crepe-full`: Chính xác nhất nhưng rất chậm

### Xử lý lỗi thường gặp:

**Lỗi: Out of memory (OOM)**
- Giảm batch size khi huấn luyện
- Bật "Sử dụng hiệu quả bộ nhớ"
- Đóng các ứng dụng khác

**Lỗi: CUDA out of memory**
- Giảm batch size
- Sử dụng CPU mode cho một số tác vụ
- Giảm độ phân giải mô hình

**Chất lượng kém:**
- Kiểm tra lại mô hình và index
- Điều chỉnh Index Strength
- Thử phương thức F0 khác
- Kiểm tra chất lượng file đầu vào

**Độ trễ cao (Real-time):**
- Sử dụng GPU
- Giảm buffer size
- Chọn mô hình nhẹ hơn
- Tắt các hiệu ứng không cần thiết

---

## 📂 Cấu trúc thư mục quan trọng

```
Vietnamese-RVC/
├── assets/
│   ├── weights/          # Mô hình RVC (.pth)
│   ├── indexes/          # File index (.index)
│   ├── models/           # Mô hình embedding, F0, v.v.
│   └── presets/          # File preset (.conversion.json)
├── audios/               # File âm thanh đầu vào/đầu ra
├── dataset/              # Dataset huấn luyện
└── main/
    └── configs/
        └── config.json   # File cấu hình chính
```

---

## 🔗 Liên kết hữu ích

- **GitHub:** https://github.com/PhamHuynhAnh16/Vietnamese-RVC
- **HuggingFace Spaces:** https://huggingface.co/spaces/AnhP/RVC-GUI
- **HuggingFace Models:** https://huggingface.co/AnhP/Vietnamese-RVC-Project
- **Voice Models:** https://voice-models.com/
- **Google Colab:** https://colab.research.google.com/github/PhamHuynhAnh16/Vietnamese-RVC-ipynb/blob/main/Vietnamese-RVC.ipynb

---

## 📞 Hỗ trợ

Nếu gặp vấn đề:
- Tạo Issue trên GitHub: https://github.com/PhamHuynhAnh16/Vietnamese-RVC/issues
- Liên hệ Discord: **pham_huynh_anh**

---

**Chúc bạn sử dụng Vietnamese-RVC thành công! 🎉**

