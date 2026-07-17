# 📡 TÀI LIỆU API — IDOL VOICE (Vietnamese-RVC Headless API)

Hệ thống đổi giọng ca sĩ bằng giọng khách hàng, chạy headless trên GPU.

- **Base URL:** `https://idolvoice.karaokeicool.vn`
- **Xác thực:** header `X-API-Key: <key>` cho mọi endpoint, **trừ 2 endpoint mở**: `/health` (healthcheck) và `/api/files/upload/audio` (tương thích client karaoke). Sai/thiếu key → `401`.
- **Mô hình xử lý:** bất đồng bộ theo hàng đợi. Các API nặng trả về `task_id` ngay; dùng `/status/{task_id}` theo dõi và `/download/{task_id}` lấy kết quả. Server xử lý **tuần tự 1 task một lúc**.
- **Swagger UI:** `https://idolvoice.karaokeicool.vn/docs` (thử API trực tiếp trên trình duyệt).

---

## 🔄 Luồng sử dụng chuẩn (theo khách hàng)

```
1. GET  /model/{customer_id}     → khách đã có model chưa?
2. POST /train                   → chưa có: train model từ file ghi âm (1 lần)
   GET  /status/{task_id}        → poll đến khi completed
3. POST /convert                 → đổi giọng bài hát bằng model của khách
   GET  /status/{task_id}        → poll đến khi completed
4. GET  /conversions/{customer_id} → danh sách bản đã convert của khách
   GET  /conversions/{id}/stream    → khách nghe thử
   GET  /conversions/{id}/download  → ưng ý thì tải về
```

Khách quay lại lần sau: bỏ qua bước 2, gọi thẳng `/convert`. Muốn train dữ liệu mới thay model cũ: `/train` với `force_retrain=true`.

---

## 1. `GET /health` — Kiểm tra server

Không cần API key.

```bash
curl https://idolvoice.karaokeicool.vn/health
```

**Response `200`:**
```json
{"status": "ok", "mode": "headless-async", "active_tasks": 3}
```

---

## 2. `POST /train` — Train model giọng theo khách hàng

Upload file ghi âm của khách → hệ thống tách giọng, huấn luyện model RVC và lưu theo `customer_id` (đăng ký vào DB bảng `rvc_customer_models`).

**Content-Type:** `multipart/form-data`

| Tham số | Kiểu | Bắt buộc | Mặc định | Mô tả |
|---|---|---|---|---|
| `customer_id` | text | ✅ | — | Mã khách hàng (chữ/số/gạch; ký tự khác bị thay bằng `_`) |
| `training_files` | file (nhiều) | ✅ | — | File ghi âm giọng khách (wav/mp3...). Lặp lại field để gửi nhiều file |
| `epochs` | int | ❌ | 150 | Số vòng huấn luyện (100–300 khuyến nghị) |
| `force_retrain` | bool | ❌ | false | `true` = xóa model cũ, train dữ liệu mới thay thế |

```bash
curl -X POST https://idolvoice.karaokeicool.vn/train \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "customer_id=KH001" \
  -F "epochs=150" \
  -F "force_retrain=false" \
  -F "training_files=@ghi_am_1.wav" \
  -F "training_files=@ghi_am_2.wav"
```

**Response — khách mới (bắt đầu train), `200`:**
```json
{
  "status": "queued",
  "task_id": "55d9ca4c-d023-4abe-a680-c6d85cd9f379",
  "customer_id": "KH001",
  "model_name": "cus_KH001",
  "message": "Đã nhận 2 file ghi âm. Bắt đầu train model (vị trí hàng đợi: 1).",
  "queue_size": 1
}
```

**Response — khách đã có model (không train lại), `200`:**
```json
{
  "status": "exists",
  "customer_id": "KH001",
  "model_name": "cus_KH001",
  "model_file": "cus_KH001_150e_....pth",
  "message": "Khách hàng đã có model. Gửi force_retrain=true nếu muốn train dữ liệu mới thay thế."
}
```

**⚠️ Yêu cầu dữ liệu train:**
- Tổng thời lượng **giọng nói thực tế** (đã trừ khoảng lặng) tối thiểu **60 giây** — ít hơn task sẽ `failed` với thông báo trong `logs`.
- Nên gửi 10–30 phút ghi âm, rõ, ít tạp âm để chất lượng tốt.
- Thời gian train: ~20–60 phút tùy dữ liệu và epochs (GPU RTX 5070 Ti).

---

## 3. `GET /model/{customer_id}` — Kiểm tra model của khách

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  https://idolvoice.karaokeicool.vn/model/KH001
```

**Response `200`:**
```json
{
  "customer_id": "KH001",
  "trained": true,
  "model_name": "cus_KH001",
  "model_file": "cus_KH001_150e_....pth",
  "index_file": "added_IVF..._cus_KH001_v2.index",
  "db_record": {
    "customer_id": "KH001",
    "model_name": "cus_KH001",
    "model_file": "cus_KH001_150e_....pth",
    "index_file": "added_..._v2.index",
    "epochs": 150,
    "trained_at": "2026-07-09 09:15:00",
    "updated_at": "2026-07-09 09:15:00"
  }
}
```

`trained: false` → cần gọi `/train` trước khi `/convert`.

---

## 4. `POST /convert` — Đổi giọng bài hát bằng model của khách

Dùng model đã train của `customer_id` để thay giọng ca sĩ trong bài hát, tự ghép lại beat.

**Content-Type:** `multipart/form-data`

| Tham số | Kiểu | Bắt buộc | Mặc định | Mô tả |
|---|---|---|---|---|
| `customer_id` | text | ✅ | — | Mã khách (phải có model — nếu chưa: lỗi `404`) |
| `target_song_id` | text | ⭕ chọn 1 | — | ID bài hát trong hệ thống (bảng `ktv_song`) — server tự lấy file audio |
| `target_song` | file | ⭕ chọn 1 | — | HOẶC upload file bài hát trực tiếp (mp3/wav...) |
| `pitch_shift` | int | ❌ | 0 | Dịch cao độ theo nửa cung: cùng giới tính `0`; nam→nữ `+12`; nữ→nam `-12` |

**Cách 1 — theo id bài hát (khuyên dùng):**
```bash
curl -X POST https://idolvoice.karaokeicool.vn/convert \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "customer_id=KH001" \
  -F "target_song_id=103691" \
  -F "pitch_shift=0"
```

**Cách 2 — upload file bài hát:**
```bash
curl -X POST https://idolvoice.karaokeicool.vn/convert \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "customer_id=KH001" \
  -F "target_song=@bai_hat.mp3" \
  -F "pitch_shift=0"
```

**Response `200`:**
```json
{
  "status": "queued",
  "task_id": "a1b2c3d4-...",
  "customer_id": "KH001",
  "model_name": "cus_KH001",
  "message": "Bắt đầu đổi giọng bằng model của khách (vị trí hàng đợi: 1).",
  "queue_size": 1
}
```

**⚠️ Lưu ý `target_song_id`:**
- Chỉ dùng được bài **đã xuất bản/đồng bộ** lên media server. Bài chưa sync → lỗi `502` kèm thông báo `"có thể bài chưa xuất bản (version v/0)"`.
- Thời gian convert: ~2–5 phút/bài (tách nhạc + đổi giọng + ghép beat).

---

## 5. `GET /status/{task_id}` — Trạng thái task

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  https://idolvoice.karaokeicool.vn/status/55d9ca4c-d023-4abe-a680-c6d85cd9f379
```

**Response `200`:**
```json
{
  "task_id": "55d9ca4c-...",
  "status": "running",
  "message": "Processing...",
  "result_path": null,
  "logs": "== BẮT ĐẦU HUẤN LUYỆN (150 epochs) ==\nĐang tiền xử lý...",
  "created_at": 1783921234.5,
  "kind": "train",
  "customer_id": "KH001"
}
```

**Vòng đời `status`:** `queued` → `running` → `completed` | `failed`

- `logs` chứa 3000 ký tự log gần nhất — dùng để hiển thị tiến độ hoặc debug khi `failed`.
- Khuyến nghị poll mỗi **10–15 giây**.
- `task_id` không tồn tại → `404`. Lưu ý: danh sách task lưu **trong RAM** — restart server sẽ mất `task_id` cũ (model đã train KHÔNG mất).

---

## 6. `GET /download/{task_id}` — Tải kết quả

Chỉ dùng được khi task `completed` (chưa xong → `400`).

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  https://idolvoice.karaokeicool.vn/download/a1b2c3d4-... -o ket_qua.mp3
```

- Task `convert`/`run_upload`: trả file **bài hát đã đổi giọng (.mp3)**.
- Task `train`: trả file model `.pth` (thường không cần tải — model đã lưu trên server).
- File kết quả được giữ trên server **90 ngày (3 tháng)** rồi tự xóa → tải về/lưu lại trong thời gian đó.

---

## 7. `GET /songs` — Danh sách / tìm kiếm bài hát

Trả về danh sách bài hát trong hệ thống (chỉ các bài có media), phân trang.

| Tham số (query) | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `q` | text | — | Từ khóa tìm theo tên bài (**có dấu hoặc không dấu** đều được) |
| `limit` | int | 50 | Số bài mỗi trang (tối đa 200) |
| `offset` | int | 0 | Vị trí bắt đầu (phân trang) |

```bash
# 50 bài mới nhất
curl -H "X-API-Key: YOUR_API_KEY" "https://idolvoice.karaokeicool.vn/songs"

# Tìm bài theo tên (không dấu cũng được)
curl -H "X-API-Key: YOUR_API_KEY" "https://idolvoice.karaokeicool.vn/songs?q=hen%20yeu&limit=20"

# Trang tiếp theo
curl -H "X-API-Key: YOUR_API_KEY" "https://idolvoice.karaokeicool.vn/songs?limit=50&offset=50"
```

**Response `200`:**
```json
{
  "total": 2,
  "limit": 20,
  "offset": 0,
  "count": 2,
  "songs": [
    {"id": 107929, "name": "HẸN YÊU", "duration": 301, "version": 26052200},
    {"id": 104717, "name": "HẸN YÊU", "duration": 328, "version": 25102800}
  ]
}
```

> ⚠️ Bài thuộc build **mới nhất** có thể chưa đồng bộ lên media server. Trước khi `/convert`, nên xác nhận bằng `/check_song/{id}`.

---

## 8. `GET /check_song/{song_id}` — Kiểm tra bài có sẵn để convert

```bash
curl -H "X-API-Key: YOUR_API_KEY" https://idolvoice.karaokeicool.vn/check_song/107929
```

**Response — bài dùng được:**
```json
{"song_id": "107929", "available": true, "size_mb": 9.2, "reason": null}
```

**Response — bài chưa đồng bộ:**
```json
{"song_id": "108578", "available": false, "size_mb": null,
 "reason": "Bài chưa xuất bản/đồng bộ lên media server (version v/0)"}
```

Luồng khuyến nghị cho app: `/songs?q=...` cho khách chọn bài → `/check_song/{id}` xác nhận → `/convert`.

---

## 9. `POST /api/files/upload/audio` — Upload file ghi âm của khách lên NAS MinIO

Nhận file ghi âm từ phòng karaoke và lưu trữ: **file audio + ảnh bìa lưu lên MinIO**
(container `minio-records` trên NAS Synology), **thông tin bản ghi lưu vào Postgres**
(bảng `rvc_recorded_files`).

Endpoint này **giữ nguyên contract** của hệ thống karaoke (cùng đường dẫn, cùng form field,
**không yêu cầu `X-API-Key`**) — phòng hát chỉ cần đổi `url` trong nhóm `recording` của
remote config sang server này là chạy, **không sửa app karaoke**.

**Content-Type:** `multipart/form-data`

| Field | Kiểu | Bắt buộc | Client karaoke tự gửi |
|---|---|---|---|
| `audio` | file | ✅ | File ghi âm `.wav` (tên file = tên bài hát) |
| `image` | file | ❌ | Ảnh bìa bài hát (tải từ hệ thống) |
| `name` | text | ❌ | Tên bài hát |
| `id` | text | ❌ | ID bài hát |
| `cluster_id` | text | ❌ | Mã chi nhánh (`roomConfig.cluster`) |
| `room_code` | text | ❌ | Mã phòng (`roomConfig.code`) |
| `singer_name` | text | ❌ | Tên ca sĩ (hoặc `Youtube`/`Unknown`) |
| `is_4k` | text | ❌ | Luôn `"true"` |
| `created_time` | text | ❌ | `YYYY-MM-DD HH:mm:ss` giờ VN (không gửi → server lấy giờ hiện tại) |

```bash
curl -X POST https://idolvoice.karaokeicool.vn/api/files/upload/audio \
  -F "audio=@ghi_am.wav;type=audio/mpeg" \
  -F "image=@anh_bia.jpg" \
  -F "name=Đoạn Tuyệt Nàng Đi" \
  -F "id=103666" \
  -F "cluster_id=IVP" \
  -F "room_code=HUH" \
  -F "created_time=2026-07-15 15:01:46" \
  -F "is_4k=true" \
  -F "singer_name=Anh Tuan"
```

**Response `200`:**
```json
{
  "status": "uploaded",
  "record_id": 12,
  "bucket": "customer-records",
  "audio_object": "IVP/HUH/2026-07-15/080146_23b9df25_103666.mp3",
  "image_object": "IVP/HUH/2026-07-15/080146_23b9df25_103666.jpg",
  "size_mb": 5.18,
  "created_time": "2026-07-15 15:01:46"
}
```

**Cách lưu trữ:**
- Object trên MinIO đặt theo `{cluster_id}/{room_code}/{ngày}/{giờ}_{uuid}_{id bài}.{đuôi}`;
  ảnh bìa cùng tên, khác đuôi. Các field đi kèm được gắn vào metadata của object.
- Mỗi lần upload thêm 1 dòng vào bảng Postgres `rvc_recorded_files`
  (`media_id`, `name`, `cluster_id`, `room_code`, `singer_name`, `is_4k`, `created_time`,
  `bucket`, `audio_object`, `image_object`, `audio_size`, `uploaded_at`) —
  `record_id` trong response là id của dòng này. Bảng tự tạo khi server khởi động.
- **Xem/tải file:** dùng MinIO Console `http://172.16.20.12:9101` (Object Browser →
  bucket `customer-records`) hoặc S3 API port `9100`. **Không** duyệt trực tiếp thư mục
  `minio/minio-records` trên NAS bằng File Station — MinIO lưu theo định dạng nội bộ
  (mỗi object là 1 folder chứa `xl.meta`), sửa/xóa tay sẽ hỏng dữ liệu.

**Xử lý lỗi phía client karaoke:** app chỉ kiểm tra HTTP status — `2xx` = thành công
(xóa bản ghi khỏi hàng đợi upload); khác `2xx` = giữ lại trong ObjectBox và **tự upload lại
khi app khởi động** (bản ghi quá 1 tuần sẽ bị bỏ). Vì vậy server lỗi tạm thời không làm mất
file ghi âm.

**Cấu hình server (biến môi trường / `.env`):**

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `MINIO_ENDPOINT` | `172.16.20.12:9100` | S3 API của MinIO (KHÔNG phải port console 9101) |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | — (bắt buộc) | Tài khoản MinIO |
| `MINIO_BUCKET` | `customer-records` | Bucket lưu ghi âm (tự tạo nếu chưa có) |
| `MINIO_SECURE` | `0` | `1` nếu MinIO chạy HTTPS |
| `RECORD_TABLE` | `rvc_recorded_files` | Tên bảng Postgres lưu bản ghi |

> 🔒 **Bảo mật:** endpoint này mở (không API key) để tương thích client karaoke. Nếu domain
> public và muốn hạn chế, chặn ở tầng reverse-proxy (chỉ cho IP nội bộ gọi
> `/api/files/upload/audio`) — không cần đụng vào client.

---

## 10. Danh sách bài đã convert — nghe thử rồi tải

Mỗi lần task `/convert` hoàn tất, kết quả được ghi vào bảng `rvc_converted_songs`.
Bộ 3 endpoint này cho app hiển thị danh sách bản đã convert của khách, **phát nghe thử**
rồi mới **tải về** — không phụ thuộc `task_id` (vốn mất khi server restart).

### `GET /conversions/{customer_id}` — Danh sách bản đã convert của khách

| Tham số (query) | Mặc định | Mô tả |
|---|---|---|
| `limit` | 50 | Số bản mỗi trang (tối đa 200) |
| `offset` | 0 | Phân trang |

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  "https://idolvoice.karaokeicool.vn/conversions/KH001?limit=20"
```

**Response `200`:**
```json
{
  "customer_id": "KH001",
  "total": 2, "limit": 20, "offset": 0, "count": 2,
  "conversions": [
    {
      "conversion_id": 2,
      "song_id": "103691",
      "song_name": "HẸN YÊU",
      "pitch_shift": 0,
      "size_mb": 4.5,
      "created_at": "2026-07-16 10:20:00",
      "available": true,
      "stream_url": "/conversions/2/stream",
      "download_url": "/conversions/2/download"
    },
    {
      "conversion_id": 1,
      "song_id": null,
      "song_name": "bai_upload",
      "pitch_shift": 0,
      "size_mb": 5.1,
      "created_at": "2026-07-01 09:00:00",
      "available": false,
      "stream_url": "/conversions/1/stream",
      "download_url": "/conversions/1/download"
    }
  ]
}
```

`available: false` = file kết quả đã bị dọn (giữ tối đa **90 ngày / 3 tháng**) — gọi `/convert` lại nếu khách muốn nghe.

### `GET /conversions/{id}/stream` — Nghe thử (phát trực tiếp)

Trả file audio **inline** để phát ngay. Ngoài header `X-API-Key`, endpoint nhận key qua
**query `?api_key=`** — vì thẻ `<audio>` của trình duyệt không gửi được header tùy chỉnh:

```html
<audio controls src="https://idolvoice.karaokeicool.vn/conversions/2/stream?api_key=YOUR_API_KEY"></audio>
```

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  "https://idolvoice.karaokeicool.vn/conversions/2/stream" -o nghe_thu.mp3
```

### `GET /conversions/{id}/download` — Tải về

Như `/stream` nhưng trả `Content-Disposition: attachment` (trình duyệt tải file thay vì phát),
tên file = tên bài hát. Cũng nhận key qua header hoặc `?api_key=`.

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  "https://idolvoice.karaokeicool.vn/conversions/2/download" -O -J
```

**Lỗi riêng của nhóm này:** `404` = `conversion_id` không tồn tại; `410` = bản ghi còn trong
DB nhưng file đã bị dọn sau 90 ngày.

---

## 11. API cũ (giữ để tương thích)

### `POST /run_upload` — Train + convert trong 1 lần gọi
```bash
curl -X POST https://idolvoice.karaokeicool.vn/run_upload \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "model_name=TenModel" \
  -F "epochs=150" -F "pitch_shift=0" -F "force_retrain=false" \
  -F "target_song_id=103666" \
  -F "training_files=@giong.wav"
```
(`target_song_id` hoặc `target_song=@file`; các field khác như `/train` + `/convert` gộp lại.)

### `POST /run` — Như trên nhưng dùng đường dẫn file có sẵn trên server (JSON)
```bash
curl -X POST https://idolvoice.karaokeicool.vn/run \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"training_files":["/app/dataset/giong.wav"],"target_song_path":"/app/audios/bai.mp3","model_name":"TenModel","epochs":150,"pitch_shift":0,"force_retrain":false}'
```

> Khuyến nghị dùng bộ API mới `/train` + `/convert` — tách bạch, tái sử dụng model theo khách hàng.

---

## 📋 Bảng tổng hợp endpoint

| # | Method | Endpoint | Chức năng | API Key |
|---|---|---|---|---|
| 1 | GET | `/health` | Kiểm tra server sống | ❌ |
| 2 | POST | `/train` | Train model giọng theo khách hàng | ✅ |
| 3 | GET | `/model/{customer_id}` | Kiểm tra model của khách | ✅ |
| 4 | POST | `/convert` | Đổi giọng bài hát bằng model của khách | ✅ |
| 5 | GET | `/status/{task_id}` | Trạng thái + log task | ✅ |
| 6 | GET | `/download/{task_id}` | Tải file kết quả | ✅ |
| 7 | GET | `/songs` | Danh sách / tìm kiếm bài hát | ✅ |
| 8 | GET | `/check_song/{song_id}` | Kiểm tra bài có sẵn để convert | ✅ |
| 9 | POST | `/api/files/upload/audio` | Upload file ghi âm lên NAS 25 (MinIO) + lưu DB | ❌ |
| 10 | GET | `/conversions/{customer_id}` | Danh sách bản đã convert của khách | ✅ |
| 11 | GET | `/conversions/{id}/stream` | Nghe thử bản convert (phát inline) | ✅ (hoặc `?api_key=`) |
| 12 | GET | `/conversions/{id}/download` | Tải bản convert về | ✅ (hoặc `?api_key=`) |
| 13 | POST | `/run_upload` | (Cũ) Train + convert 1 lần | ✅ |
| 14 | POST | `/run` | (Cũ) Như trên, input là path trên server | ✅ |

---

## ❗ Mã lỗi thường gặp

| HTTP | Ý nghĩa | Cách xử lý |
|---|---|---|
| 400 | Thiếu tham số (vd: không gửi `target_song_id` lẫn `target_song`) / task chưa `completed` khi download | Kiểm tra request |
| 401 | Sai hoặc thiếu `X-API-Key` | Gửi đúng header |
| 404 | Khách chưa có model / `task_id` không tồn tại / bài hát không có media / `conversion_id` không tồn tại | Train trước; kiểm tra id |
| 410 | Bản convert còn trong DB nhưng file đã bị dọn (quá 90 ngày) | Gọi `/convert` lại |
| 502 | Không lấy được audio theo `target_song_id` (bài chưa xuất bản `v/0`, media server lỗi) / không kết nối được MinIO | Dùng id bài đã xuất bản hoặc upload file; kiểm tra NAS 25 |
| 503 | Server chưa cấu hình DB hoặc MinIO (`MINIO_ACCESS_KEY`/`MINIO_SECRET_KEY`) | Bổ sung biến môi trường rồi restart |
| 500 | Lỗi xử lý nội bộ; riêng upload ghi âm: file đã lên MinIO nhưng ghi DB thất bại (detail có ghi rõ) | Xem `logs` trong `/status` hoặc `detail`, báo quản trị |

Task `failed` (qua `/status`): nguyên nhân phổ biến — dữ liệu train < 60 giây giọng thực tế, file âm thanh hỏng, hết VRAM (job trước quá nặng). Đọc trường `logs` để biết chi tiết.

---

## 🧩 Ví dụ tích hợp — luồng đầy đủ (bash)

```bash
BASE="https://idolvoice.karaokeicool.vn"
KEY="YOUR_API_KEY"
CUS="KH001"

# 1) Khách đã có model chưa?
TRAINED=$(curl -s -H "X-API-Key: $KEY" "$BASE/model/$CUS" | grep -o '"trained":[a-z]*' | cut -d: -f2)

# 2) Chưa có -> train
if [ "$TRAINED" != "true" ]; then
  TASK=$(curl -s -X POST "$BASE/train" -H "X-API-Key: $KEY" \
    -F "customer_id=$CUS" -F "epochs=150" \
    -F "training_files=@ghi_am.wav" | grep -o '"task_id":"[^"]*"' | cut -d'"' -f4)
  echo "Training task: $TASK"
  while :; do
    S=$(curl -s -H "X-API-Key: $KEY" "$BASE/status/$TASK" | grep -o '"status":"[^"]*"' | head -1 | cut -d'"' -f4)
    echo "train: $S"; [ "$S" = "completed" ] && break
    [ "$S" = "failed" ] && exit 1
    sleep 15
  done
fi

# 3) Convert bài hát
TASK=$(curl -s -X POST "$BASE/convert" -H "X-API-Key: $KEY" \
  -F "customer_id=$CUS" -F "target_song_id=103691" -F "pitch_shift=0" \
  | grep -o '"task_id":"[^"]*"' | cut -d'"' -f4)
echo "Convert task: $TASK"
while :; do
  S=$(curl -s -H "X-API-Key: $KEY" "$BASE/status/$TASK" | grep -o '"status":"[^"]*"' | head -1 | cut -d'"' -f4)
  echo "convert: $S"; [ "$S" = "completed" ] && break
  [ "$S" = "failed" ] && exit 1
  sleep 15
done

# 4) Tải kết quả
curl -H "X-API-Key: $KEY" "$BASE/download/$TASK" -o ket_qua.mp3
echo "Xong: ket_qua.mp3"
```

---

## 📝 Ghi chú vận hành

- **`pitch_shift`** (nửa cung): cùng giới tính `0`; model nam hát bài ca sĩ nữ `-12`; model nữ hát bài ca sĩ nam `+12`; lệch nhẹ thử `±3..6`.
- **Model theo khách được giữ vĩnh viễn** (`assets/weights/` + DB) — convert các lần sau không cần train lại.
- File upload input tự xóa sau khi task xong; file **kết quả convert** trong `audios/` giữ **90 ngày (3 tháng)** rồi tự xóa (đổi bằng biến `RESULT_RETENTION_DAYS`); file trung gian (tách beat/vocal...) dọn sau **10 ngày** (`RETENTION_DAYS`).
- Server xử lý tuần tự — nhiều request cùng lúc sẽ xếp hàng (xem `queue_size` trong response).
- Task registry nằm trong RAM: restart server làm mất `task_id` đang theo dõi (model/kết quả trên đĩa không mất).
