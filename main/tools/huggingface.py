import os
import time
import tqdm
import requests

try:
    import wget
except:
    wget = None

def HF_download_file(url, output_path=None, max_retries=3, retry_delay=5):
    """
    Tải file từ HuggingFace với retry mechanism
    """
    url = url.replace("/blob/", "/resolve/").replace("?download=true", "").strip()
    output_path = os.path.basename(url) if output_path is None else (os.path.join(output_path, os.path.basename(url)) if os.path.isdir(output_path) else output_path)

    # Tạo thư mục nếu chưa có
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Kiểm tra file đã tồn tại chưa (logic cũ bị bỏ để allow resume)
    # Block cũ: return ngay nếu file > 0. Giờ sẽ để xuống dưới check header Range
    pass

    # Thử tải bằng requests (có retry và progress bar tốt hơn)
    for attempt in range(max_retries):
        try:
            # Kiểm tra file đã tải một phần để resume
            resume_header = {}
            existing_size = 0
            if os.path.exists(output_path):
                existing_size = os.path.getsize(output_path)
                if existing_size > 0:
                    resume_header = {'Range': f'bytes={existing_size}-'}
                    print(f"Resume tải file từ byte {existing_size}")

            # Tăng timeout và giảm chunk size để ổn định hơn
            response = requests.get(url, stream=True, timeout=600, headers=resume_header)
            
            # Handle 416 Range Not Satisfiable (nghĩa là file đã tải xong rồi)
            if response.status_code == 416:
                print(f"File đã tải đầy đủ (Server trả về 416): {output_path}")
                return output_path

            if response.status_code == 200 or response.status_code == 206:  # 206 = Partial Content
                total_size = int(response.headers.get("content-length", 0))
                if existing_size > 0 and response.status_code == 206:
                    total_size = existing_size + total_size
                
                mode = "ab" if existing_size > 0 else "wb"
                progress_bar = tqdm.tqdm(
                    total=total_size, 
                    initial=existing_size,
                    desc=os.path.basename(url), 
                    ncols=100, 
                    unit="B", 
                    unit_scale=True,
                    unit_divisor=1024,
                    leave=True
                )

                # Giảm chunk size xuống 1MB để ổn định hơn với mạng không ổn định
                chunk_size = 1024 * 1024  # 1MB chunks
                with open(output_path, mode) as f:
                    try:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                progress_bar.update(len(chunk))
                                f.write(chunk)
                                f.flush()  # Force write to disk ngay lập tức
                    except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as e:
                        # Nếu bị lỗi, lưu phần đã tải để có thể resume
                        f.flush()
                        raise e

                progress_bar.close()
                
                # Kiểm tra file đã tải đầy đủ
                final_size = os.path.getsize(output_path)
                if total_size == 0:
                    # Nếu không biết tổng size, coi như thành công
                    print(f"✓ Tải hoàn tất: {output_path} ({final_size / (1024*1024):.2f} MB)")
                    return output_path
                elif final_size == total_size:
                    print(f"✓ Tải thành công: {output_path} ({final_size / (1024*1024):.2f} MB)")
                    return output_path
                elif total_size > 0 and final_size >= total_size * 0.99:  # Cho phép sai số 1%
                    print(f"✓ Tải gần như hoàn tất: {output_path} ({final_size / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)")
                    return output_path
                else:
                    # File chưa đầy đủ, raise exception để retry
                    raise Exception(f"File tải không đầy đủ. Kỳ vọng: {total_size} bytes ({total_size / (1024*1024):.2f} MB), Thực tế: {final_size} bytes ({final_size / (1024*1024):.2f} MB)")
                    
            else:
                raise ValueError(f"HTTP {response.status_code}: {response.text[:100]}")
                
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            current_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            print(f"Lần thử {attempt + 1}/{max_retries} thất bại: {str(e)}")
            print(f"Đã tải được: {current_size / (1024*1024):.2f} MB")
            
            if attempt < max_retries - 1:
                # Tăng thời gian đợi sau mỗi lần thử
                wait_time = retry_delay * (attempt + 1)
                print(f"Đợi {wait_time} giây trước khi thử lại...")
                time.sleep(wait_time)
            else:
                current_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
                if current_size > 1024 * 1024:  # Nếu đã tải được > 1MB
                    print(f"\n⚠️  Cảnh báo: File tải không đầy đủ ({current_size / (1024*1024):.2f} MB)")
                    print(f"💡 Gợi ý:")
                    print(f"   1. Chạy lại chức năng để tiếp tục tải (resume)")
                    print(f"   2. Hoặc tải thủ công từ: {url}")
                    print(f"   3. Đặt file vào: {os.path.dirname(output_path)}")
                
        except Exception as e:
            print(f"Lần thử {attempt + 1}/{max_retries} thất bại: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                print(f"Đợi {wait_time} giây trước khi thử lại...")
                time.sleep(wait_time)
            else:
                # Nếu vẫn lỗi, thử dùng wget (fallback)
                if wget is not None:
                    try:
                        print("Thử tải bằng wget...")
                        wget.download(url, out=output_path)
                        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                            return output_path
                    except Exception as wget_error:
                        print(f"Wget cũng thất bại: {wget_error}")
                
                raise Exception(f"Không thể tải file sau {max_retries} lần thử. Lỗi cuối: {str(e)}")

    return output_path