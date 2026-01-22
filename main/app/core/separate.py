import os
import sys
import subprocess

sys.path.append(os.getcwd())

from main.app.core.ui import gr_info, gr_warning
from main.app.variables import python, translations, configs

def separate_music(
    drop_audio_files,
    input_path,
    output_dirs,
    export_format, 
    model_name, 
    karaoke_model,
    reverb_model,
    denoise_model,
    sample_rate,
    shifts, 
    batch_size, 
    overlap, 
    aggression,
    hop_length, 
    window_size,
    segments_size, 
    post_process_threshold,
    enable_tta,
    enable_denoise, 
    high_end_process,
    enable_post_process,
    separate_backing,
    separate_reverb
):
    output_dirs = os.path.dirname(output_dirs) or output_dirs
    
    # Xác định danh sách file cần xử lý
    files_to_process = []
    
    # Ưu tiên sử dụng các file từ selected_files_list (State) nếu có
    if drop_audio_files:
        # drop_audio_files là một list các file paths (từ State)
        if isinstance(drop_audio_files, list):
            for file_path in drop_audio_files:
                if file_path and isinstance(file_path, str) and os.path.exists(file_path) and os.path.isfile(file_path):
                    files_to_process.append(file_path)
        elif isinstance(drop_audio_files, str):
            # Trường hợp chỉ có 1 file path string
            if os.path.exists(drop_audio_files) and os.path.isfile(drop_audio_files):
                files_to_process.append(drop_audio_files)
    
    # Debug log
    if files_to_process:
        gr_info(f"Tìm thấy {len(files_to_process)} file từ danh sách đã chọn: {[os.path.basename(f) for f in files_to_process]}")
    
    # Nếu không có file từ drop_audio, sử dụng input_path
    if not files_to_process:
        if input_path and os.path.exists(input_path):
            if os.path.isfile(input_path):
                files_to_process.append(input_path)
            elif os.path.isdir(input_path):
                # Nếu là thư mục, lấy tất cả file audio trong đó
                for f in os.listdir(input_path):
                    file_path = os.path.join(input_path, f)
                    if os.path.isfile(file_path) and os.path.splitext(f)[1].lower() in (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"):
                        files_to_process.append(file_path)
    
    if not files_to_process:
        gr_warning(translations["input_not_valid"])
        return [None]*4 + [""]
    
    if not os.path.exists(output_dirs): 
        gr_warning(translations["output_not_valid"])
        return [None]*4 + [""]

    if not os.path.exists(output_dirs): 
        os.makedirs(output_dirs)
    
    status_messages = []
    status_messages.append(f"Bắt đầu xử lý {len(files_to_process)} file...\n")
    
    # Xử lý từng file
    last_outputs = [None]*4
    for idx, file_path in enumerate(files_to_process, 1):
        try:
            status_messages.append(f"[{idx}/{len(files_to_process)}] Đang xử lý: {os.path.basename(file_path)}...")
            gr_info(f"Đang xử lý file {idx}/{len(files_to_process)}: {os.path.basename(file_path)}")
            
            subprocess.run([
                python, configs["separate_path"], 
                "--input_path", file_path,
                "--output_dirs", output_dirs,
                "--export_format", export_format,
                "--model_name", model_name,
                "--karaoke_model", karaoke_model,
                "--reverb_model", reverb_model,
                "--denoise_model", denoise_model,
                "--sample_rate", str(sample_rate),
                "--shifts", str(shifts),
                "--batch_size", str(batch_size),
                "--overlap", str(overlap),
                "--aggression", str(aggression),
                "--hop_length", str(hop_length),
                "--window_size", str(window_size),
                "--segments_size", str(segments_size),
                "--post_process_threshold", str(post_process_threshold),
                "--enable_tta", str(enable_tta),
                "--enable_denoise", str(enable_denoise),
                "--high_end_process", str(high_end_process),
                "--enable_post_process", str(enable_post_process),
                "--separate_backing", str(separate_backing),
                "--separate_reverb", str(separate_reverb),
            ], check=True)
            
            filename, _ = os.path.splitext(os.path.basename(file_path))
            file_output_dir = os.path.join(output_dirs, filename)
            
            # Lưu output của file cuối cùng để hiển thị
            last_outputs = [
                os.path.join(
                    file_output_dir, 
                    f"Original_Vocals_No_Reverb.{export_format}" if separate_reverb else f"Original_Vocals.{export_format}"
                ) if os.path.exists(os.path.join(file_output_dir, f"Original_Vocals_No_Reverb.{export_format}" if separate_reverb else f"Original_Vocals.{export_format}")) else None,
                os.path.join(
                    file_output_dir, 
                    f"Instruments.{export_format}"
                ) if os.path.exists(os.path.join(file_output_dir, f"Instruments.{export_format}")) else None,
                os.path.join(
                    file_output_dir, 
                    f"Main_Vocals_No_Reverb.{export_format}" if separate_reverb else f"Main_Vocals.{export_format}"
                ) if separate_backing and os.path.exists(os.path.join(file_output_dir, f"Main_Vocals_No_Reverb.{export_format}" if separate_reverb else f"Main_Vocals.{export_format}")) else None,
                os.path.join(
                    file_output_dir, 
                    f"Backing_Vocals.{export_format}"
                ) if separate_backing and os.path.exists(os.path.join(file_output_dir, f"Backing_Vocals.{export_format}")) else None
            ]
            
            status_messages.append(f"✓ Hoàn thành: {os.path.basename(file_path)}\n")
            
        except subprocess.CalledProcessError as e:
            status_messages.append(f"✗ Lỗi khi xử lý {os.path.basename(file_path)}: {str(e)}\n")
            gr_warning(f"Lỗi khi xử lý file {os.path.basename(file_path)}")
        except Exception as e:
            status_messages.append(f"✗ Lỗi khi xử lý {os.path.basename(file_path)}: {str(e)}\n")
            gr_warning(f"Lỗi khi xử lý file {os.path.basename(file_path)}")
    
    status_messages.append(f"\nĐã hoàn thành xử lý {len(files_to_process)} file!")
    status_text = "".join(status_messages)
    
    gr_info(translations["success"])
    
    return last_outputs + [status_text]