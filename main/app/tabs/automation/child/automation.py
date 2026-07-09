import os
import sys
import shutil
import librosa
import gradio as gr
from main.app.variables import translations, configs
from main.app.core.ui import gr_info, gr_warning, gr_error
from main.app.core.separate import separate_music
from main.app.core.training import preprocess, extract, create_index, training
from main.app.core.inference import convert_audio
from main.app.tabs.training.child.training import get_next_cos_name

def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def _pick_latest_model_file(model_name: str) -> str | None:
    weights_dir = configs.get("weights_path", os.path.join("assets", "weights"))
    if not os.path.exists(weights_dir):
        return None

    candidates = [
        f for f in os.listdir(weights_dir)
        if f.startswith(model_name) and f.endswith(".pth")
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda f: os.path.getmtime(os.path.join(weights_dir, f)), reverse=True)
    return candidates[0]

def check_dataset_duration(dataset_dir: str) -> float:
    total_duration = 0.0
    if not os.path.exists(dataset_dir):
        return 0.0
    
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".wav"):
                try:
                    path = os.path.join(root, file)
                    # Load audio without changing sr to keep it fast/accurate to file
                    y, sr = librosa.load(path, sr=None)
                    # Detect non-silent intervals (top_db=40 is a reasonable default for vocals)
                    intervals = librosa.effects.split(y, top_db=40)
                    # Calculate duration
                    non_silent_duration = sum((end - start) for start, end in intervals) / sr
                    total_duration += non_silent_duration
                except Exception as e:
                    print(f"Error checking duration for {file}: {e}")
                    
    return total_duration

def _pick_index_file(model_name: str) -> str:
    logs_dir = os.path.join(configs.get("logs_path", os.path.join("assets", "logs")), model_name)
    if not os.path.exists(logs_dir):
        return ""

    # Ưu tiên index kiểu "added_*.index"
    idx = [f for f in os.listdir(logs_dir) if f.endswith(".index") and "added" in f]
    if idx:
        idx.sort(key=lambda f: os.path.getmtime(os.path.join(logs_dir, f)), reverse=True)
        return os.path.join(logs_dir, idx[0])

    # Fallback: lấy index mới nhất nếu không có "added"
    all_idx = [f for f in os.listdir(logs_dir) if f.endswith(".index")]
    if not all_idx:
        return ""
    all_idx.sort(key=lambda f: os.path.getmtime(os.path.join(logs_dir, f)), reverse=True)
    return os.path.join(logs_dir, all_idx[0])

def automation_workflow(
    training_files, 
    target_song, 
    model_name, 
    epochs,
    pitch_shift,
    force_retrain
):
    if not training_files:
        return None, "Lỗi: Chưa chọn file giọng hát để train."
    if not target_song:
        return None, "Lỗi: Chưa chọn bài hát của ca sĩ để đổi giọng."
    if not model_name:
        return None, "Lỗi: Chưa đặt tên mô hình."

    logs = []
    def log(msg):
        logs.append(msg)
        return "\n".join(logs)

    try:
        # Define paths globally for the workflow
        audios_root = configs.get("audios_path", "audios")
        dataset_root = "dataset"
        dataset_train_ready = os.path.join(dataset_root, model_name)

        # Check if model exists for Reuse Logic
        latest_model = _pick_latest_model_file(model_name)
        skip_training = False
        
        if latest_model:
            if force_retrain:
                yield None, log(f"Phát hiện model cũ '{latest_model}' nhưng chạy lại theo yêu cầu.")
            else:
                 yield None, log(f"Phát hiện model cũ '{latest_model}'. Bỏ qua bước train.")
                 skip_training = True

        # =================================================================================
        # BƯỚC 1: TÁCH GIỌNG TRAIN (DATASET)
        # =================================================================================
        if not skip_training:
            yield None, log(f"== BẮT ĐẦU BƯỚC 1: TÁCH DATASET CHO MODEL {model_name} ==")
        
            # Tạo thư mục dataset tạm thời
            dataset_dir = os.path.join(dataset_root, model_name)
            if os.path.exists(dataset_dir):
                shutil.rmtree(dataset_dir)
            os.makedirs(dataset_dir, exist_ok=True)

            # Di chuyển file upload vào thư mục tạm để xử lý (nếu cần) hoặc dùng trực tiếp
            # separate_music expects a list of file paths or a directory
            # Vì separate_music output ra structure riêng, ta sẽ dùng output_dirs là dataset_dir
            # Tuy nhiên separate_music tạo subfolder cho mỗi bài hát. 
            # Để đơn giản cho training, ta cần gom tất cả 'Vocals' vào 1 folder dataset model.
            
            # Tách từng file một và gom vocal
            # model_name used in path defined above
            
            temp_separate_dir = os.path.join(audios_root, f"temp_train_{model_name}")
            stub_dir = os.path.join(temp_separate_dir, "stub")
            os.makedirs(stub_dir, exist_ok=True)
            
            # Gọi tách nhạc
            # Support both Gradio file objs and string paths
            file_paths = []
            for f in training_files:
                if isinstance(f, str):
                    file_paths.append(f)
                elif hasattr(f, 'name'):
                    file_paths.append(f.name)
            
            yield None, log(f"Đang tách {len(file_paths)} file giọng train...")
            
            # Optimize: Skip instrumental denoising for training data
            os.environ["SKIP_INST_DENOISE"] = "1"
            
            separate_music(
                drop_audio_files=file_paths,
                input_path="",
                output_dirs=os.path.join(stub_dir, "stub"),
                export_format="wav",
                model_name="HP-Vocal-1",
                karaoke_model="", reverb_model="MDX-Reverb", denoise_model="Lite",
                sample_rate=44100, shifts=2, batch_size=1, overlap=0.25, aggression=10, 
                hop_length=1024, window_size=512, segments_size=256, post_process_threshold=0.2,
                enable_tta=False, enable_denoise=True, high_end_process=False, enable_post_process=False,
                separate_backing=False, separate_reverb=True # Tách reverb để lấy Original_Vocals_No_Reverb
            )
            
            # Gom Vocals vào dataset folder
            os.makedirs(dataset_train_ready, exist_ok=True)
            count_files = 0
            for root, dirs, files in os.walk(temp_separate_dir):
                for file in files:
                    # Chỉ lấy file Original_Vocals_No_Reverb
                    if "Original_Vocals_No_Reverb" in file and file.endswith(".wav"):
                        # Move and rename unique
                        src = os.path.join(root, file)
                        dst = os.path.join(dataset_train_ready, f"{count_files}.wav")
                        shutil.move(src, dst)
                        count_files += 1
            
            # Dọn dẹp temp
            shutil.rmtree(temp_separate_dir, ignore_errors=True)
            os.environ["SKIP_INST_DENOISE"] = "0"
            
            if count_files == 0:
                 yield None, log(f"Lỗi: Không tìm thấy file giọng tách được trong {temp_separate_dir}. Vui lòng kiểm tra lại log console.")
                 return

            # Quality Check: Duration (Silence Awareness)
            yield None, log(f"Đang kiểm tra chất lượng dữ liệu train (loại bỏ khoảng lặng)...")
            effective_duration = check_dataset_duration(dataset_train_ready)
            if effective_duration < 60:
                 yield None, log(f"🛑 LỖI: Tổng thời lượng giọng hát thực tế (đã trừ khoảng lặng) là {effective_duration:.2f}s.\n"
                                 f"Hệ thống yêu cầu tối thiểu 60s để đảm bảo chất lượng model.\n"
                                 f"Vui lòng thêm file giọng hát hoặc dùng file dài hơn.")
                 return
                 
            yield None, log(f"✅ Dữ liệu hợp lệ: {count_files} files, Thời lượng thực tế: {effective_duration:.2f}s")
            yield None, log(f"Đã chuẩn bị xong dữ liệu train.")
        else:
             # Need to define dataset_train_ready for Training step even if skipped?
             # Training step expects 'dataset_train_ready' as argument to preprocess.
             # But Step 3 is ALSO skipped if skip_training is True.
             # So we are good.
             yield None, log("⏩ Bỏ qua Bước 1 (Tách dataset) vì đang dùng lại model cũ.")

        # =================================================================================
        # BƯỚC 2: HUẤN LUYỆN MÔ HÌNH
        # =================================================================================
        # =================================================================================
        # BƯỚC 2: XỬ LÝ BÀI HÁT ĐÍCH (Được đưa lên trước Training)
        # =================================================================================
        yield None, log(f"== BẮT ĐẦU BƯỚC 2: XỬ LÝ BÀI HÁT ĐÍCH ==")
        
        target_path = target_song if isinstance(target_song, str) else target_song.name
        target_filename = os.path.splitext(os.path.basename(target_path))[0]
        output_target_dir = os.path.join(audios_root, target_filename)
        
        yield None, log(f"Đang tách nhạc bài: {target_filename}...")
        
        # Tạo stub cho audios để workaround lỗi cắt path
        audios_stub = os.path.join(audios_root, "stub")
        os.makedirs(audios_stub, exist_ok=True)

        separate_music(
            drop_audio_files=target_path,
            input_path="",
            output_dirs=os.path.join(audios_stub, "stub"), # separate_music creates subfolder automatically inside audios
            export_format="mp3",
            model_name="HP-Vocal-1",
            karaoke_model="", reverb_model="MDX-Reverb", denoise_model="Lite",
            sample_rate=44100, shifts=2, batch_size=1, overlap=0.25, aggression=10, 
            hop_length=1024, window_size=512, segments_size=256, post_process_threshold=0.2,
            enable_tta=False, enable_denoise=True, high_end_process=False, enable_post_process=False,
            separate_backing=False, separate_reverb=True # Tách reverb để lấy sạch
        )
        
        # Tìm file Vocal và Instrument
        # separate output structure: audios/<filename>/...
        # File names: Original_Vocals_No_Reverb.mp3 (if dereverb), Instruments.mp3
        vocal_file = os.path.join(output_target_dir, "Original_Vocals_No_Reverb.mp3")
        if not os.path.exists(vocal_file):
             vocal_file = os.path.join(output_target_dir, "Original_Vocals.mp3")
        
        instrument_file = os.path.join(output_target_dir, "Instruments.mp3")
        
        if not os.path.exists(vocal_file) or not os.path.exists(instrument_file):
            yield None, log("Lỗi: Không tìm thấy file tách nhạc (Vocal/Instrument).")
            return

        yield None, log("Tách nhạc thành công.")

        # =================================================================================
        # BƯỚC 3: HUẤN LUYỆN MÔ HÌNH
        # =================================================================================
        if not skip_training:
            yield None, log(f"== BẮT ĐẦU BƯỚC 3: HUẤN LUYỆN ({epochs} epochs) ==")
        
            # Preprocess
            yield None, log("Đang tiền xử lý...")
            for output in preprocess(
                model_name=model_name,
                dataset=dataset_train_ready, # Might be undefined if skipped, but this block is also skipped
                sample_rate="48k", # Default training sr
                cpu_core=os.cpu_count(),
                cut_preprocess="Automatic",
                process_effects=False, # Không effect
                clean_dataset=False, 
                clean_strength=0.7,
                chunk_len=3.0, overlap_len=0.3, normalization_mode="none"
            ):
                yield None, log(output)
            
            # Extract features
            yield None, log("Đang trích xuất đặc trưng (f0)...")
            for output in extract(
                model_name=model_name,
                version="v2",
                method="rmvpe",
                pitch_guidance=True,
                hop_length=160,
                cpu_cores=os.cpu_count(),
                gpu=0, # Auto detect usually logic handled inside
                sample_rate="48k",
                embedders="hubert_base",
                custom_embedders="",
                onnx_f0_mode=False,
                embedders_mode="fairseq",
                f0_autotune=False, f0_autotune_strength=1.0,
                hybrid_method="rmvpe", rms_extract=False, alpha=0.5
            ):
                 yield None, log(output)
            
            # Create Index
            yield None, log("Đang tạo chỉ mục (index)...")
            for output in create_index(model_name, "v2", "Auto"):
                 yield None, log(output)
            
            # Training
            yield None, log("Đang huấn luyện (Training)... Việc này có thể mất thời gian.")
            for output in training(
                model_name=model_name,
                rvc_version="v2",
                save_every_epoch=50,
                save_only_latest=True,
                save_every_weights=True,
                total_epoch=epochs,
                sample_rate="48k",
                batch_size=8, # Safe default
                gpu=0,
                pitch_guidance=True,
                not_pretrain=False, # Use default pretrained
                custom_pretrained=False,
                pretrain_g="", 
                pretrain_d="",
                detector=False,
                threshold=50,
                clean_up=False,
                cache=True,
                model_author="", 
                vocoder="Default",
                checkpointing=False,
                deterministic=False, 
                benchmark=False, 
                optimizer="AdamW",
                energy_use=False,
                custom_reference=False, 
                reference_name="",
                multiscale_mel_loss=False
            ):
                yield None, log(output)
            yield None, log("Huấn luyện hoàn tất!")
        else:
             yield None, log("⏩ Bỏ qua Bước 3 (Huấn luyện) vì đang dùng lại model cũ.")

        # =================================================================================
        # BƯỚC 4: CHUYỂN ĐỔI VÀ GHÉP (INFERENCE)
        # =================================================================================
        yield None, log(f"== BẮT ĐẦU BƯỚC 4: ĐỔI GIỌNG VÀ GHÉP NHẠC ==")
        
        # Tìm model mới nhất trong weights (tên có thể là cos02_150e_1800s.pth)
        model_pth = _pick_latest_model_file(model_name)
        if not model_pth:
            yield None, log(f"Lỗi: Không tìm thấy model .pth trong `{configs.get('weights_path', 'assets/weights')}` cho `{model_name}`.")
            return
        yield None, log(f"Đang dùng model: {model_pth}")

        # Tìm index file mới nhất (ưu tiên added_*.index)
        index_file = _pick_index_file(model_name)
        if index_file:
            yield None, log(f"Đang dùng index: {index_file}")
        else:
            yield None, log("Cảnh báo: Không tìm thấy file index, sẽ chạy không dùng index (chất lượng có thể kém hơn).")
        
        # Output path
        final_output_path = os.path.join(audios_root, f"{target_filename}_COVER_{model_name}.mp3")
        _ensure_dir(os.path.dirname(final_output_path))
        
        yield None, log(f"Đang đổi giọng và ghép beat... (Model: {model_name})")
        
        # Gọi convert_audio
        # Params: clean, autotune, use_audio... large signature
        # We use explicit arguments based on convert.py signature
        result_paths = convert_audio(
            clean=True, clean_strength=0.5,
            autotune=False,
            use_audio=False, # Input directly path
            use_original=False, convert_backing=False, not_merge_backing=False, merge_instrument=False,
            pitch=pitch_shift,
            model=model_pth,
            index=index_file, index_rate=0.75,
            input=vocal_file,
            output=final_output_path,
            format="mp3",
            method="rmvpe", hybrid_method="rmvpe", hop_length=160,
            embedders="hubert_base", custom_embedders="",
            resample_sr=0, filter_radius=3, rms_mix_rate=0.25, protect=0.33,
            split_audio=False, f0_autotune_strength=1.0, input_audio_name="",
            checkpointing=False, onnx_f0_mode=False,
            formant_shifting=False, formant_qfrency=1.0, formant_timbre=1.0,
            f0_file=None, embedders_mode="fairseq",
            proposal_pitch=False, proposal_pitch_threshold=255.0,
            audio_processing=False, alpha=0.5,
            mix_beat=True, beat_file=instrument_file,
            mix_auto_gain=True,
            add_echo=True, echo_wet=0.25, echo_delay_ms=125
        )
        
        if not result_paths or len(result_paths) < 7:
            yield None, log("Lỗi: convert_audio không trả về kết quả hợp lệ (có thể do thiếu model/đầu vào/đầu ra).")
            return

        # convert_audio returns list: [vocal, backing, merge_back, original, merge_inst, update, mix_result]
        final_mix = result_paths[6]
        
        if final_mix and os.path.exists(final_mix):
             yield final_mix, log(f"== HOÀN TẤT! FILE KẾT QUẢ: {final_mix} ==")
        else:
             # Log thêm thông tin để debug nhanh
             weights_dir = configs.get("weights_path", "assets/weights")
             logs_dir = configs.get("logs_path", "assets/logs")
             yield None, log(
                 "Lỗi: Không tạo được file kết quả cuối cùng.\n"
                 f"- model_pth: {model_pth} (weights_dir={weights_dir})\n"
                 f"- index: {index_file or '(none)'} (logs_dir={logs_dir})\n"
                 f"- vocal_file: {vocal_file} (exists={os.path.exists(vocal_file)})\n"
                 f"- instrument_file: {instrument_file} (exists={os.path.exists(instrument_file)})\n"
                 f"- output: {final_output_path}"
             )

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        yield None, log(f"LỖI KHÔNG MONG MUỐN:\n{err}")


def train_workflow(training_files, model_name, epochs, force_retrain=False):
    """API 1: Train model giọng từ file ghi âm của khách hàng (không cần bài hát).

    Kết thúc thành công sẽ yield (đường_dẫn_model_pth, log)."""
    if not training_files:
        yield None, "Lỗi: Chưa chọn file giọng hát để train."
        return
    if not model_name:
        yield None, "Lỗi: Chưa đặt tên mô hình."
        return

    logs = []
    def log(msg):
        logs.append(msg)
        return "\n".join(logs)

    try:
        audios_root = configs.get("audios_path", "audios")
        dataset_root = "dataset"
        dataset_train_ready = os.path.join(dataset_root, model_name)
        weights_dir = configs.get("weights_path", os.path.join("assets", "weights"))
        model_logs_dir = os.path.join(configs.get("logs_path", os.path.join("assets", "logs")), model_name)

        latest_model = _pick_latest_model_file(model_name)
        if latest_model and not force_retrain:
            yield os.path.join(weights_dir, latest_model), log(
                f"Model '{latest_model}' đã tồn tại. Dùng lại model cũ (gửi force_retrain=true nếu muốn train dữ liệu mới thay thế).")
            return

        if latest_model and force_retrain:
            yield None, log(f"Phát hiện model cũ '{latest_model}'. Xóa để train dữ liệu mới thay thế...")
            for f in os.listdir(weights_dir):
                if f.startswith(model_name) and f.endswith(".pth"):
                    try:
                        os.remove(os.path.join(weights_dir, f))
                    except Exception:
                        pass
            shutil.rmtree(model_logs_dir, ignore_errors=True)
            shutil.rmtree(dataset_train_ready, ignore_errors=True)

        # ================= BƯỚC 1: TÁCH GIỌNG TRAIN (DATASET) =================
        yield None, log(f"== BẮT ĐẦU TÁCH DATASET CHO MODEL {model_name} ==")

        dataset_dir = os.path.join(dataset_root, model_name)
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
        os.makedirs(dataset_dir, exist_ok=True)

        temp_separate_dir = os.path.join(audios_root, f"temp_train_{model_name}")
        stub_dir = os.path.join(temp_separate_dir, "stub")
        os.makedirs(stub_dir, exist_ok=True)

        file_paths = []
        for f in training_files:
            if isinstance(f, str):
                file_paths.append(f)
            elif hasattr(f, 'name'):
                file_paths.append(f.name)

        yield None, log(f"Đang tách {len(file_paths)} file giọng train...")

        os.environ["SKIP_INST_DENOISE"] = "1"

        separate_music(
            drop_audio_files=file_paths,
            input_path="",
            output_dirs=os.path.join(stub_dir, "stub"),
            export_format="wav",
            model_name="HP-Vocal-1",
            karaoke_model="", reverb_model="MDX-Reverb", denoise_model="Lite",
            sample_rate=44100, shifts=2, batch_size=1, overlap=0.25, aggression=10,
            hop_length=1024, window_size=512, segments_size=256, post_process_threshold=0.2,
            enable_tta=False, enable_denoise=True, high_end_process=False, enable_post_process=False,
            separate_backing=False, separate_reverb=True
        )

        os.makedirs(dataset_train_ready, exist_ok=True)
        count_files = 0
        for root, dirs, files in os.walk(temp_separate_dir):
            for file in files:
                if "Original_Vocals_No_Reverb" in file and file.endswith(".wav"):
                    src = os.path.join(root, file)
                    dst = os.path.join(dataset_train_ready, f"{count_files}.wav")
                    shutil.move(src, dst)
                    count_files += 1

        shutil.rmtree(temp_separate_dir, ignore_errors=True)
        os.environ["SKIP_INST_DENOISE"] = "0"

        if count_files == 0:
            yield None, log("Lỗi: Không tìm thấy file giọng tách được. Vui lòng kiểm tra lại file ghi âm.")
            return

        yield None, log("Đang kiểm tra chất lượng dữ liệu train (loại bỏ khoảng lặng)...")
        effective_duration = check_dataset_duration(dataset_train_ready)
        if effective_duration < 60:
            yield None, log(f"🛑 LỖI: Tổng thời lượng giọng thực tế (đã trừ khoảng lặng) là {effective_duration:.2f}s (< 60s tối thiểu).\n"
                            f"Vui lòng gửi thêm file ghi âm hoặc dùng file dài hơn.")
            return

        yield None, log(f"✅ Dữ liệu hợp lệ: {count_files} files, thời lượng thực tế: {effective_duration:.2f}s")

        # ================= BƯỚC 2: HUẤN LUYỆN MÔ HÌNH =================
        yield None, log(f"== BẮT ĐẦU HUẤN LUYỆN ({epochs} epochs) ==")

        yield None, log("Đang tiền xử lý...")
        for output in preprocess(
            model_name=model_name,
            dataset=dataset_train_ready,
            sample_rate="48k",
            cpu_core=os.cpu_count(),
            cut_preprocess="Automatic",
            process_effects=False,
            clean_dataset=False,
            clean_strength=0.7,
            chunk_len=3.0, overlap_len=0.3, normalization_mode="none"
        ):
            yield None, log(output)

        yield None, log("Đang trích xuất đặc trưng (f0)...")
        for output in extract(
            model_name=model_name,
            version="v2",
            method="rmvpe",
            pitch_guidance=True,
            hop_length=160,
            cpu_cores=os.cpu_count(),
            gpu=0,
            sample_rate="48k",
            embedders="hubert_base",
            custom_embedders="",
            onnx_f0_mode=False,
            embedders_mode="fairseq",
            f0_autotune=False, f0_autotune_strength=1.0,
            hybrid_method="rmvpe", rms_extract=False, alpha=0.5
        ):
            yield None, log(output)

        yield None, log("Đang tạo chỉ mục (index)...")
        for output in create_index(model_name, "v2", "Auto"):
            yield None, log(output)

        yield None, log("Đang huấn luyện (Training)... Việc này có thể mất thời gian.")
        for output in training(
            model_name=model_name,
            rvc_version="v2",
            save_every_epoch=50,
            save_only_latest=True,
            save_every_weights=True,
            total_epoch=epochs,
            sample_rate="48k",
            batch_size=8,
            gpu=0,
            pitch_guidance=True,
            not_pretrain=False,
            custom_pretrained=False,
            pretrain_g="",
            pretrain_d="",
            detector=False,
            threshold=50,
            clean_up=False,
            cache=True,
            model_author="",
            vocoder="Default",
            checkpointing=False,
            deterministic=False,
            benchmark=False,
            optimizer="AdamW",
            energy_use=False,
            custom_reference=False,
            reference_name="",
            multiscale_mel_loss=False
        ):
            yield None, log(output)

        model_pth = _pick_latest_model_file(model_name)
        if not model_pth:
            yield None, log("Lỗi: Train xong nhưng không tìm thấy file model .pth trong weights.")
            return
        index_file = _pick_index_file(model_name)
        yield os.path.join(weights_dir, model_pth), log(
            f"== TRAIN HOÀN TẤT! Model: {model_pth} | Index: {index_file or '(không có)'} ==")

    except Exception:
        import traceback
        yield None, log(f"LỖI KHÔNG MONG MUỐN:\n{traceback.format_exc()}")


def convert_workflow(target_song, model_name, pitch_shift):
    """API 2: Đổi giọng bài hát bằng model đã train sẵn của khách hàng."""
    if not target_song:
        yield None, "Lỗi: Chưa chọn bài hát để đổi giọng."
        return

    logs = []
    def log(msg):
        logs.append(msg)
        return "\n".join(logs)

    try:
        audios_root = configs.get("audios_path", "audios")

        model_pth = _pick_latest_model_file(model_name)
        if not model_pth:
            yield None, log(f"Lỗi: Chưa có model '{model_name}'. Hãy gọi API train trước.")
            return
        yield None, log(f"Đang dùng model: {model_pth}")

        index_file = _pick_index_file(model_name)
        if index_file:
            yield None, log(f"Đang dùng index: {index_file}")
        else:
            yield None, log("Cảnh báo: Không tìm thấy file index, sẽ chạy không dùng index (chất lượng có thể kém hơn).")

        # ================= BƯỚC 1: TÁCH BÀI HÁT ĐÍCH =================
        target_path = target_song if isinstance(target_song, str) else target_song.name
        target_filename = os.path.splitext(os.path.basename(target_path))[0]
        output_target_dir = os.path.join(audios_root, target_filename)

        yield None, log(f"Đang tách nhạc bài: {target_filename}...")

        audios_stub = os.path.join(audios_root, "stub")
        os.makedirs(audios_stub, exist_ok=True)

        separate_music(
            drop_audio_files=target_path,
            input_path="",
            output_dirs=os.path.join(audios_stub, "stub"),
            export_format="mp3",
            model_name="HP-Vocal-1",
            karaoke_model="", reverb_model="MDX-Reverb", denoise_model="Lite",
            sample_rate=44100, shifts=2, batch_size=1, overlap=0.25, aggression=10,
            hop_length=1024, window_size=512, segments_size=256, post_process_threshold=0.2,
            enable_tta=False, enable_denoise=True, high_end_process=False, enable_post_process=False,
            separate_backing=False, separate_reverb=True
        )

        vocal_file = os.path.join(output_target_dir, "Original_Vocals_No_Reverb.mp3")
        if not os.path.exists(vocal_file):
            vocal_file = os.path.join(output_target_dir, "Original_Vocals.mp3")

        instrument_file = os.path.join(output_target_dir, "Instruments.mp3")

        if not os.path.exists(vocal_file) or not os.path.exists(instrument_file):
            yield None, log("Lỗi: Không tìm thấy file tách nhạc (Vocal/Instrument).")
            return

        yield None, log("Tách nhạc thành công.")

        # ================= BƯỚC 2: ĐỔI GIỌNG VÀ GHÉP NHẠC =================
        yield None, log(f"== BẮT ĐẦU ĐỔI GIỌNG VÀ GHÉP NHẠC (Model: {model_name}) ==")

        final_output_path = os.path.join(audios_root, f"{target_filename}_COVER_{model_name}.mp3")
        _ensure_dir(os.path.dirname(final_output_path))

        result_paths = convert_audio(
            clean=True, clean_strength=0.5,
            autotune=False,
            use_audio=False,
            use_original=False, convert_backing=False, not_merge_backing=False, merge_instrument=False,
            pitch=pitch_shift,
            model=model_pth,
            index=index_file, index_rate=0.75,
            input=vocal_file,
            output=final_output_path,
            format="mp3",
            method="rmvpe", hybrid_method="rmvpe", hop_length=160,
            embedders="hubert_base", custom_embedders="",
            resample_sr=0, filter_radius=3, rms_mix_rate=0.25, protect=0.33,
            split_audio=False, f0_autotune_strength=1.0, input_audio_name="",
            checkpointing=False, onnx_f0_mode=False,
            formant_shifting=False, formant_qfrency=1.0, formant_timbre=1.0,
            f0_file=None, embedders_mode="fairseq",
            proposal_pitch=False, proposal_pitch_threshold=255.0,
            audio_processing=False, alpha=0.5,
            mix_beat=True, beat_file=instrument_file,
            mix_auto_gain=True,
            add_echo=True, echo_wet=0.25, echo_delay_ms=125
        )

        if not result_paths or len(result_paths) < 7:
            yield None, log("Lỗi: convert_audio không trả về kết quả hợp lệ (có thể do thiếu model/đầu vào/đầu ra).")
            return

        final_mix = result_paths[6]

        if final_mix and os.path.exists(final_mix):
            yield final_mix, log(f"== HOÀN TẤT! FILE KẾT QUẢ: {final_mix} ==")
        else:
            yield None, log(
                "Lỗi: Không tạo được file kết quả cuối cùng.\n"
                f"- model_pth: {model_pth}\n"
                f"- index: {index_file or '(none)'}\n"
                f"- vocal_file: {vocal_file} (exists={os.path.exists(vocal_file)})\n"
                f"- instrument_file: {instrument_file} (exists={os.path.exists(instrument_file)})\n"
                f"- output: {final_output_path}"
            )

    except Exception:
        import traceback
        yield None, log(f"LỖI KHÔNG MONG MUỐN:\n{traceback.format_exc()}")


def automation_tab():
    with gr.Row():
        gr.Markdown("""
        # 🤖 QUY TRÌNH TỰ ĐỘNG HÓA (AUTO PIPELINE)
        Chức năng này sẽ tự động chạy toàn bộ quy trình: 
        1. Tách giọng mẫu -> 2. Tách nhạc đích (Ca sĩ) -> 3. Train Model -> 4. Đổi giọng & Ghép nhạc.
        """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Dữ Liệu Huấn Luyện (Giọng Mẫu)")
            training_files = gr.Files(label="Chọn các file giọng hát mẫu (WAV/MP3...)", file_types=["audio"])
            model_name = gr.Textbox(label="Tên Mô Hình (Tự động tạo)", value=get_next_cos_name(), interactive=True)
            epochs = gr.Slider(label="Số vòng huấn luyện (Epochs)", minimum=10, maximum=1000, value=150, step=10)
            force_retrain = gr.Checkbox(label="Huấn luyện lại từ đầu (Bỏ qua model cũ)", value=False, info="Tích vào đây nếu bạn muốn train lại model này thay vì dùng lại.")
        
        with gr.Column():
            gr.Markdown("### 2. Bài Hát Cần Đổi Giọng")
            target_song = gr.File(label="Chọn bài hát của ca sĩ (WAV/MP3...)", file_types=["audio"])
            pitch_shift = gr.Slider(label="Chỉnh cao độ (Pitch)", minimum=-12, maximum=12, value=0, step=1, info="Nam -> Nữ: +12, Nữ -> Nam: -12")
            btn_run = gr.Button("🚀 CHẠY TẤT CẢ (AUTO RUN)", variant="primary", scale=2)
    
    with gr.Row():
        logs = gr.Textbox(label="Nhật ký xử lý (Logs)", lines=15, interactive=False)
        output_audio = gr.Audio(label="KẾT QUẢ CUỐI CÙNG", interactive=False)

    btn_run.click(
        fn=automation_workflow,
        inputs=[training_files, target_song, model_name, epochs, pitch_shift, force_retrain],
        outputs=[output_audio, logs],
        api_name="run_automation"
    )
