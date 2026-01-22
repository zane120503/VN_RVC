import os
import sys
import shutil
import gradio as gr
from main.app.variables import translations, configs, index_path
from main.app.core.ui import gr_info, gr_warning, gr_error
from main.app.core.separate import separate_music
from main.app.core.training import preprocess, extract, create_index, training
from main.app.core.inference import convert_audio
from main.app.tabs.training.child.training import get_next_cos_name

def automation_workflow(
    training_files, 
    target_song, 
    model_name, 
    epochs,
    pitch_shift
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
        # =================================================================================
        # BƯỚC 1: TÁCH GIỌNG TRAIN (DATASET)
        # =================================================================================
        yield None, log(f"== BẮT ĐẦU BƯỚC 1: TÁCH DATASET CHO MODEL {model_name} ==")
        
        # Tạo thư mục dataset tạm thời
        dataset_dir = os.path.join("dataset", model_name)
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
        os.makedirs(dataset_dir, exist_ok=True)

        # Di chuyển file upload vào thư mục tạm để xử lý (nếu cần) hoặc dùng trực tiếp
        # separate_music expects a list of file paths or a directory
        # Vì separate_music output ra structure riêng, ta sẽ dùng output_dirs là dataset_dir
        # Tuy nhiên separate_music tạo subfolder cho mỗi bài hát. 
        # Để đơn giản cho training, ta cần gom tất cả 'Vocals' vào 1 folder dataset model.
        
        # Tách từng file một và gom vocal
        dataset_train_ready = os.path.join("dataset", model_name) # Đây là folder chứa wav 48k/32k sạch
        # Nhưng separate_music output ra subfolder.
        # Ta sẽ tách vào temp_separate trước
        
        temp_separate_dir = os.path.join("audios", f"temp_train_{model_name}")
        stub_dir = os.path.join(temp_separate_dir, "stub")
        os.makedirs(stub_dir, exist_ok=True)
        
        # Gọi tách nhạc
        file_paths = [f.name for f in training_files]
        
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

        yield None, log(f"Đã chuẩn bị xong dữ liệu train: {count_files} files.")

        # =================================================================================
        # BƯỚC 2: HUẤN LUYỆN MÔ HÌNH
        # =================================================================================
        # =================================================================================
        # BƯỚC 2: XỬ LÝ BÀI HÁT ĐÍCH (Được đưa lên trước Training)
        # =================================================================================
        yield None, log(f"== BẮT ĐẦU BƯỚC 2: XỬ LÝ BÀI HÁT ĐÍCH ==")
        
        target_path = target_song.name
        target_filename = os.path.splitext(os.path.basename(target_path))[0]
        output_target_dir = os.path.join("audios", target_filename)
        
        yield None, log(f"Đang tách nhạc bài: {target_filename}...")
        
        # Tạo stub cho audios để workaround lỗi cắt path
        audios_stub = os.path.join("audios", "stub")
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
        yield None, log(f"== BẮT ĐẦU BƯỚC 3: HUẤN LUYỆN ({epochs} epochs) ==")
        
        # Preprocess
        yield None, log("Đang tiền xử lý...")
        for output in preprocess(
            model_name=model_name,
            dataset=dataset_train_ready,
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

        # =================================================================================
        # BƯỚC 4: CHUYỂN ĐỔI VÀ GHÉP (INFERENCE)
        # =================================================================================
        yield None, log(f"== BẮT ĐẦU BƯỚC 4: ĐỔI GIỌNG VÀ GHÉP NHẠC ==")
        
        # Tìm file model .pth và .index
        model_pth = f"{model_name}.pth" # Usually resides in assets/weights
        
        # Helper tìm model file nếu tên bị đổi (ví dụ thêm số steps)
        weights_dir = os.path.join("assets", "weights")
        if not os.path.exists(os.path.join(weights_dir, model_pth)):
             chk_candidates = []
             if os.path.exists(weights_dir):
                 for f in os.listdir(weights_dir):
                     if f.startswith(model_name) and f.endswith(".pth"):
                         chk_candidates.append(f)
             
             if chk_candidates:
                 # Sort by name length or mtime? 
                 # Usually _latest suffix or _100e. sort by mtime is safer for "latest" run
                 chk_candidates.sort(key=lambda x: os.path.getmtime(os.path.join(weights_dir, x)), reverse=True)
                 model_pth = chk_candidates[0]
                 yield None, log(f"Đã tìm thấy checkpoint model mới nhất: {model_pth}")
             else:
                 yield None, log(f"Cảnh báo: Không tìm thấy file model khớp tên {model_pth} trong {weights_dir}")
        
        # Tìm index file (Vừa tạo ở bước 2)
        # logs/<model_name>/added_...index
        index_file = ""
        logs_model_dir = os.path.join("assets", "logs", model_name)
        if os.path.exists(logs_model_dir):
            for f in os.listdir(logs_model_dir):
                if f.endswith(".index") and "added" in f:
                    index_file = os.path.join(logs_model_dir, f)
                    break
        
        # Output path
        final_output_path = os.path.join("audios", f"{target_filename}_COVER_{model_name}.mp3")
        
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
        
        # convert_audio returns list. mix output is usually at index 6 based on convert.py
        # [vocal, backing, merge_back, original, merge_inst, update, mix_result]
        final_mix = result_paths[6]
        
        if final_mix and os.path.exists(final_mix):
             yield final_mix, log(f"== HOÀN TẤT! FILE KẾT QUẢ: {final_mix} ==")
        else:
             yield None, log("Lỗi: Không tạo được file kết quả cuối cùng.")

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        yield None, log(f"LỖI KHÔNG MONG MUỐN:\n{err}")


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
        inputs=[training_files, target_song, model_name, epochs, pitch_shift],
        outputs=[output_audio, logs]
    )
