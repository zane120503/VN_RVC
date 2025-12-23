import os
import sys

sys.path.append(os.getcwd())

from main.app.core.inference import whisper_process
from main.library.utils import check_spk_diarization
from main.app.core.ui import gr_info, gr_warning, process_output
from main.app.variables import config, translations, configs, logger

def create_ass(model_size, input_audio, output_file, word_timestamps, style_name="Dòng trên"):
    """
    Tạo file ASS từ audio với word timestamps
    Format giống file mẫu: {\K500} {\K52}word1 {\K59}word2 ...
    """
    import multiprocessing as mp

    if not input_audio or not os.path.exists(input_audio) or os.path.isdir(input_audio): 
        gr_warning(translations["input_not_valid"])
        return [None]*2
    
    if not output_file.endswith(".ass"): output_file += ".ass"
        
    if not output_file:
        gr_warning(translations["output_not_valid"])
        return [None]*2
    
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

    info = ""
    output_file = process_output(output_file)

    # Kiểm tra và tải mô hình Whisper (nếu chưa có)
    try:
        check_spk_diarization(model_size, speechbrain=False)
        gr_info("Đang tạo file ASS...")
    except Exception as e:
        error_msg = str(e)
        if "không đầy đủ" in error_msg.lower() or "incomplete" in error_msg.lower():
            gr_warning(f"Mô hình Whisper tải chưa đầy đủ. Vui lòng chạy lại để tiếp tục tải, hoặc tải thủ công từ HuggingFace.")
        else:
            gr_warning(f"Lỗi khi kiểm tra mô hình Whisper: {error_msg}")
        return [None]*2

    try:
        mp.set_start_method("spawn")
    except:
        pass

    whisper_queue = mp.Queue()
    whisperprocess = mp.Process(target=whisper_process, args=(model_size, input_audio, configs, config.device, whisper_queue, True))
    whisperprocess.start()

    result = whisper_queue.get()
    
    if isinstance(result, Exception):
        gr_warning(f"Lỗi khi xử lý Whisper: {result}")
        whisperprocess.join()
        return [None]*2
    
    segments = result
    whisperprocess.join()

    # Tạo file ASS
    with open(output_file, "w", encoding="utf-8") as f:
        # Write ASS header
        f.write("[Script Info]\n")
        f.write("Title: Vietnamese-RVC Generated Subtitle\n")
        f.write("ScriptType: v4.00+\n")
        f.write("\n")
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write(f"Style: {style_name},UTM Alexander Icon,75,&H000000FF,&H00FFFFFF,&H24000000,&H00000000,-1,0,0,0,100,100,0,0,1,1,0.5,1,200,0,150,1\n")
        f.write("\n")
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        f.write("\n")

        # Write dialogues với word timestamps
        for i, segment in enumerate(segments):
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"].strip()
            
            # Format timestamp: 0:00:11.32 -> 0:00:11.32
            start_str = format_ass_timestamp(start_time)
            end_str = format_ass_timestamp(end_time)
            
            # Tạo karaoke text với {\K} tags
            karaoke_text = create_karaoke_text(segment, start_time)
            
            # Write dialogue line
            dialogue = f"Dialogue: 0,{start_str},{end_str},{style_name},,0,0,0,,{karaoke_text}\n"
            f.write(dialogue)
            info += dialogue
    
    logger.info(f"Đã tạo file ASS: {output_file}")
    gr_info(translations["success"])

    return [{"value": output_file, "visible": True, "__type__": "update"}, info]

def format_ass_timestamp(seconds):
    """
    Format timestamp cho ASS: H:MM:SS.cc (giống file mẫu)
    Ví dụ: 0:00:11.32
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    # Format: H:MM:SS.cc (2 chữ số cho centiseconds)
    centiseconds = int(milliseconds / 10)
    
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

def create_karaoke_text(segment, segment_start):
    """
    Tạo text với karaoke tags {\K} từ word timestamps
    Format: {\K500} {\K52}Anh {\K59}xa {\K77}nhớ...
    Mỗi {\K} tag được đặt trước từ tương ứng, giá trị là centiseconds (1/100 giây)
    """
    words = segment.get("words", [])
    
    if not words:
        # Nếu không có word timestamps, chỉ trả về text bình thường
        return segment.get("text", "").strip()
    
    # Tính toán duration cho mỗi từ (centiseconds)
    karaoke_parts = []
    
    if len(words) > 0:
        # Khoảng delay ban đầu (từ segment start đến từ đầu tiên)
        first_word_start = words[0]["start"]
        initial_delay = max(0, (first_word_start - segment_start) * 100)
        if initial_delay > 0:
            karaoke_parts.append(f"{{\\K{int(initial_delay)}}}")
        
        # Xử lý từng từ
        for i, word in enumerate(words):
            word_text = word.get("word", "").strip()
            if not word_text:
                continue
            
            word_start = word.get("start", 0)
            word_end = word.get("end", 0)
            
            # Tính duration của từ (centiseconds) - thời gian từ bắt đầu đến kết thúc của từ
            word_duration = max(1, (word_end - word_start) * 100)
            
            # Format: {\Kduration}word - mỗi từ có tag riêng, khoảng trắng trước tag (trừ tag đầu tiên sau delay)
            # Ví dụ: {\K500} {\K52}Anh {\K59}xa {\K77}nhớ
            # Nếu có initial_delay, thêm khoảng trắng trước tag đầu tiên
            # Nếu không có initial_delay và đây là từ đầu tiên, không cần khoảng trắng
            if i == 0 and initial_delay == 0:
                karaoke_parts.append(f"{{\\K{int(word_duration)}}}{word_text}")
            else:
                karaoke_parts.append(f" {{\\K{int(word_duration)}}}{word_text}")
        
        return "".join(karaoke_parts)
    else:
        return segment.get("text", "").strip()

