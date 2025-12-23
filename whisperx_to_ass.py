"""
WhisperX to ASS - Chuyển file audio thành phụ đề .ass với word-level timing.

Sử dụng WhisperX (faster-whisper + alignment) để tạo JSON theo format:
[
  {
    "line_id": 1,
    "text_content": "...",
    "words_timing": [
      {"index": 0, "end_time": "MM:SS.mm"},
      ...
    ]
  },
  ...
]

Sau đó tái sử dụng JSONToASSConverter để xuất file .ass karaoke từng từ.
"""

import argparse
import sys
import io
from pathlib import Path

from src.utils import load_config, setup_logging
from src.whisperx_engine import WhisperXEngine
from src.json_to_ass import JSONToASSConverter


def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Fallback cho console Windows không hỗ trợ Unicode đầy đủ
        safe_args = []
        for arg in args:
            if isinstance(arg, str):
                safe_args.append(arg.encode("ascii", "replace").decode("ascii"))
            else:
                safe_args.append(str(arg).encode("ascii", "replace").decode("ascii"))
        print(*safe_args, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="WhisperX -> ASS (Audio -> word-level karaoke subtitle)"
    )
    parser.add_argument(
        "--audio",
        "-i",
        type=str,
        required=True,
        help="Đường dẫn file audio input (wav/mp3/m4a/...)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Đường dẫn file .ass output (mặc định: cùng tên với audio, trong thư mục output/)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Đường dẫn file config KaraSub (default: config.yaml)",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="auto",
        help="Ngôn ngữ cho WhisperX (auto, vi, en, ...). auto = tự phát hiện",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Thiết bị chạy WhisperX: auto / cuda / cpu (default: auto)",
    )
    parser.add_argument(
        "--diarization",
        action="store_true",
        help="Bật diarization (nhận diện người nói) - chủ yếu cho podcast/talkshow",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Hiển thị log chi tiết",
    )

    args = parser.parse_args()

    # Fix encoding cho Windows
    if sys.platform == "win32":
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
            else:
                sys.stdout = io.TextIOWrapper(
                    sys.stdout.buffer, encoding="utf-8", errors="replace"
                )
                sys.stderr = io.TextIOWrapper(
                    sys.stderr.buffer, encoding="utf-8", errors="replace"
                )
        except Exception:
            pass

    # Setup logging
    setup_logging(verbose=args.verbose)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        safe_print(f"ERROR: File audio không tồn tại: {audio_path}")
        sys.exit(1)

    # Output path
    if args.output:
        output_ass = Path(args.output)
    else:
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_ass = output_dir / f"{audio_path.stem}_whisperx.ass"

    output_ass.parent.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        safe_print(f"ERROR: File config không tồn tại: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))

    # Thiết lập cấu hình WhisperX trong config
    whisper_cfg = config.get("whisperx", {}) or {}
    whisper_cfg["model_name"] = whisper_cfg.get("model_name", "medium")
    whisper_cfg["language"] = args.language or whisper_cfg.get("language", "auto")
    whisper_cfg["device"] = args.device or whisper_cfg.get("device", "auto")
    whisper_cfg["diarization"] = bool(args.diarization)
    whisper_cfg["batch_size"] = whisper_cfg.get("batch_size", 16)
    config["whisperx"] = whisper_cfg

    safe_print("========================================")
    safe_print("WhisperX Audio -> ASS (word-level)")
    safe_print("========================================")
    safe_print(f"Audio   : {audio_path}")
    safe_print(f"Output  : {output_ass}")
    safe_print(f"Config  : {config_path}")
    safe_print(f"Lang    : {whisper_cfg['language']}")
    safe_print(f"Device  : {whisper_cfg['device']}")
    safe_print(f"Diarize : {whisper_cfg['diarization']}")
    safe_print("----------------------------------------")

    try:
        engine = WhisperXEngine(config)
        converter = JSONToASSConverter(config)

        # Bước 1: WhisperX -> JSON
        raw_json = output_ass.parent / f"{output_ass.stem}_raw.json"
        safe_print("Bước 1: Đang chạy WhisperX (audio -> JSON word-level)...")
        raw_json_path = engine.transcribe_to_json(str(audio_path), str(raw_json))
        safe_print(f"[OK] Đã tạo JSON: {raw_json_path}")

        # Bước 2: JSON -> ASS
        safe_print("Bước 2: Đang convert JSON sang ASS karaoke...")
        converter.convert(str(raw_json_path), str(output_ass))
        safe_print(f"[OK] Đã tạo file ASS: {output_ass}")

        safe_print("========================================")
        safe_print("Hoàn thành!")
        safe_print("========================================")

    except Exception as e:
        safe_print(f"ERROR: Có lỗi xảy ra: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


