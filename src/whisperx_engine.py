"""
WhisperX Engine - Chuyển audio thành JSON với word-level timing

Format JSON output:
[
  {
    "line_id": 1,
    "text_content": "some text",
    "words_timing": [
      {"index": 0, "end_time": "MM:SS.mm"},
      ...
    ]
  },
  ...
]
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class WhisperXEngine:
    """Wrapper cho WhisperX để lấy transcript + word-level timestamps."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.whisper_cfg = config.get("whisperx", {}) or {}

        self.model_name = self.whisper_cfg.get("model_name", "medium")
        self.language = self.whisper_cfg.get("language", "auto")
        self.device = self.whisper_cfg.get("device", "auto")
        self.diarization = bool(self.whisper_cfg.get("diarization", False))
        self.batch_size = int(self.whisper_cfg.get("batch_size", 16))

        try:
            import torch  # noqa: F401
        except ImportError:
            logger.warning("torch chưa được cài đặt, sẽ sử dụng CPU cho WhisperX")
            self.device = "cpu"
        else:
            import torch

            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            f"Khởi tạo WhisperXEngine: model={self.model_name}, "
            f"language={self.language}, device={self.device}, "
            f"diarization={self.diarization}"
        )

        self._model = None
        self._align_model = None
        self._align_metadata = None

    def _ensure_models_loaded(self, audio_language: str = None):
        """Load WhisperX model và align model."""
        if self._model is not None and self._align_model is not None:
            return

        try:
            import whisperx  # type: ignore
        except ImportError as e:
            raise ImportError(
                "WhisperX chưa được cài đặt. Cài bằng:\n"
                "  pip install whisperx\n"
                "Lưu ý: Có thể cần cài thêm torch phù hợp với GPU/CPU."
            ) from e

        lang_arg = None if self.language in (None, "", "auto") else self.language

        # Chọn compute_type phù hợp: GPU dùng float16, CPU dùng int8 để tránh lỗi float16
        compute_type = "float16"
        if self.device == "cpu":
            compute_type = "int8"
        # Cho phép override qua config
        compute_type = self.whisper_cfg.get("compute_type", compute_type)

        logger.info(f"Đang load WhisperX model... (compute_type={compute_type})")
        self._model = whisperx.load_model(
            self.model_name,
            self.device,
            language=lang_arg,
            compute_type=compute_type,
            vad_on=False,
        )

        if audio_language is None and hasattr(self._model, "language"):
            audio_language = getattr(self._model, "language", None)

        logger.info("Đang load WhisperX align model...")
        if audio_language is None or audio_language == "auto":
            lang_code = "en" if self.language in (None, "", "auto") else self.language
        else:
            lang_code = audio_language

        self._align_model, self._align_metadata = whisperx.load_align_model(
            language_code=lang_code,
            device=self.device,
        )

    def transcribe_to_json(self, audio_path: str, output_json_path: str) -> str:
        """
        Chạy WhisperX trên file audio và lưu JSON theo format pipeline.
        """
        import json

        try:
            import whisperx  # type: ignore
        except ImportError as e:
            raise ImportError(
                "WhisperX chưa được cài đặt. Cài bằng:\n"
                "  pip install whisperx\n"
            ) from e

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"File audio không tồn tại: {audio_path}")

        logger.info(f"Đang load audio: {audio_path}")
        audio = whisperx.load_audio(str(audio_path))

        self._ensure_models_loaded()

        logger.info("Đang chạy WhisperX transcribe...")
        result = self._model.transcribe(
            audio,
            batch_size=self.batch_size,
        )

        detected_language = result.get("language", None)
        logger.info(f"WhisperX detected language: {detected_language}")

        self._ensure_models_loaded(audio_language=detected_language)

        logger.info("Đang align để lấy word-level timestamps...")
        aligned_result = whisperx.align(
            result["segments"],
            self._align_model,
            self._align_metadata,
            audio,
            self.device,
        )

        segments = aligned_result.get("segments", [])
        logger.info(f"Đã nhận {len(segments)} segments từ WhisperX")

        raw_data: List[Dict[str, Any]] = []
        line_id = 1

        for seg in segments:
            text = (seg.get("text") or "").strip()
            words = seg.get("words") or []

            if not text:
                continue

            if not words:
                seg_start = float(seg.get("start", 0.0))
                seg_end = float(seg.get("end", seg_start))
                duration = max(0.0, seg_end - seg_start)
                tokens = text.split()
                if not tokens:
                    continue

                word_duration = duration / len(tokens) if len(tokens) > 0 else duration
                words_timing = []
                for idx in range(len(tokens)):
                    word_end_time = seg_start + (idx + 1) * word_duration
                    words_timing.append(
                        {"index": idx, "end_time": self._format_timestamp(word_end_time)}
                    )
            else:
                tokens = []
                words_timing = []
                for idx, w in enumerate(words):
                    w_text = (w.get("word") or "").strip()
                    if not w_text:
                        continue
                    tokens.append(w_text)
                    end_time = float(w.get("end", w.get("start", 0.0)))
                    words_timing.append(
                        {
                            "index": len(tokens) - 1,
                            "end_time": self._format_timestamp(end_time),
                        }
                    )

                if not tokens:
                    continue

                text = " ".join(tokens)

            raw_data.append(
                {
                    "line_id": line_id,
                    "text_content": text,
                    "words_timing": words_timing,
                }
            )
            line_id += 1

        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"WhisperXEngine: Đã tạo {len(raw_data)} entries trong {output_path}"
        )

        return str(output_path)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format số giây thành 'MM:SS.mm'."""
        if seconds < 0:
            seconds = 0.0
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:05.2f}"

"""
WhisperX Engine - Chuyển audio thành JSON với word-level timing

Format JSON output tương thích với pipeline hiện tại:
[
  {
    "line_id": 1,
    "text_content": "some text",
    "words_timing": [
      {"index": 0, "end_time": "MM:SS.mm"},
      ...
    ]
  },
  ...
]
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class WhisperXEngine:
    """Wrapper cho WhisperX để lấy transcript + word-level timestamps."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.whisper_cfg = config.get("whisperx", {}) or {}

        # Model / device / language
        self.model_name = self.whisper_cfg.get("model_name", "medium")
        self.language = self.whisper_cfg.get("language", "auto")  # "auto" hoặc mã ngôn ngữ (vi, en, ...)
        self.device = self.whisper_cfg.get("device", "auto")  # "auto", "cuda", "cpu"
        self.diarization = bool(self.whisper_cfg.get("diarization", False))
        self.batch_size = int(self.whisper_cfg.get("batch_size", 16))

        # Quyết định device thực tế
        try:
            import torch  # noqa: F401
        except ImportError:
            logger.warning("torch chưa được cài đặt, sẽ sử dụng CPU cho WhisperX")
            self.device = "cpu"
        else:
            import torch

            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            f"Khởi tạo WhisperXEngine: model={self.model_name}, "
            f"language={self.language}, device={self.device}, "
            f"diarization={self.diarization}"
        )

        # Lazy-load models khi cần
        self._model = None
        self._align_model = None
        self._align_metadata = None

    def _ensure_models_loaded(self, audio_language: str = None):
        """Load WhisperX model và align model."""
        if self._model is not None and self._align_model is not None:
            return

        try:
            import whisperx  # type: ignore
        except ImportError as e:
            raise ImportError(
                "WhisperX chưa được cài đặt. Cài bằng:\n"
                "  pip install whisperx\n"
                "Lưu ý: Có thể cần cài thêm torch phù hợp với GPU/CPU."
            ) from e

        # Ngôn ngữ cho model: None = auto detect
        lang_arg = None if self.language in (None, "", "auto") else self.language

        logger.info("Đang load WhisperX model...")
        self._model = whisperx.load_model(
            self.model_name,
            self.device,
            language=lang_arg,
        )

        # Nếu không chỉ định language, sẽ detect sau khi transcribe
        if audio_language is None and hasattr(self._model, "language"):
            audio_language = getattr(self._model, "language", None)

        # Align model dùng để có word-level timestamps chính xác
        logger.info("Đang load WhisperX align model...")
        if audio_language is None or audio_language == "auto":
            # Nếu vẫn chưa biết ngôn ngữ, dùng language cấu hình (nếu không phải auto)
            lang_code = "en" if self.language in (None, "", "auto") else self.language
        else:
            lang_code = audio_language

        self._align_model, self._align_metadata = whisperx.load_align_model(
            language_code=lang_code,
            device=self.device,
        )

    def transcribe_to_json(self, audio_path: str, output_json_path: str) -> str:
        """
        Chạy WhisperX trên file audio và lưu JSON theo format pipeline.

        Args:
            audio_path: Đường dẫn file audio input.
            output_json_path: Đường dẫn file JSON output.

        Returns:
            Đường dẫn file JSON đã tạo (str).
        """
        from pathlib import Path
        import json

        try:
            import whisperx  # type: ignore
        except ImportError as e:
            raise ImportError(
                "WhisperX chưa được cài đặt. Cài bằng:\n"
                "  pip install whisperx\n"
                "Hoặc xem hướng dẫn chi tiết trong README."
            ) from e

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"File audio không tồn tại: {audio_path}")

        logger.info(f"Đang load audio: {audio_path}")
        audio = whisperx.load_audio(str(audio_path))

        # Load model (auto detect language nếu cần)
        self._ensure_models_loaded()

        logger.info("Đang chạy WhisperX transcribe...")
        result = self._model.transcribe(
            audio,
            batch_size=self.batch_size,
        )

        detected_language = result.get("language", None)
        logger.info(f"WhisperX detected language: {detected_language}")

        # Load align model (cần language code)
        self._ensure_models_loaded(audio_language=detected_language)

        logger.info("Đang align để lấy word-level timestamps...")
        aligned_result = whisperx.align(
            result["segments"],
            self._align_model,
            self._align_metadata,
            audio,
            self.device,
        )

        segments = aligned_result.get("segments", [])
        logger.info(f"Đã nhận {len(segments)} segments từ WhisperX")

        raw_data: List[Dict[str, Any]] = []
        line_id = 1

        for seg in segments:
            text = (seg.get("text") or "").strip()
            words = seg.get("words") or []

            if not text:
                continue

            if not words:
                # Không có word-level → chia đều thời gian theo độ dài text
                seg_start = float(seg.get("start", 0.0))
                seg_end = float(seg.get("end", seg_start))
                duration = max(0.0, seg_end - seg_start)
                tokens = text.split()
                if not tokens:
                    continue

                word_duration = duration / len(tokens) if len(tokens) > 0 else duration
                words_timing = []
                for idx in range(len(tokens)):
                    word_end_time = seg_start + (idx + 1) * word_duration
                    words_timing.append(
                        {"index": idx, "end_time": self._format_timestamp(word_end_time)}
                    )
            else:
                # Dùng word-level từ WhisperX
                tokens = []
                words_timing = []
                for idx, w in enumerate(words):
                    w_text = (w.get("word") or "").strip()
                    if not w_text:
                        continue
                    tokens.append(w_text)
                    end_time = float(w.get("end", w.get("start", 0.0)))
                    words_timing.append(
                        {
                            "index": len(tokens) - 1,
                            "end_time": self._format_timestamp(end_time),
                        }
                    )

                if not tokens:
                    continue

                # Gộp lại thành text_content để khớp với tokens
                text = " ".join(tokens)

            raw_data.append(
                {
                    "line_id": line_id,
                    "text_content": text,
                    "words_timing": words_timing,
                }
            )
            line_id += 1

        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"WhisperXEngine: Đã tạo {len(raw_data)} entries trong {output_path}"
        )

        return str(output_path)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format số giây thành 'MM:SS.mm' (phù hợp với JSONToASSConverter)."""
        if seconds < 0:
            seconds = 0.0
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:05.2f}"


