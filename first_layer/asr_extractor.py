"""
中文ASR：提取语音与时间戳
使用 faster-whisper 或 openai-whisper 进行中文语音识别
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from .evidence_bundle import ASREvidence, ASRSegment

logger = logging.getLogger(__name__)


class ASRExtractor:
    """中文语音识别器 —— 提取语音文本与时间戳

    模型下载：默认使用 HuggingFace 镜像 hf-mirror.com（国内友好）。
    也可手动下载模型到本地目录，设置 model_dir 参数。
    手动下载方式：
      git clone https://hf-mirror.com/Systran/faster-whisper-medium
    """

    # HuggingFace 国内镜像
    HF_MIRROR = "https://hf-mirror.com"

    def __init__(
        self,
        model_size: str = "medium",       # tiny / base / small / medium / large-v3
        device: str = "auto",             # cpu / cuda / auto
        compute_type: str = "auto",       # float16 / int8 / auto
        language: str = "zh",
        beam_size: int = 5,
        cpu_threads: int = 4,             # CPU 线程数（越大转写越快，0=自动）
        model_dir: str | None = None,     # 本地模型目录（跳过下载）
        use_hf_mirror: bool = True,       # 是否使用国内 HF 镜像
        **kwargs: Any,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.cpu_threads = cpu_threads
        self.model_dir = model_dir
        self.use_hf_mirror = use_hf_mirror
        self.model = None  # 延迟加载
        self._model_kwargs = kwargs

        # 设置 HF 相关环境变量（必须在模型下载前设置）
        import os
        if use_hf_mirror:
            os.environ.setdefault("HF_ENDPOINT", self.HF_MIRROR)
        # 禁用 Xet：hf-mirror 无法代理 Xet(CAS) 存储，否则模型大文件下载会 401
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def extract(self, audio_path: str | Path) -> ASREvidence:
        """从音频文件提取中文语音文本与时间戳"""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        model = self._load_model()
        evidence = ASREvidence(
            model_name=f"faster-whisper-{self.model_size}",
            language=self.language,
        )

        logger.info("ASR 语音转写中...")
        try:
            segments_raw, info = model.transcribe(
                str(audio_path),
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=True,            # 过滤静音
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                ),
            )

            # 收集 segments
            all_segments: list[ASRSegment] = []
            for seg in segments_raw:
                # avg_logprob 是对数概率（通常为负值），
                # 用 exp 转换为 [0,1] 区间内的置信度
                conf = math.exp(min(seg.avg_logprob, 0.0))
                conf = min(1.0, max(0.0, conf))
                all_segments.append(ASRSegment(
                    text=seg.text.strip(),
                    start_time=seg.start,
                    end_time=seg.end,
                    confidence=conf,
                    language=self.language,
                ))

            evidence.segments = all_segments
            evidence.full_text = " ".join(s.text for s in all_segments)

            # 计算整体置信度
            if all_segments:
                evidence.overall_confidence = sum(
                    s.confidence for s in all_segments
                ) / len(all_segments)

            logger.info(
                "ASR 完成: %d 段, 总字数 %d, 平均置信度 %.3f",
                len(all_segments),
                len(evidence.full_text),
                evidence.overall_confidence,
            )

        except Exception as e:
            logger.error("ASR 提取失败: %s", e)
            raise

        return evidence

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _load_model(self):
        """延迟加载模型（优先本地目录，其次 HF 镜像下载）"""
        if self.model is not None:
            return self.model

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "请安装 faster-whisper: pip install faster-whisper"
            )

        # 如果指定了本地模型目录，直接用
        if self.model_dir and Path(self.model_dir).exists():
            logger.info("从本地目录加载模型: %s", self.model_dir)
            self.model = WhisperModel(
                self.model_dir,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                local_files_only=True,
                **self._model_kwargs,
            )
            return self.model

        # 项目内置的小模型目录（models/faster-whisper-<size>，提速用）
        _root = Path(__file__).resolve().parent.parent
        _local = _root / "models" / f"faster-whisper-{self.model_size}"
        if _local.is_dir():
            logger.info("从项目本地目录加载模型: %s", _local)
            self.model = WhisperModel(
                str(_local),
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                local_files_only=True,
                **self._model_kwargs,
            )
            return self.model

        # 在线下载（使用 HF 镜像）
        logger.info(
            "加载 faster-whisper 模型: %s (device=%s, mirror=%s) —— 首次运行需下载模型，请耐心等待...",
            self.model_size, self.device,
            self.HF_MIRROR if self.use_hf_mirror else "huggingface.co",
        )
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            **self._model_kwargs,
        )
        return self.model


# ---------------------------------------------------------------------------
# 备选：使用 openai-whisper
# ---------------------------------------------------------------------------

class WhisperASRExtractor:
    """使用 openai/whisper 的备选方案"""

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "auto",
        language: str = "zh",
    ):
        self.model_size = model_size
        self.device = device
        self.language = language
        self.model = None

    def extract(self, audio_path: str | Path) -> ASREvidence:
        import torch
        import whisper

        if self.model is None:
            device_str = self.device
            if device_str == "auto":
                device_str = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("加载 whisper 模型: %s", self.model_size)
            self.model = whisper.load_model(self.model_size, device=device_str)

        result = self.model.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=5,
            verbose=False,
        )

        evidence = ASREvidence(
            model_name=f"whisper-{self.model_size}",
            language=self.language,
        )

        segments = []
        for seg in result.get("segments", []):
            segments.append(ASRSegment(
                text=seg["text"].strip(),
                start_time=seg["start"],
                end_time=seg["end"],
                confidence=seg.get("confidence", 0.0),
                language=self.language,
            ))

        evidence.segments = segments
        evidence.full_text = result.get("text", "")
        if segments:
            evidence.overall_confidence = sum(
                s.confidence for s in segments
            ) / len(segments)

        return evidence
