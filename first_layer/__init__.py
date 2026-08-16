"""
第一层：专业模型提取事实
- 中文ASR：提取语音与时间戳
- OCR模型：提取画面字幕
- 视觉理解模型：描述关键帧内容
- 音频识别模型：歌曲与版本候选
- 统一证据包 EvidenceBundle
"""

import os

# ---------------------------------------------------------------------------
# 必须在导入 torch / numpy / ctranslate2 等库之前设置（否则多个库各自捆绑的
# libiomp5md.dll 会重复初始化，触发 "OMP: Error #15" 并卡死）。
# 放在包 __init__ 里保证所有入口（run_first_layer / run_pipeline 等）都生效。
# ---------------------------------------------------------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# hf-mirror 无法代理 HuggingFace 的 Xet(CAS) 存储，禁用 Xet 走普通 HTTP 下载
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# 所有模型统一放项目内 models/ 目录（百度同步盘内），HuggingFace 缓存指向项目
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault(
    "HF_HOME", os.path.join(_PROJECT_ROOT, "models", "hf_cache")
)

from .evidence_bundle import (
    EvidenceBundle,
    ASREvidence,
    OCREvidence,
    KeyFrameDescription,
    AudioRecognitionResult,
    EvidenceSource,
    ContentCategory,
)
from .pipeline import FirstLayerPipeline

__all__ = [
    "EvidenceBundle",
    "ASREvidence",
    "OCREvidence",
    "KeyFrameDescription",
    "AudioRecognitionResult",
    "EvidenceSource",
    "ContentCategory",
    "FirstLayerPipeline",
]
