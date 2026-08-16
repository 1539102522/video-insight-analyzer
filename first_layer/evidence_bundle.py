"""
统一证据包 EvidenceBundle
将所有第一层模型的输出统一为结构化证据，附带时间戳、信任度、来源。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# 枚举定义
# ---------------------------------------------------------------------------

class EvidenceSource(str, Enum):
    """证据来源"""
    ASR = "asr"
    OCR = "ocr"
    VISUAL = "visual"
    AUDIO = "audio"


class ContentCategory(str, Enum):
    """内容分类"""
    MUSIC = "music"
    RECIPE = "recipe"
    PROSE = "prose"
    MIXED = "mixed"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# 各模型证据
# ---------------------------------------------------------------------------

class ASRSegment(BaseModel):
    """ASR 单段语音识别结果"""
    text: str = Field(..., description="识别文本")
    start_time: float = Field(..., ge=0, description="起始时间（秒）")
    end_time: float = Field(..., ge=0, description="结束时间（秒）")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="该段置信度")
    language: str = Field(default="zh", description="检测语言")

    @field_validator("end_time")
    @classmethod
    def validate_times(cls, v: float, info: Any) -> float:
        start = info.data.get("start_time")
        if start is not None and v < start:
            raise ValueError("end_time must be >= start_time")
        return v


class ASREvidence(BaseModel):
    """ASR 证据"""
    source: EvidenceSource = Field(default=EvidenceSource.ASR, frozen=True)
    model_name: str = Field(default="", description="使用的ASR模型名称")
    language: str = Field(default="zh", description="主要语言")
    segments: list[ASRSegment] = Field(default_factory=list)
    full_text: str = Field(default="", description="完整转录文本")
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class OCRSegment(BaseModel):
    """OCR 单帧识别结果"""
    text: str = Field(..., description="识别到的文字")
    bbox: list[float] = Field(default_factory=list, description="边界框 [x1,y1,x2,y2]")
    frame_index: int = Field(..., ge=0, description="关键帧序号")
    timestamp: float = Field(..., ge=0, description="帧对应视频时间（秒）")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class OCREvidence(BaseModel):
    """OCR 证据"""
    source: EvidenceSource = Field(default=EvidenceSource.OCR, frozen=True)
    model_name: str = Field(default="", description="使用的OCR模型名称")
    segments: list[OCRSegment] = Field(default_factory=list)
    full_text: str = Field(default="", description="合并后的全部OCR文字")
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class KeyFrameDescription(BaseModel):
    """关键帧视觉理解"""
    frame_index: int = Field(..., ge=0)
    timestamp: float = Field(..., ge=0)
    description: str = Field(..., description="画面内容自然语言描述")
    objects: list[str] = Field(default_factory=list, description="检测到的物体")
    scene_type: str = Field(default="", description="场景类型")
    text_in_frame: str = Field(default="", description="画面中可见文字")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class VisualEvidence(BaseModel):
    """视觉理解证据"""
    source: EvidenceSource = Field(default=EvidenceSource.VISUAL, frozen=True)
    model_name: str = Field(default="")
    keyframes: list[KeyFrameDescription] = Field(default_factory=list)
    overall_summary: str = Field(default="", description="视频整体内容概要")


class SongCandidate(BaseModel):
    """歌曲候选"""
    title: str = Field(..., description="歌曲名")
    artist: str = Field(default="", description="歌手/艺术家")
    version: str = Field(default="", description="版本信息（原版/翻唱/remix等）")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    match_segment: tuple[float, float] = Field(
        default=(0.0, 0.0), description="匹配到的音频区间 (start, end)"
    )
    fingerprint_id: str = Field(default="", description="音频指纹ID")


class AudioRecognitionResult(BaseModel):
    """音频识别证据"""
    source: EvidenceSource = Field(default=EvidenceSource.AUDIO, frozen=True)
    model_name: str = Field(default="")
    has_music: bool = Field(default=False, description="是否检测到音乐")
    music_segments: list[tuple[float, float]] = Field(
        default_factory=list, description="检测到的音乐片段 (start, end)"
    )
    songs: list[SongCandidate] = Field(default_factory=list)
    background_music: list[SongCandidate] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    audio_duration: float = Field(default=0.0, ge=0)


# ---------------------------------------------------------------------------
# 统一证据包
# ---------------------------------------------------------------------------

class EvidenceBundle(BaseModel):
    """统一证据包 —— 第一层所有模型的输出汇总"""

    # 允许自定义提取器写入注册表字段之外的证据（可扩展）
    model_config = ConfigDict(extra="allow")

    # 元信息
    bundle_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:12],
        description="证据包唯一ID",
    )
    video_path: str = Field(..., description="原始视频路径")
    video_duration: float = Field(default=0.0, ge=0, description="视频时长（秒）")
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="创建时间",
    )

    # 内容分类（初步猜测，第二层会精确判断）
    category_hint: ContentCategory = Field(default=ContentCategory.UNKNOWN)

    # 四类证据
    asr: ASREvidence | None = Field(default=None, description="ASR语音证据")
    ocr: OCREvidence | None = Field(default=None, description="OCR字幕证据")
    visual: VisualEvidence | None = Field(default=None, description="视觉理解证据")
    audio: AudioRecognitionResult | None = Field(default=None, description="音频识别证据")

    # 错误信息
    errors: list[dict[str, Any]] = Field(default_factory=list, description="各模块错误记录")

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    def all_text(self) -> str:
        """合并所有文本证据（ASR + OCR + 视觉中的文字）"""
        parts: list[str] = []
        if self.asr:
            parts.append(self.asr.full_text)
        if self.ocr:
            parts.append(self.ocr.full_text)
        if self.visual:
            for kf in self.visual.keyframes:
                if kf.text_in_frame:
                    parts.append(kf.text_in_frame)
        return "\n".join(parts)

    def is_complete(self) -> bool:
        """判断是否至少有一个模型成功返回了证据"""
        return any([self.asr, self.ocr, self.visual, self.audio])

    def summary(self) -> dict[str, Any]:
        """生成摘要"""
        return {
            "bundle_id": self.bundle_id,
            "video_path": self.video_path,
            "duration": self.video_duration,
            "has_asr": self.asr is not None,
            "has_ocr": self.ocr is not None,
            "has_visual": self.visual is not None,
            "has_audio": self.audio is not None,
            "errors_count": len(self.errors),
            "category_hint": self.category_hint.value,
        }
