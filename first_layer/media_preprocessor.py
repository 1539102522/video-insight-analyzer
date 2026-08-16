"""
媒体预处理：FFmpeg / ffprobe 转码、抽音频、提取关键帧
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 用 shutil.which 做 PATH 查找
shutil_which = shutil.which


@dataclass
class MediaInfo:
    """视频元信息"""
    path: str
    duration: float = 0.0          # 秒
    width: int = 0
    height: int = 0
    fps: float = 0.0
    codec: str = ""
    audio_codec: str = ""
    sample_rate: int = 0
    has_audio: bool = False
    has_video: bool = False
    format_name: str = ""
    file_size: int = 0             # bytes
    bit_rate: int = 0


@dataclass
class PreprocessResult:
    """预处理产物"""
    audio_path: str = ""           # 提取的音频文件路径
    keyframe_dir: str = ""         # 关键帧目录
    converted_video: str = ""      # 转码后视频（如需要）
    media_info: MediaInfo = field(default_factory=lambda: MediaInfo(path=""))
    keyframe_count: int = 0
    keyframe_interval: float = 1.0 # 关键帧间隔（秒）


class MediaPreprocessor:
    """使用 FFmpeg/ffprobe 进行视频预处理

    自动检测 ffmpeg：
    1. 优先使用传入的路径
    2. 尝试系统 PATH 中的 ffmpeg/ffprobe
    3. 尝试 imageio-ffmpeg 自带的静态二进制
    """

    def __init__(
        self,
        ffmpeg_bin: str = "ffmpeg",
        ffprobe_bin: str = "ffprobe",
        work_dir: str | None = None,
        keyframe_interval: float = 2.0,   # 每 N 秒取一帧
    ):
        # 自动检测 ffmpeg/ffprobe 路径
        self.ffmpeg = self._resolve_ffmpeg(ffmpeg_bin)
        self.ffprobe = self._resolve_ffprobe(ffprobe_bin)
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="video_preprocess_"))
        self.keyframe_interval = keyframe_interval
        self.work_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def process(self, video_path: str | Path) -> PreprocessResult:
        """完整预处理流程：探针 → 提取音频 → 提取关键帧"""
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        media_info = self._probe(video_path)
        result = PreprocessResult(media_info=media_info)

        # 提取音频
        if media_info.has_audio:
            result.audio_path = self._extract_audio(video_path)

        # 提取关键帧
        result.keyframe_dir, result.keyframe_count = self._extract_keyframes(
            video_path, media_info.duration
        )
        result.keyframe_interval = self.keyframe_interval

        return result

    def probe(self, video_path: str | Path) -> MediaInfo:
        """仅探测媒体元信息（不抽帧不转码），用于时长/格式校验"""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        return self._probe(video_path)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _probe(self, video_path: Path) -> MediaInfo:
        """用 ffprobe 或 OpenCV 获取视频元信息"""
        # 优先 ffprobe
        if self.ffprobe:
            try:
                return self._probe_ffprobe(video_path)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "ffprobe 解析失败，回退到 OpenCV: %s", e
                )

        # 回退到 OpenCV
        return self._probe_opencv(video_path)

    def _probe_ffprobe(self, video_path: Path) -> MediaInfo:
        cmd = [
            self.ffprobe,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]
        try:
            output = subprocess.check_output(cmd, text=True, encoding="utf-8")
            data = json.loads(output)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            raise RuntimeError(f"ffprobe 解析失败: {e}")

        fmt = data.get("format", {})
        streams = data.get("streams", [])

        info = MediaInfo(path=str(video_path))
        info.duration = float(fmt.get("duration", 0))
        info.format_name = fmt.get("format_name", "")
        info.file_size = int(fmt.get("size", 0))
        info.bit_rate = int(fmt.get("bit_rate", 0))

        for s in streams:
            codec_type = s.get("codec_type", "")
            if codec_type == "video":
                info.has_video = True
                info.width = s.get("width", 0)
                info.height = s.get("height", 0)
                info.codec = s.get("codec_name", "")
                # 计算 FPS
                fps_str = s.get("r_frame_rate", "0/1")
                try:
                    num, den = fps_str.split("/")
                    info.fps = float(num) / float(den) if float(den) != 0 else 0.0
                except (ValueError, ZeroDivisionError):
                    info.fps = 0.0
            elif codec_type == "audio":
                info.has_audio = True
                info.audio_codec = s.get("codec_name", "")
                info.sample_rate = int(s.get("sample_rate", 0))

        return info

    def _probe_opencv(self, video_path: Path) -> MediaInfo:
        """用 OpenCV 获取视频元信息（ffprobe 不可用时的兜底方案）"""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV 无法打开视频: {video_path}")

        info = MediaInfo(path=str(video_path))
        info.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info.fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        info.duration = frame_count / info.fps if info.fps > 0 else 0.0
        info.has_video = info.width > 0 and info.height > 0
        info.codec = "h264"  # OpenCV 不直接暴露 codec name

        # 检查是否有音频（通过 ffmpeg 快速检测）
        if self.ffmpeg:
            try:
                probe_cmd = [
                    self.ffmpeg, "-i", str(video_path),
                    "-t", "5",                # 只分析前 5 秒，避免长视频探测超时
                    "-af", "volumedetect", "-f", "null", "-"
                ]
                result = subprocess.run(
                    probe_cmd, capture_output=True, text=True, timeout=30,
                    # ffmpeg 输出可能含非 GBK 字节，强制 utf-8 容错解码
                    encoding="utf-8", errors="replace",
                )
                info.has_audio = (
                    "Stream #0:1" in result.stderr or "Audio:" in result.stderr
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("音频探测失败: %s", e)
                info.has_audio = False

        cap.release()
        info.file_size = video_path.stat().st_size
        return info

    def _extract_audio(self, video_path: Path) -> str:
        """提取音频为 WAV 16kHz 单声道（适配大多数ASR模型）"""
        # 使用 ASCII 文件名：避免 ffmpeg/OpenCV 在 Windows 下处理中文路径出错
        out_path = str(self.work_dir / "audio.wav")
        cmd = [
            self.ffmpeg,
            "-y",
            "-i", str(video_path),
            "-vn",                    # 不要视频
            "-acodec", "pcm_s16le",   # PCM 16-bit
            "-ar", "16000",           # 16kHz 采样率
            "-ac", "1",               # 单声道
            out_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return out_path

    def _extract_keyframes(self, video_path: Path, duration: float) -> tuple[str, int]:
        """按固定间隔提取关键帧"""
        # 使用 ASCII 目录名：避免 ffmpeg/OpenCV 在 Windows 下处理中文路径出错
        out_dir = self.work_dir / "keyframes"
        out_dir.mkdir(exist_ok=True)

        out_pattern = str(out_dir / "frame_%04d.jpg")

        # 使用 fps 滤镜按间隔抽帧
        fps = 1.0 / max(self.keyframe_interval, 0.5)
        cmd = [
            self.ffmpeg,
            "-y",
            "-i", str(video_path),
            "-vf", f"fps={fps:.4f}",
            "-q:v", "2",              # 高质量 JPEG
            out_pattern,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # 统计帧数
        frames = sorted(out_dir.glob("frame_*.jpg"))
        return str(out_dir), len(frames)

    # ------------------------------------------------------------------
    # FFmpeg 路径自动检测
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_ffmpeg(default: str = "ffmpeg") -> str:
        """自动检测 ffmpeg 路径"""
        # 1. 如果传入了具体路径且存在，直接使用
        if default != "ffmpeg" and Path(default).exists():
            return default

        # 2. 尝试系统 PATH
        if shutil_which("ffmpeg"):
            return "ffmpeg"

        # 3. 尝试 imageio-ffmpeg 自带的静态二进制
        try:
            import imageio_ffmpeg
            path = imageio_ffmpeg.get_ffmpeg_exe()
            if path and Path(path).exists():
                return path
        except ImportError:
            pass

        raise RuntimeError(
            "找不到 ffmpeg！请执行以下任一操作：\n"
            "  1. conda install -y ffmpeg -c conda-forge\n"
            "  2. pip install imageio-ffmpeg\n"
            "  3. 从 https://ffmpeg.org/download.html 下载并加入 PATH"
        )

    @staticmethod
    def _resolve_ffprobe(default: str = "ffprobe") -> str:
        """自动检测 ffprobe 路径，找不到则返回空（后续用 OpenCV 兜底）"""
        if default != "ffprobe" and Path(default).exists():
            return default

        if shutil_which("ffprobe"):
            return "ffprobe"

        # imageio-ffmpeg 不带 ffprobe，用 '' 标记让 _probe 走 OpenCV 兜底
        return ""


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def quick_preprocess(video_path: str, keyframe_interval: float = 2.0) -> PreprocessResult:
    """快速预处理"""
    preprocessor = MediaPreprocessor(keyframe_interval=keyframe_interval)
    return preprocessor.process(video_path)
