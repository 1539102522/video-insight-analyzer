"""
音频识别模型：识别歌曲与版本候选
支持两种方案：
1. 音频指纹匹配（如 audfprint / dejavu）
2. 调用在线 API（如 ShazamIO / ACRCloud）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .evidence_bundle import AudioRecognitionResult, SongCandidate
from .progress import ProgressBar

logger = logging.getLogger(__name__)


class AudioExtractor:
    """
    音频识别器 —— 检测音乐并识别歌曲/版本候选

    检测流程：
    1. 音乐/语音分段检测（VAD + 音乐分类）
    2. 对音乐片段进行指纹匹配
    3. 返回候选歌曲列表
    """

    def __init__(
        self,
        backend: str = "fingerprint",         # fingerprint / shazamio / dejavu
        sample_rate: int = 16000,
        min_music_duration: float = 3.0,      # 最短音乐片段（秒）
        music_flatness_threshold: float = 0.3,  # 绝对平坦度阈值：低于此值视为谐波丰富的音乐
        fingerprint_db_path: str | None = None,
        **kwargs: Any,
    ):
        self.backend = backend
        self.sample_rate = sample_rate
        self.min_music_duration = min_music_duration
        self.music_flatness_threshold = music_flatness_threshold
        self.fingerprint_db_path = fingerprint_db_path
        self._model_kwargs = kwargs

        # 延迟加载
        self._music_classifier = None
        self._fingerprint_engine = None

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def extract(self, audio_path: str | Path) -> AudioRecognitionResult:
        """从音频文件识别歌曲"""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        result = AudioRecognitionResult(
            model_name=f"audio-extractor-{self.backend}",
        )

        try:
            # 加载音频
            logger.info("音频识别: 加载音频中...")
            audio, sr = self._load_audio(audio_path)
            result.audio_duration = len(audio) / sr

            # 步骤 1：检测音乐片段
            logger.info("音频识别: 检测音乐片段中...")
            music_segments = self._detect_music_segments(audio, sr)
            result.has_music = len(music_segments) > 0
            result.music_segments = [
                (round(seg_start, 1), round(seg_end, 1))
                for seg_start, seg_end, _ in music_segments
            ]

            if not result.has_music:
                logger.info("未检测到音乐片段")
                return result

            # 步骤 2：对每个音乐片段进行识别
            logger.info("音频识别: 检测到 %d 个音乐片段，开始识别...", len(music_segments))
            all_candidates: list[SongCandidate] = []
            bar = ProgressBar(total=len(music_segments), label="音频识别")
            for idx, (seg_start, seg_end, seg_confidence) in enumerate(music_segments):
                bar.update(idx + 1, detail=f"{seg_start:.1f}s-{seg_end:.1f}s")
                segment = audio[int(seg_start * sr):int(seg_end * sr)]
                candidates = self._identify_segment(segment, sr, seg_start, seg_end)
                all_candidates.extend(candidates)
            bar.finish()

            # 步骤 3：去重合并
            result.songs = self._deduplicate_candidates(all_candidates)

            # 区分前景/背景音乐（简化：第一个为前景，其余为背景）
            if result.songs:
                result.songs = sorted(
                    result.songs, key=lambda x: x.confidence, reverse=True
                )
                if len(result.songs) > 1:
                    result.background_music = result.songs[1:]

            if result.songs:
                result.overall_confidence = result.songs[0].confidence

            logger.info(
                "音频识别完成: 检测到 %d 首候选歌曲",
                len(result.songs),
            )

        except Exception as e:
            logger.error("音频识别失败: %s", e)
            raise

        return result

    # ------------------------------------------------------------------
    # 音频加载
    # ------------------------------------------------------------------

    def _load_audio(self, path: Path) -> tuple[np.ndarray, int]:
        """加载音频为 numpy 数组"""
        try:
            import soundfile as sf
            audio, sr = sf.read(str(path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # 转单声道
            return audio, sr
        except Exception:
            # 备选：使用 scipy / librosa
            import scipy.io.wavfile as wav
            sr, audio = wav.read(str(path))
            audio = audio.astype(np.float32) / 32768.0
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio, sr

    # ------------------------------------------------------------------
    # 音乐片段检测
    # ------------------------------------------------------------------

    def _detect_music_segments(
        self, audio: np.ndarray, sr: int
    ) -> list[tuple[float, float, float]]:
        """
        检测音频中的音乐片段
        返回 [(start_sec, end_sec, confidence), ...]
        """
        # 整体响度过低（基本无声）直接视为无音乐，避免对静音做无谓分析
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        rms_db = 20.0 * np.log10(rms + 1e-12)
        logger.info("音频整体响度: RMS=%.1f dBFS", rms_db)
        if rms_db < -45.0:
            logger.info("音频过静（RMS < -45 dBFS），视为无音乐")
            return []

        # 方案 A：使用频谱能量 + 平坦度分析区分音乐/语音
        segments = self._energy_based_music_detection(audio, sr)

        # 过滤过短的片段
        segments = [
            s for s in segments
            if s[1] - s[0] >= self.min_music_duration
        ]
        return segments

    def _energy_based_music_detection(
        self, audio: np.ndarray, sr: int, window_size: float = 1.0
    ) -> list[tuple[float, float, float]]:
        """
        基于频谱能量的音乐检测
        音乐通常比语音有更丰富的谐波和更持续的频谱
        """
        import librosa

        hop_length = 512
        window_samples = int(window_size * sr)

        # 计算梅尔频谱
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, hop_length=hop_length, n_mels=128
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 计算每帧的能量和频谱平坦度
        energy = np.mean(mel_db, axis=0)
        spectral_flatness = self._compute_spectral_flatness(mel_spec)

        # 音乐特征：高能量 + 低平坦度（谐波丰富）；语音/噪声：平坦度高
        # 注意：能量阈值相对本文件取 30 分位，但平坦度必须用绝对阈值——
        # 若平坦度也相对自身取分位，任何音频都必然有固定比例的帧被判为
        # “非音乐”，会把音乐片段切得稀碎（曾导致整段音乐被漏检）。
        energy_threshold = np.percentile(energy, 30)
        flatness_threshold = self.music_flatness_threshold

        # 标记音乐帧
        is_music = (energy > energy_threshold) & (spectral_flatness < flatness_threshold)

        # 诊断日志：音乐帧占比与最长连续段，便于排查漏检
        longest = 0
        run = 0
        for v in is_music:
            run = run + 1 if v else 0
            longest = max(longest, run)
        logger.info(
            "音乐帧占比 %.0f%%，最长连续 %.1f 秒（能量阈值 %.1f dB, 平坦度阈值 %.2f, 需连续 >= %.0f 秒）",
            float(is_music.mean()) * 100, longest * hop_length / sr,
            energy_threshold, flatness_threshold, self.min_music_duration,
        )

        # 合并连续的 True（允许 1 秒内的间歇，容忍鼓点等瞬态帧）
        return self._merge_segments(is_music, hop_length, sr, min_gap=1.0)

    def _compute_spectral_flatness(self, spec: np.ndarray) -> np.ndarray:
        """计算频谱平坦度"""
        geo_mean = np.exp(np.mean(np.log(spec + 1e-10), axis=0))
        arith_mean = np.mean(spec, axis=0) + 1e-10
        return geo_mean / arith_mean

    @staticmethod
    def _merge_segments(
        mask: np.ndarray, hop_length: int, sr: int, min_gap: float = 0.5
    ) -> list[tuple[float, float, float]]:
        """合并连续的 True 帧为时间段，容忍不超过 min_gap 秒的间隙"""
        segments: list[tuple[float, float, float]] = []
        if len(mask) == 0:
            return segments

        gap_frames = max(1, int(min_gap * sr / hop_length))
        start = 0        # 当前片段起始帧
        last_true = 0    # 当前片段最近一个 True 帧
        in_segment = False

        for i, val in enumerate(mask):
            if val:
                if not in_segment:
                    start = i
                    in_segment = True
                last_true = i
            else:
                # 用“距最近 True 帧的距离”判断间隙，而不是距片段起点
                if in_segment and i - last_true > gap_frames:
                    start_sec = start * hop_length / sr
                    end_sec = (last_true + 1) * hop_length / sr
                    confidence = float(np.mean(mask[start:last_true + 1]))
                    segments.append((start_sec, end_sec, confidence))
                    in_segment = False

        if in_segment:
            start_sec = start * hop_length / sr
            end_sec = (last_true + 1) * hop_length / sr
            confidence = float(np.mean(mask[start:last_true + 1]))
            segments.append((start_sec, end_sec, confidence))

        return segments

    # ------------------------------------------------------------------
    # 歌曲识别
    # ------------------------------------------------------------------

    def _identify_segment(
        self, audio: np.ndarray, sr: int, start_sec: float, end_sec: float
    ) -> list[SongCandidate]:
        """对单个音频片段进行歌曲识别"""
        if self.backend == "shazamio":
            return self._shazamio_identify(audio, sr, start_sec, end_sec)
        elif self.backend == "fingerprint":
            return self._fingerprint_identify(audio, sr, start_sec, end_sec)
        elif self.backend == "dejavu":
            return self._dejavu_identify(audio, sr, start_sec, end_sec)
        else:
            logger.warning("未知音频识别后端: %s", self.backend)
            return []

    # ------------------------------------------------------------------
    # ShazamIO（在线 API）
    # ------------------------------------------------------------------

    def _shazamio_identify(
        self, audio: np.ndarray, sr: int, start_sec: float, end_sec: float
    ) -> list[SongCandidate]:
        """使用 ShazamIO 识别歌曲"""
        try:
            import asyncio
            from shazamio import Shazam
            import tempfile

            # 保存临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                import soundfile as sf
                sf.write(f.name, audio, sr)
                tmp_path = f.name

            async def _recognize():
                shazam = Shazam()
                result = await shazam.recognize(tmp_path)
                return result

            result = asyncio.run(_recognize())

            # 清理临时文件
            Path(tmp_path).unlink(missing_ok=True)

            track = result.get("track", {})
            if not track:
                return []

            return [SongCandidate(
                title=track.get("title", "未知"),
                artist=track.get("subtitle", "未知"),
                version="",
                confidence=0.85,
                match_segment=(start_sec, end_sec),
            )]

        except ImportError:
            logger.warning("shazamio 未安装，跳过 Shazam 识别")
            return []
        except Exception as e:
            logger.warning("Shazam 识别失败: %s", e)
            return []

    # ------------------------------------------------------------------
    # 音频指纹（本地）
    # ------------------------------------------------------------------

    def _fingerprint_identify(
        self, audio: np.ndarray, sr: int, start_sec: float, end_sec: float
    ) -> list[SongCandidate]:
        """
        基于音频指纹的本地识别
        这是一个简化实现，生产环境建议接入完整指纹库
        """
        # 提取音频指纹特征
        fingerprint = self._extract_fingerprint(audio, sr)

        # 如果有指纹数据库，进行匹配
        if self.fingerprint_db_path:
            return self._match_fingerprint_db(fingerprint, start_sec, end_sec)

        # 否则返回指纹信息（供后续使用）
        logger.debug(
            "提取音频指纹: 片段 %.1f-%.1f, 指纹长度 %d",
            start_sec, end_sec, len(fingerprint),
        )
        return []  # 无数据库时返回空

    def _extract_fingerprint(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """提取音频指纹（简化的频谱峰值）"""
        try:
            import librosa

            # 计算常数Q变换（CQT）作为指纹特征
            cqt = np.abs(librosa.cqt(
                audio, sr=sr, hop_length=512, n_bins=84, bins_per_octave=12
            ))
            # 取每帧的最大响应对应的 bin 作为简化指纹
            peaks = np.argmax(cqt, axis=0)
            return peaks
        except Exception:
            # 简化：使用 STFT 的峰值
            n_fft = 2048
            hop = 512
            frames = len(audio) // hop
            fingerprint = np.zeros(frames)
            for i in range(frames):
                chunk = audio[i * hop:i * hop + n_fft]
                if len(chunk) < n_fft:
                    break
                spec = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
                fingerprint[i] = np.argmax(spec)
            return fingerprint

    def _match_fingerprint_db(
        self, fingerprint: np.ndarray, start_sec: float, end_sec: float
    ) -> list[SongCandidate]:
        """与指纹数据库匹配（占位实现）"""
        # TODO: 实现实际的指纹数据库匹配逻辑
        # 可参考 audfprint / dejavu 等库
        return []

    # ------------------------------------------------------------------
    # Dejavu（本地指纹匹配）
    # ------------------------------------------------------------------

    def _dejavu_identify(
        self, audio: np.ndarray, sr: int, start_sec: float, end_sec: float
    ) -> list[SongCandidate]:
        """使用 Dejavu 进行音频指纹匹配"""
        try:
            from dejavu import Dejavu
            from dejavu.logic.recognizer.file_recognizer import FileRecognizer
            import tempfile

            if self._fingerprint_engine is None:
                config = {
                    "database": {
                        "host": "127.0.0.1",
                        "user": "root",
                        "password": "",
                        "database": "dejavu",
                    }
                }
                self._fingerprint_engine = Dejavu(config)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                import soundfile as sf
                sf.write(f.name, audio, sr)
                tmp_path = f.name

            result = self._fingerprint_engine.recognize(
                FileRecognizer, tmp_path
            )

            Path(tmp_path).unlink(missing_ok=True)

            if result and result.get("song_name"):
                return [SongCandidate(
                    title=result.get("song_name", "未知"),
                    artist=result.get("artist", "未知"),
                    version="",
                    confidence=result.get("confidence", 0.0),
                    match_segment=(start_sec, end_sec),
                    fingerprint_id=result.get("song_id", ""),
                )]

            return []

        except ImportError:
            logger.debug("dejavu 未安装")
            return []
        except Exception as e:
            logger.warning("Dejavu 识别失败: %s", e)
            return []

    # ------------------------------------------------------------------
    # 去重
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate_candidates(
        candidates: list[SongCandidate],
        title_threshold: float = 0.8,
    ) -> list[SongCandidate]:
        """合并相同歌曲的候选"""
        if not candidates:
            return []

        # 按歌名分组，保留置信度最高的
        seen: dict[str, SongCandidate] = {}
        for c in candidates:
            key = f"{c.title}|{c.artist}"
            if key in seen:
                if c.confidence > seen[key].confidence:
                    seen[key] = c
            else:
                seen[key] = c

        return sorted(
            seen.values(), key=lambda x: x.confidence, reverse=True
        )
