"""
第一层总调度管线
按流程图串联：预处理 → ASR/OCR/视觉/音频 并行提取 → 统一证据包
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any

from .evidence_bundle import (
    EvidenceBundle,
    ASREvidence,
    OCREvidence,
    VisualEvidence,
    AudioRecognitionResult,
    ContentCategory,
)
from .media_preprocessor import MediaPreprocessor, PreprocessResult
from .asr_extractor import ASRExtractor
from .ocr_extractor import OCRExtractor
from .visual_extractor import VisualExtractor
from .audio_extractor import AudioExtractor
from .extractors import (
    EXTRACTOR_REGISTRY,
    ExtractorSpec,
    register_extractor,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 内置提取器注册（扩展点：新增模型在这里 register_extractor 即可）
# ---------------------------------------------------------------------------


def _asr_kwargs(p: "FirstLayerPipeline") -> dict[str, Any]:
    return {
        "model_size": p.asr_model_size,
        "device": p.asr_device,
        "model_dir": p.asr_model_dir,
        "use_hf_mirror": p.asr_use_hf_mirror,
        "beam_size": p.asr_beam,
        "cpu_threads": p.asr_cpu_threads,
    }


def _ocr_kwargs(p: "FirstLayerPipeline") -> dict[str, Any]:
    return {
        "engine": p.ocr_engine,
        "lang": p.ocr_lang,
        "use_gpu": p.ocr_use_gpu,
        "model_dir": p.ocr_model_dir,
        "download_enabled": p.ocr_download_enabled,
        "max_side": p.ocr_max_side,
        "max_frames": p.ocr_max_frames,
    }


def _visual_kwargs(p: "FirstLayerPipeline") -> dict[str, Any]:
    return {"backend": p.visual_backend, "model_name": p.visual_model,
            "max_frames": p.vlm_max_frames}


def _audio_kwargs(p: "FirstLayerPipeline") -> dict[str, Any]:
    return {"backend": p.audio_backend}


register_extractor(ExtractorSpec(
    id="asr", name="ASR 语音识别", input_key="audio",
    cls=ASRExtractor, make_kwargs=_asr_kwargs,
    backends=["faster-whisper"],
))
register_extractor(ExtractorSpec(
    id="ocr", name="OCR 画面文字", input_key="keyframes",
    cls=OCRExtractor, make_kwargs=_ocr_kwargs,
    backends=["easyocr"],
))
register_extractor(ExtractorSpec(
    id="visual", name="视觉理解", input_key="keyframes",
    cls=VisualExtractor, make_kwargs=_visual_kwargs,
    backends=["clip", "vlm"],
    backend_labels={"clip": "CLIP (ViT-B/32)", "vlm": "InternVL2-2B (VLM)"},
))
register_extractor(ExtractorSpec(
    id="audio", name="音频识别", input_key="audio",
    cls=AudioExtractor, make_kwargs=_audio_kwargs,
    backends=["fingerprint", "shazamio", "dejavu"],
))


class FirstLayerPipeline:
    """
    第一层：专业模型提取事实

    流程：
    1. 媒体预处理（FFmpeg 转码/抽音频/关键帧）
    2. 并行执行四个专业模型：
       - 中文ASR    → 语音文本 + 时间戳
       - OCR        → 画面字幕
       - 视觉理解    → 关键帧描述
       - 音频识别    → 歌曲候选
    3. 汇总为统一证据包 EvidenceBundle
    """

    def __init__(
        self,
        # 预处理参数
        keyframe_interval: float = 2.0,
        ffmpeg_bin: str = "ffmpeg",
        ffprobe_bin: str = "ffprobe",
        work_dir: str | None = None,
        # 模型参数
        asr_model_size: str = "medium",
        asr_device: str = "auto",
        asr_model_dir: str | None = None,
        asr_use_hf_mirror: bool = True,
        asr_beam: int = 5,
        asr_cpu_threads: int = 4,
        ocr_engine: str = "easyocr",
        ocr_lang: str = "ch",
        ocr_use_gpu: bool = True,
        ocr_model_dir: str | None = None,
        ocr_download_enabled: bool = True,
        ocr_max_side: int = 0,
        ocr_max_frames: int = 0,
        visual_backend: str = "clip",
        visual_model: str = "OpenGVLab/InternVL2-2B",
        vlm_max_frames: int = 12,
        audio_backend: str = "fingerprint",
        # 并行控制
        max_workers: int = 4,
        model_timeout: float | None = None,   # 单模型超时（秒），None 表示不超时
        enable_asr: bool = True,
        enable_ocr: bool = True,
        enable_visual: bool = True,
        enable_audio: bool = True,
        enable_map: dict[str, bool] | None = None,  # 通用开关（id->bool，覆盖上面的开关）
        **kwargs: Any,
    ):
        # 预处理
        self.preprocessor = MediaPreprocessor(
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
            work_dir=work_dir,
            keyframe_interval=keyframe_interval,
        )

        # 各模型（延迟初始化，实例缓存在 _ext_<id> 属性上）
        self._asr_extractor: ASRExtractor | None = None
        self._ocr_extractor: OCRExtractor | None = None
        self._visual_extractor: VisualExtractor | None = None
        self._audio_extractor: AudioExtractor | None = None

        # 模型配置
        self.asr_model_size = asr_model_size
        self.asr_device = asr_device
        self.asr_model_dir = asr_model_dir
        self.asr_use_hf_mirror = asr_use_hf_mirror
        self.asr_beam = asr_beam
        self.asr_cpu_threads = asr_cpu_threads
        self.ocr_engine = ocr_engine
        self.ocr_lang = ocr_lang
        self.ocr_use_gpu = ocr_use_gpu
        self.ocr_model_dir = ocr_model_dir
        self.ocr_download_enabled = ocr_download_enabled
        self.ocr_max_side = ocr_max_side
        self.ocr_max_frames = ocr_max_frames
        self.visual_backend = visual_backend
        self.visual_model = visual_model
        self.vlm_max_frames = vlm_max_frames
        self.audio_backend = audio_backend

        # 通用开关表：注册表里的模型都可控，未在表里的用注册表默认值
        self.enable_map: dict[str, bool] = {
            "asr": enable_asr,
            "ocr": enable_ocr,
            "visual": enable_visual,
            "audio": enable_audio,
        }
        if enable_map:
            self.enable_map.update(enable_map)

        self.max_workers = max_workers
        self.model_timeout = model_timeout

        # 模型加载互斥锁：保证多个模型串行初始化，
        # 避免多线程同时加载原生库（如 torch / ctranslate2 的 OpenMP）引发冲突或卡死。
        self._model_load_lock = threading.Lock()

        self._extra_kwargs = kwargs

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def run(self, video_path: str | Path) -> EvidenceBundle:
        """
        执行完整的第一层提取管线

        Args:
            video_path: 短视频文件路径（支持 MP4/MOV 等）

        Returns:
            EvidenceBundle: 统一证据包
        """
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        logger.info("=" * 60)
        logger.info("第一层管线启动: %s", video_path.name)
        logger.info("=" * 60)
        t0 = time.time()

        # ------------------------------------------------------------------
        # 步骤 1：媒体预处理
        # ------------------------------------------------------------------
        logger.info("[1/3] 媒体预处理...")
        try:
            preprocess = self.preprocessor.process(video_path)
        except Exception as e:
            logger.error("预处理失败: %s", e)
            # 创建最小 bundle 记录错误
            return EvidenceBundle(
                video_path=str(video_path),
                errors=[{"stage": "preprocess", "error": str(e)}],
            )

        logger.info(
            "  时长: %.1fs, 分辨率: %dx%d, 音频: %s, 关键帧: %d张",
            preprocess.media_info.duration,
            preprocess.media_info.width,
            preprocess.media_info.height,
            "有" if preprocess.media_info.has_audio else "无",
            preprocess.keyframe_count,
        )

        # ------------------------------------------------------------------
        # 步骤 2：并行执行四个专业模型
        # ------------------------------------------------------------------
        logger.info("[2/3] 专业模型并行提取...")

        bundle = EvidenceBundle(
            video_path=str(video_path),
            video_duration=preprocess.media_info.duration,
            category_hint=ContentCategory.UNKNOWN,
        )

        # 构建任务列表（由注册表驱动，可扩展）
        tasks: dict[str, dict[str, Any]] = {}
        ctx = {
            "audio": preprocess.audio_path,
            "keyframes": preprocess.keyframe_dir,
            "interval": preprocess.keyframe_interval,
        }
        enabled_ids: list[str] = []
        for sid, spec in EXTRACTOR_REGISTRY.items():
            if not self.enable_map.get(sid, spec.default_enabled):
                continue
            if spec.input_key == "audio" and not preprocess.audio_path:
                continue
            if spec.input_key == "keyframes" and not preprocess.keyframe_dir:
                continue
            tasks[sid] = {"fn": self._run_extractor, "args": (sid, ctx)}
            enabled_ids.append(sid)

        # VLM 模式：视觉理解先串行执行（VLM 占满 CPU，且与 EasyOCR 并发会触发
        # torch 设备冲突 "Tensor on device meta..."），再并行跑其余模型
        if self.visual_backend == "vlm" and "visual" in tasks and len(tasks) > 1:
            logger.info("VLM 后端：先执行视觉理解，再并行执行其余模型（避免 torch 并发冲突）")
            task = tasks.pop("visual")
            self._execute_and_assign(bundle, "visual", task["fn"], task["args"])

        # 并行执行（当只有 1 个任务时直接用主线程）
        if len(tasks) <= 1:
            for name, task in tasks.items():
                self._execute_and_assign(bundle, name, task["fn"], task["args"])
        else:
            self._run_parallel(bundle, tasks)

        # ------------------------------------------------------------------
        # 步骤 3：初步分类提示
        # ------------------------------------------------------------------
        logger.info("[3/3] 生成分类提示...")
        bundle.category_hint = self._classify_hint(bundle)

        elapsed = time.time() - t0
        state = ", ".join(
            f"{spec.id}={getattr(bundle, spec.result_field) is not None}"
            for spec in EXTRACTOR_REGISTRY.values()
        )
        logger.info(
            "第一层管线完成: 耗时 %.1fs, %s, 分类提示=%s",
            elapsed, state, bundle.category_hint.value,
        )

        return bundle

    # ------------------------------------------------------------------
    # 各模型执行方法
    # ------------------------------------------------------------------

    def _get_or_init(self, attr_name: str, factory: Any) -> Any:
        """串行地延迟初始化模型（双重检查锁），避免多线程同时加载原生库。"""
        extractor = getattr(self, attr_name, None)
        if extractor is not None:
            return extractor
        with self._model_load_lock:
            extractor = getattr(self, attr_name, None)
            if extractor is None:
                logger.info("初始化模型: %s ...", attr_name.lstrip("_"))
                extractor = factory()
                setattr(self, attr_name, extractor)
        return extractor

    def _run_extractor(self, extractor_id: str, ctx: dict[str, Any]) -> Any:
        """按注册表描述创建/运行一个提取器（通用入口）"""
        spec = EXTRACTOR_REGISTRY[extractor_id]

        def factory() -> Any:
            kwargs = spec.make_kwargs(self) if spec.make_kwargs else {}
            return spec.cls(**kwargs)

        instance = self._get_or_init(f"_ext_{extractor_id}", factory)
        if spec.run_audio is not None:
            return spec.run_audio(instance, ctx["audio"])
        if spec.run_keyframes is not None:
            return spec.run_keyframes(instance, ctx["keyframes"], ctx["interval"])
        if spec.input_key == "audio":
            return instance.extract(ctx["audio"])
        return instance.extract_from_dir(ctx["keyframes"], ctx["interval"])

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _run_parallel(self, bundle: EvidenceBundle, tasks: dict[str, dict[str, Any]]) -> None:
        """
        并行执行多个模型任务：
        - 已完成的任务结果立即写入 bundle；
        - 单个任务抛异常时记录错误，不影响其他任务；
        - 设置 model_timeout 时，超时的任务标记为错误并跳过。
        """
        logger.info(
            "并行执行 %d 个模型任务: %s",
            len(tasks), ", ".join(tasks.keys()),
        )
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(tasks))
        ) as executor:
            future_map: dict[Any, str] = {
                executor.submit(task["fn"], *task["args"]): name
                for name, task in tasks.items()
            }
            pending = dict(future_map)

            while pending:
                if self.model_timeout is not None:
                    done, _ = wait(
                        pending.keys(),
                        timeout=self.model_timeout,
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
                        # 超时：剩余任务全部标记为超时错误，不再等待
                        for future, name in list(pending.items()):
                            logger.error(
                                "模型 [%s] 执行超时（>%ss），已跳过",
                                name, self.model_timeout,
                            )
                            bundle.errors.append({
                                "stage": name,
                                "error": f"timeout after {self.model_timeout}s",
                            })
                        break
                else:
                    done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)

                for future in done:
                    name = pending.pop(future)
                    try:
                        result = future.result()
                        self._assign_result(bundle, name, result)
                        logger.info(
                            "模型 [%s] 完成 ✓ (剩余 %d 个: %s)",
                            name, len(pending), ", ".join(pending.values()),
                        )
                    except Exception as e:
                        logger.error("模型 [%s] 执行失败: %s", name, e)
                        bundle.errors.append({
                            "stage": name,
                            "error": str(e),
                        })

    def _execute_and_assign(
        self, bundle: EvidenceBundle, name: str, fn: Any, args: tuple
    ) -> None:
        try:
            result = fn(*args)
            self._assign_result(bundle, name, result)
        except Exception as e:
            logger.error("模型 [%s] 执行失败: %s", name, e)
            bundle.errors.append({"stage": name, "error": str(e)})

    @staticmethod
    def _assign_result(bundle: EvidenceBundle, name: str, result: Any) -> None:
        """把模型结果写入证据包（字段名由注册表的 result_field 决定，可扩展）"""
        spec = EXTRACTOR_REGISTRY.get(name)
        field = spec.result_field if spec else name
        setattr(bundle, field, result)

    @staticmethod
    def _classify_hint(bundle: EvidenceBundle) -> ContentCategory:
        """
        根据已有证据给出初步分类提示
        精确分类由第二层大模型完成
        """
        # 从 ASR 和 OCR 文本中检测关键词
        all_text = bundle.all_text().lower()

        # 音乐类关键词
        music_keywords = ["歌", "唱", "音乐", "演唱", "专辑", "歌手", "乐队", "翻唱",
                          "remix", "cover", "原唱", "伴奏", "歌曲"]
        # 菜谱类关键词
        recipe_keywords = ["食材", "步骤", "做法", "烹饪", "克", "毫升", "火", "锅",
                           "炒", "煮", "蒸", "炸", "烤", "菜谱", "食谱"]
        # 文稿类关键词
        prose_keywords = ["文章", "作者", "段落", "篇", "读", "朗诵", "文学"]

        music_score = sum(1 for kw in music_keywords if kw in all_text)
        recipe_score = sum(1 for kw in recipe_keywords if kw in all_text)
        prose_score = sum(1 for kw in prose_keywords if kw in all_text)

        # 如果有音频识别结果中的歌曲
        if bundle.audio and bundle.audio.songs:
            music_score += 3

        scores = {
            ContentCategory.MUSIC: music_score,
            ContentCategory.RECIPE: recipe_score,
            ContentCategory.PROSE: prose_score,
        }

        # 如果所有得分都很低
        max_score = max(scores.values())
        if max_score == 0:
            return ContentCategory.UNKNOWN

        # 如果多个类别得分接近
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[0][1] - sorted_scores[1][1] <= 1:
            return ContentCategory.MIXED

        return sorted_scores[0][0]


# ---------------------------------------------------------------------------
# 便捷入口
# ---------------------------------------------------------------------------

def extract_evidence(
    video_path: str,
    keyframe_interval: float = 2.0,
    asr_model: str = "medium",
    asr_device: str = "auto",
    asr_model_dir: str | None = None,
    asr_use_hf_mirror: bool = True,
    asr_beam: int = 5,
    asr_cpu_threads: int = 4,
    ocr_engine: str = "easyocr",
    ocr_lang: str = "ch",
    ocr_model_dir: str | None = None,
    ocr_download_enabled: bool = True,
    ocr_max_side: int = 0,
    ocr_max_frames: int = 0,
    visual_backend: str = "clip",
    visual_model: str = "OpenGVLab/InternVL2-2B",
    vlm_max_frames: int = 12,
    audio_backend: str = "fingerprint",
    enable_asr: bool = True,
    enable_ocr: bool = True,
    enable_visual: bool = True,
    enable_audio: bool = True,
    enable_map: dict[str, bool] | None = None,  # 通用开关（id->bool，覆盖上面开关）
    model_timeout: float | None = None,
    verbose: bool = True,
) -> EvidenceBundle:
    """
    快速入口：对短视频执行第一层证据提取

    Example:
        bundle = extract_evidence("video.mp4")
        print(bundle.summary())
        print(bundle.asr.full_text)
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )

    pipeline = FirstLayerPipeline(
        keyframe_interval=keyframe_interval,
        asr_model_size=asr_model,
        asr_device=asr_device,
        asr_model_dir=asr_model_dir,
        asr_use_hf_mirror=asr_use_hf_mirror,
        asr_beam=asr_beam,
        asr_cpu_threads=asr_cpu_threads,
        ocr_engine=ocr_engine,
        ocr_lang=ocr_lang,
        ocr_model_dir=ocr_model_dir,
        ocr_download_enabled=ocr_download_enabled,
        ocr_max_side=ocr_max_side,
        ocr_max_frames=ocr_max_frames,
        visual_backend=visual_backend,
        visual_model=visual_model,
        vlm_max_frames=vlm_max_frames,
        audio_backend=audio_backend,
        enable_asr=enable_asr,
        enable_ocr=enable_ocr,
        enable_visual=enable_visual,
        enable_audio=enable_audio,
        enable_map=enable_map,
        model_timeout=model_timeout,
    )

    return pipeline.run(video_path)
