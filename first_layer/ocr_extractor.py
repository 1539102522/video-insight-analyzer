"""
OCR模型：提取画面字幕
使用 PaddleOCR 或 EasyOCR 识别关键帧中的文字
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .evidence_bundle import OCREvidence, OCRSegment
from .progress import ProgressBar

logger = logging.getLogger(__name__)


class OCRExtractor:
    """画面文字识别器 —— 从关键帧中提取字幕/文字"""

    def __init__(
        self,
        engine: str = "paddleocr",          # paddleocr / easyocr
        lang: str = "ch",                    # PaddleOCR: ch / en / chinese_cht
        use_gpu: bool = True,
        min_confidence: float = 0.5,
        max_side: int = 0,                   # 帧最长边限制（0=原图；如 1280 可提速 OCR）
        max_frames: int = 0,                 # 最大处理帧数（0=全部帧；>0 均匀抽样提速）
        model_dir: str | None = None,        # 本地模型目录（EasyOCR）
        download_enabled: bool = True,       # 是否允许自动下载模型
        **kwargs: Any,
    ):
        self.engine = engine
        self.lang = lang
        self.use_gpu = use_gpu
        self.min_confidence = min_confidence
        self.max_side = max_side
        self.max_frames = max_frames
        if model_dir is None:
            # 默认用项目内 models/easyocr 目录（模型已随项目存放）
            _root = Path(__file__).resolve().parent.parent
            _default = _root / "models" / "easyocr"
            self.model_dir = str(_default) if _default.is_dir() else None
        else:
            self.model_dir = model_dir
        self.download_enabled = download_enabled
        self.model = None
        self._model_error: Exception | None = None  # 缓存加载失败，避免逐帧重复下载
        self._model_kwargs = kwargs

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def extract_from_dir(
        self, keyframe_dir: str | Path, interval: float = 2.0
    ) -> OCREvidence:
        """对关键帧目录中的所有帧进行 OCR"""
        keyframe_dir = Path(keyframe_dir)
        if not keyframe_dir.exists():
            raise FileNotFoundError(f"关键帧目录不存在: {keyframe_dir}")

        frame_files = sorted(keyframe_dir.glob("*.jpg")) + \
                       sorted(keyframe_dir.glob("*.png")) + \
                       sorted(keyframe_dir.glob("*.jpeg"))

        if not frame_files:
            logger.warning("关键帧目录为空: %s", keyframe_dir)
            return OCREvidence(model_name=self.engine)

        if self.max_frames > 0 and len(frame_files) > self.max_frames:
            # 均匀抽样控制帧数上限（长视频提速，字幕/文字通常分布均匀）
            total = len(frame_files)
            idxs = [round(i * (total - 1) / (self.max_frames - 1))
                    for i in range(self.max_frames)]
            frame_files = [frame_files[i] for i in sorted(set(idxs))]
            logger.info("OCR 帧数 %d → 抽样 %d 帧（max_frames=%d）",
                        total, len(frame_files), self.max_frames)

        evidence = OCREvidence(model_name=self.engine)
        all_segments: list[OCRSegment] = []

        bar = ProgressBar(total=len(frame_files), label="OCR")
        for i, fpath in enumerate(frame_files):
            timestamp = i * interval
            bar.update(i + 1, detail=fpath.name)
            try:
                segments = self._ocr_single(fpath, frame_index=i, timestamp=timestamp)
                all_segments.extend(segments)
            except Exception as e:
                logger.warning("OCR 帧 %s 失败: %s", fpath.name, e)
                # 模型加载失败属于全局错误，无需逐帧重试
                if self._model_error is not None:
                    logger.error("OCR 模型不可用，跳过剩余帧: %s", self._model_error)
                    break
        bar.finish()

        evidence.segments = all_segments
        evidence.full_text = "\n".join(s.text for s in all_segments)
        if all_segments:
            evidence.overall_confidence = sum(
                s.confidence for s in all_segments
            ) / len(all_segments)

        logger.info(
            "OCR 完成: %d 帧, %d 个文字区域, 平均置信度 %.3f",
            len(frame_files), len(all_segments), evidence.overall_confidence,
        )
        return evidence

    def extract_single(self, image_path: str | Path, frame_index: int = 0,
                       timestamp: float = 0.0) -> list[OCRSegment]:
        """对单张图片进行 OCR"""
        return self._ocr_single(Path(image_path), frame_index, timestamp)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _ocr_single(self, image_path: Path, frame_index: int,
                    timestamp: float) -> list[OCRSegment]:
        if self.engine == "paddleocr":
            return self._paddle_ocr(image_path, frame_index, timestamp)
        elif self.engine == "easyocr":
            return self._easy_ocr(image_path, frame_index, timestamp)
        else:
            raise ValueError(f"不支持的 OCR 引擎: {self.engine}")

    def _paddle_ocr(self, image_path: Path, frame_index: int,
                    timestamp: float) -> list[OCRSegment]:
        """使用 PaddleOCR"""
        model = self._load_paddle_model()
        result = model.ocr(str(image_path), cls=True)

        if result is None or result[0] is None:
            return []

        segments: list[OCRSegment] = []
        for line in result[0]:
            bbox_points = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            text = line[1][0]
            confidence = line[1][1]

            if confidence < self.min_confidence:
                continue

            # 转换为 [x1, y1, x2, y2] 格式
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

            segments.append(OCRSegment(
                text=text,
                bbox=bbox,
                frame_index=frame_index,
                timestamp=timestamp,
                confidence=confidence,
            ))

        return segments

    def _easy_ocr(self, image_path: Path, frame_index: int,
                  timestamp: float) -> list[OCRSegment]:
        """使用 EasyOCR"""
        model = self._load_easy_model()
        # 用 PIL 加载后传 numpy 数组，避免 EasyOCR 内部 cv2.imread
        # 在 Windows 上无法打开中文路径的问题
        import numpy as np
        from PIL import Image
        im = Image.open(image_path).convert("RGB")
        if self.max_side > 0 and max(im.size) > self.max_side:
            # 帧压缩：长边限制可显著提速 OCR，字幕识别基本不受影响
            ratio = self.max_side / max(im.size)
            im = im.resize((max(1, int(im.width * ratio)), max(1, int(im.height * ratio))))
        image = np.array(im)
        result = model.readtext(image)

        segments: list[OCRSegment] = []
        for (bbox_points, text, confidence) in result:
            if confidence < self.min_confidence:
                continue

            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

            segments.append(OCRSegment(
                text=text,
                bbox=bbox,
                frame_index=frame_index,
                timestamp=timestamp,
                confidence=confidence,
            ))

        return segments

    def _load_paddle_model(self):
        if self.model is not None:
            return self.model
        if self._model_error is not None:
            raise self._model_error
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            self._model_error = e
            raise ImportError("请安装 PaddleOCR: pip install paddlepaddle paddleocr") from e

        logger.info("加载 PaddleOCR 模型 (lang=%s) —— 首次运行需下载模型，请耐心等待...", self.lang)
        try:
            self.model = PaddleOCR(
                lang=self.lang,
                use_gpu=self.use_gpu,
                **self._model_kwargs,
            )
        except Exception as e:
            self._model_error = e
            raise
        return self.model

    def _load_easy_model(self):
        if self.model is not None:
            return self.model
        if self._model_error is not None:
            raise self._model_error
        try:
            import easyocr
        except ImportError as e:
            self._model_error = e
            raise ImportError("请安装 EasyOCR: pip install easyocr") from e

        gpu = self.use_gpu
        # EasyOCR 语言代码映射：EasyOCR 使用 ch_sim / ch_tra，不接受 'ch'
        lang_alias = {"ch": "ch_sim", "cht": "ch_tra"}
        if isinstance(self.lang, str):
            langs = [self.lang]
        else:
            langs = list(self.lang)
        easy_langs = [lang_alias.get(l, l) for l in langs]

        if self.download_enabled:
            logger.info(
                "加载 EasyOCR 模型 (lang=%s -> %s, gpu=%s) —— 首次运行需下载模型，请耐心等待...",
                self.lang, easy_langs, gpu,
            )
        else:
            logger.info(
                "加载 EasyOCR 模型 (lang=%s -> %s, gpu=%s, 离线模式, 目录=%s)",
                self.lang, easy_langs, gpu,
                self.model_dir or "默认 项目 models/easyocr",
            )
        try:
            self.model = easyocr.Reader(
                easy_langs,
                gpu=gpu,
                model_storage_directory=self.model_dir,
                download_enabled=self.download_enabled,
                **self._model_kwargs,
            )
        except Exception as e:
            self._model_error = e
            raise
        return self.model
