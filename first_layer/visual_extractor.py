"""
视觉理解模型：描述关键帧内容
使用轻量级 VLM（如 Qwen2-VL、InternVL2 等）或备选的 CLIP + 检测方案
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .evidence_bundle import KeyFrameDescription, VisualEvidence
from .progress import ProgressBar

logger = logging.getLogger(__name__)

# 常见菜品/食物候选（英文标签 → 中文名）。OpenAI CLIP 为英文训练，中文标签效果差。
FOOD_CANDIDATES: dict[str, str] = {
    "braised pork belly": "红烧肉", "hot pot": "火锅", "barbecue skewers": "烧烤",
    "spicy hot pot": "麻辣烫", "dumplings": "饺子", "steamed buns": "包子",
    "noodles": "面条", "fried rice": "炒饭", "chow mein": "炒面",
    "ramen": "拉面", "rice noodles": "米线", "congee porridge": "粥",
    "pancake": "煎饼", "fried dough sticks": "油条", "soy milk": "豆浆",
    "boiled egg": "鸡蛋", "vegetables": "蔬菜", "salad": "沙拉",
    "fruit": "水果", "strawberry": "草莓", "cake": "蛋糕", "bread": "面包",
    "bubble tea": "奶茶", "coffee": "咖啡", "juice": "果汁", "beer": "啤酒",
    "crayfish": "小龙虾", "crab": "螃蟹", "shrimp": "虾", "grilled fish": "烤鱼",
    "steak": "牛排", "fried chicken": "炸鸡", "roast chicken": "烤鸡",
    "hamburger": "汉堡", "pizza": "披萨", "sushi": "寿司",
    "korean barbecue": "烤肉", "hot pot sliced mutton": "涮羊肉",
    "sweet and sour pork": "糖醋里脊", "kung pao chicken": "宫保鸡丁",
    "mapo tofu": "麻婆豆腐", "scrambled eggs with tomato": "番茄炒蛋",
    "shredded potato": "土豆丝", "green vegetables": "青菜", "soup": "汤",
    "spicy stir fry pot": "麻辣香锅", "roast duck": "烤鸭",
    "pork trotters": "猪蹄", "zongzi": "粽子", "mooncake": "月饼",
    "tangyuan glutinous rice balls": "汤圆", "ice cream": "冰淇淋",
    "dessert": "甜品", "cookies": "饼干", "chocolate": "巧克力",
    "steamed fish": "清蒸鱼", "sichuan boiled fish": "水煮鱼",
    "pickled cabbage fish": "酸菜鱼", "chicken wings": "鸡翅",
}

# 触发菜品识别的情景
FOOD_SCENES = ("美食食物特写", "厨房", "餐厅", "菜谱", "产品展示")


class VisualExtractor:
    """
    视觉理解提取器 —— 描述关键帧的画面内容

    支持多种后端：
    - vlm: 使用 VLM（Qwen2-VL / InternVL2）端到端描述
    - clip: 使用 CLIP + YOLO 进行物体检测和场景分类（轻量备选）
    """

    def __init__(
        self,
        backend: str = "vlm",
        model_name: str = "OpenGVLab/InternVL2-2B",
        device: str = "auto",
        prompt_template: str | None = None,
        max_frames: int = 12,                # 最大处理帧数（长视频提速，均匀抽样）
        **kwargs: Any,
    ):
        self.backend = backend
        self.model_name = model_name
        self.device = device
        self.max_frames = max_frames
        self.prompt_template = prompt_template or (
            "请用中文详细描述这张图片的内容，必须严格按下面三行格式输出，每行都不能省略：\n"
            "场景: <场景类型与内容描述>\n"
            "物体: <主要物体列表，用逗号分隔；没有则写无>\n"
            "文字: <画面中的文字原文；没有则写无>"
        )
        self.model = None
        self.processor = None
        self._model_error: Exception | None = None  # 缓存加载失败，避免逐帧重复下载
        self._model_kwargs = kwargs

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def extract_from_dir(
        self, keyframe_dir: str | Path, interval: float = 2.0
    ) -> VisualEvidence:
        """对关键帧目录进行批量视觉理解"""
        keyframe_dir = Path(keyframe_dir)
        if not keyframe_dir.exists():
            raise FileNotFoundError(f"关键帧目录不存在: {keyframe_dir}")

        frame_files = sorted(keyframe_dir.glob("*.jpg")) + \
                       sorted(keyframe_dir.glob("*.png")) + \
                       sorted(keyframe_dir.glob("*.jpeg"))

        if not frame_files:
            logger.warning("关键帧目录为空: %s", keyframe_dir)
            return VisualEvidence(model_name=self.model_name)

        if self.max_frames > 0 and len(frame_files) > self.max_frames:
            # 均匀抽样控制帧数上限（VLM/CLIP 长视频提速）
            total = len(frame_files)
            idxs = [round(i * (total - 1) / (self.max_frames - 1))
                    for i in range(self.max_frames)]
            frame_files = [frame_files[i] for i in sorted(set(idxs))]
            logger.info("视觉帧数 %d → 抽样 %d 帧（max_frames=%d）",
                        total, len(frame_files), self.max_frames)

        evidence = VisualEvidence(model_name=self.model_name)
        keyframes: list[KeyFrameDescription] = []

        bar = ProgressBar(total=len(frame_files), label="视觉")
        for i, fpath in enumerate(frame_files):
            timestamp = i * interval
            bar.update(i + 1, detail=fpath.name)
            try:
                kf = self._describe_frame(fpath, frame_index=i, timestamp=timestamp)
                keyframes.append(kf)
            except Exception as e:
                logger.warning("视觉理解 帧 %s 失败: %s", fpath.name, e)
        bar.finish()

        evidence.keyframes = keyframes
        # 反映实际使用的后端（VLM 失败时可能已自动降级为 CLIP）
        evidence.model_name = (
            "CLIP (ViT-B/32)" if self.backend == "clip" else self.model_name
        )
        evidence.overall_summary = self._generate_summary(keyframes)

        logger.info(
            "视觉理解完成: %d 帧描述生成",
            len(keyframes),
        )
        return evidence

    def extract_single(self, image_path: str | Path, frame_index: int = 0,
                       timestamp: float = 0.0) -> KeyFrameDescription:
        """对单张图片进行视觉理解"""
        return self._describe_frame(Path(image_path), frame_index, timestamp)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _describe_frame(self, image_path: Path, frame_index: int,
                        timestamp: float) -> KeyFrameDescription:
        if self.backend == "vlm":
            try:
                return self._vlm_describe(image_path, frame_index, timestamp)
            except Exception as e:
                # VLM 不可用（缺依赖/下载失败等）时自动降级为 CLIP
                logger.warning("VLM 后端不可用（%s），自动降级为 CLIP", e)
                self.backend = "clip"
        if self.backend == "clip":
            return self._clip_describe(image_path, frame_index, timestamp)
        raise ValueError(f"不支持的视觉后端: {self.backend}")

    # ------------------------------------------------------------------
    # VLM 方案
    # ------------------------------------------------------------------

    def _vlm_describe(self, image_path: Path, frame_index: int,
                      timestamp: float) -> KeyFrameDescription:
        """使用 InternVL2 / Qwen2-VL 等 VLM 描述画面"""
        model, processor, tokenizer = self._load_vlm()

        import torch

        # 尝试 InternVL2 接口
        try:
            # InternVL2 风格：chat(tokenizer, pixel_values, question)
            # pixel_values 形状 [图块数, 3, 448, 448]，单块即可
            if hasattr(model, "chat"):
                import torch
                from PIL import Image

                image = Image.open(image_path).convert("RGB")
                image = image.resize((448, 448), Image.BICUBIC)
                arr = np.asarray(image, dtype=np.float32) / 255.0
                arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array(
                    [0.229, 0.224, 0.225]
                )
                pixel_values = torch.from_numpy(
                    arr.transpose(2, 0, 1)
                ).unsqueeze(0).to(next(model.parameters()).dtype)
                question = "<image>\n" + self.prompt_template
                response = model.chat(
                    tokenizer,
                    pixel_values,
                    question,
                    generation_config={
                        "max_new_tokens": 200,
                        "do_sample": False,
                        "repetition_penalty": 1.15,
                    },
                )
                raw_description = response
            elif hasattr(processor, "__call__"):
                # Qwen2-VL / 通用 HF VLM 风格
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {"type": "text", "text": self.prompt_template},
                        ],
                    }
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(
                    text=text, images=image, return_tensors="pt"
                ).to(model.device)

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=256)
                generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
                raw_description = processor.decode(
                    generated_ids[0], skip_special_tokens=True
                )
            else:
                raw_description = f"[VLM 调用失败: 不支持的模型接口]"

        except Exception as e:
            logger.warning("VLM 推理失败: %s，降级为占位描述", e)
            raw_description = f"[视觉理解失败: {e}]"

        # 输出清洗：小模型易重复罗列物体，去重并截断
        for marker in ("物体:", "物体："):
            if marker in raw_description:
                head, _, tail = raw_description.partition(marker)
                first_line, _, rest = tail.partition("\n")
                items = [
                    x.strip()
                    for x in first_line.replace("、", ",").split(",")
                    if x.strip() and x.strip() != "无"
                ]
                items = list(dict.fromkeys(items))[:15]
                raw_description = (
                    f"{head}{marker} {', '.join(items) if items else '无'}"
                    f"\n{rest}"
                )
                break

        return self._parse_description(
            raw_description, frame_index, timestamp
        )

    def _resolve_model_path(self) -> str:
        """解析 VLM 模型路径：本地目录 > 项目内 models/<模型名> > HF 模型 ID"""
        p = Path(self.model_name)
        if p.is_dir():
            return str(p.resolve())
        root = Path(__file__).resolve().parent.parent
        # 项目内按完整名（models/OpenGVLab/InternVL2-2B）或末段名（models/InternVL2-2B）查找
        for candidate in (
            root / self.model_name,
            root / "models" / Path(self.model_name).name,
        ):
            if candidate.is_dir():
                return str(candidate.resolve())
        return self.model_name

    @staticmethod
    def _ascii_tokenizer_dir(model_path: str) -> str:
        """Windows 下 sentencepiece 的 C++ 加载器无法打开中文路径，
        把分词相关的小文件复制到 ASCII 临时目录，返回该目录（原路径为 ASCII 时原样返回）。"""
        if os.name != "nt" or model_path.isascii():
            return model_path
        import shutil
        import tempfile
        import time as _time

        src_dir = Path(model_path)
        tmp_dir = Path(tempfile.gettempdir()) / "iv2_tokenizer"
        small_files = (
            "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
            "added_tokens.json", "config.json", "generation_config.json",
            "preprocessor_config.json", "conversation.py",
            "configuration_internlm2.py", "configuration_internvl_chat.py",
            "configuration_intern_vit.py", "modeling_internlm2.py",
            "modeling_internvl_chat.py", "modeling_intern_vit.py",
            "tokenization_internlm2.py", "tokenization_internlm2_fast.py",
        )
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                tmp_dir.mkdir(exist_ok=True)
                for name in small_files:
                    src = src_dir / name
                    if src.exists():
                        shutil.copy2(src, tmp_dir / name)
                return str(tmp_dir)
            except Exception as e:  # 同步盘/杀软短暂锁文件时重试
                last_err = e
                _time.sleep(3)
        logger.warning("复制分词文件到临时目录失败: %s，回退原路径", last_err)
        return model_path

    def _load_vlm(self):
        """延迟加载 VLM 模型"""
        if self.model is not None:
            return self.model, self.processor, self._tokenizer
        if self._model_error is not None:
            raise self._model_error

        try:
            import torch
            from transformers import AutoModel, AutoProcessor, AutoTokenizer

            model_path = self._resolve_model_path()
            # 权重用原路径（Python IO 支持中文路径）；分词器用 ASCII 副本目录
            # （sentencepiece C++ 无法打开中文路径）
            tokenizer_path = self._ascii_tokenizer_dir(model_path)
            logger.info("加载 VLM 模型: %s —— 首次运行需下载大模型（数 GB），请耐心等待...", model_path)
            device_map = self.device
            if device_map == "auto":
                device_map = "cuda" if torch.cuda.is_available() else "cpu"

            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if "cuda" in str(device_map) else torch.float32,
                device_map=device_map,
                trust_remote_code=True,
                **self._model_kwargs,
            )
            try:
                self.processor = AutoProcessor.from_pretrained(
                    tokenizer_path, trust_remote_code=True
                )
            except Exception:
                self.processor = None

            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True
            )
        except Exception as e:
            self._model_error = e
            raise

        return self.model, self.processor, self._tokenizer

    # ------------------------------------------------------------------
    # CLIP 备选方案（轻量，不依赖 VLM）
    # ------------------------------------------------------------------

    def _clip_describe(self, image_path: Path, frame_index: int,
                       timestamp: float) -> KeyFrameDescription:
        """使用 CLIP 零样本分类 + YOLO 检测"""
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        # 场景分类候选
        scene_candidates = [
            "厨房", "餐厅", "户外", "办公室", "舞台", "录音棚",
            "教室", "街道", "自然风景", "人物特写", "文字页面",
            "手写笔记", "乐谱", "菜谱", "产品展示",
            "美食食物特写", "演唱会现场",
        ]
        scene_type = self._clip_classify(image, scene_candidates)

        # 物体检测（YOLO）
        objects = self._detect_objects(image)

        # 美食情景 → 具体菜品候选识别（CLIP 零样本 top-k）
        food_hits: list[str] = []
        if scene_type in FOOD_SCENES:
            try:
                topk = self._clip_classify_topk(
                    image, list(FOOD_CANDIDATES.keys()), k=5
                )
                for en_label, score in topk:
                    if score >= 0.12:
                        zh = FOOD_CANDIDATES[en_label]
                        food_hits.append(f"{zh}({score:.2f})")
                        if zh not in objects:
                            objects.append(zh)
            except Exception as e:
                logger.debug("菜品候选识别失败: %s", e)

        # OCR（如果有现成的 OCR 文字）
        text_in_frame = self._find_text(image)

        desc_parts = [f"场景: {scene_type}"]
        if objects:
            desc_parts.append(f"物体: {', '.join(objects)}")
        if food_hits:
            desc_parts.append(f"候选菜品: {', '.join(food_hits)}")
        description = ". ".join(desc_parts)

        return KeyFrameDescription(
            frame_index=frame_index,
            timestamp=timestamp,
            description=description,
            objects=objects,
            scene_type=scene_type,
            text_in_frame=text_in_frame,
            confidence=0.8,
        )

    def _clip_classify(self, image: Any, candidates: list[str]) -> str:
        """CLIP 零样本分类（返回最优候选）"""
        topk = self._clip_classify_topk(image, candidates, k=1)
        return topk[0][0] if topk else "未知场景"

    def _clip_classify_topk(
        self, image: Any, candidates: list[str], k: int = 3
    ) -> list[tuple[str, float]]:
        """CLIP 零样本分类，返回 top-k（标签, 得分）"""
        try:
            import clip
            import torch

            model, preprocess = self._get_clip_model()
            image_input = preprocess(image).unsqueeze(0)

            device = next(model.parameters()).device
            image_input = image_input.to(device)

            text_inputs = torch.cat([
                clip.tokenize(f"a photo of {c}") for c in candidates
            ]).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            scores = similarity[0].tolist()
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            return ranked[:k]
        except Exception:
            return []

    def _get_clip_model(self):
        if self.model is not None:
            return self.model, self._clip_preprocess

        import clip
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("加载 CLIP 模型 (ViT-B/32)")
        # 模型统一放项目内 models/clip
        clip_dir = Path(__file__).resolve().parent.parent / "models" / "clip"
        clip_dir.mkdir(parents=True, exist_ok=True)
        model, preprocess = clip.load(
            "ViT-B/32", device=device, download_root=str(clip_dir)
        )
        self.model = model
        self._clip_preprocess = preprocess
        return model, preprocess

    def _detect_objects(self, image: Any) -> list[str]:
        """YOLO 物体检测（备选）"""
        try:
            import torch
            model = self._get_yolo_model()
            results = model(image)
            objects = []
            for r in results:
                for c in r.boxes.cls:
                    name = model.names[int(c)]
                    if name not in objects:
                        objects.append(name)
            return objects[:10]
        except Exception:
            return []

    def _get_yolo_model(self):
        if hasattr(self, "_yolo_model"):
            return self._yolo_model
        from ultralytics import YOLO
        logger.info("加载 YOLOv8n 模型")
        self._yolo_model = YOLO("yolov8n.pt")
        return self._yolo_model

    def _find_text(self, image: Any) -> str:
        """简单文字发现（依赖 OCR 模块已有结果，此处留空）"""
        return ""

    # ------------------------------------------------------------------
    # 描述解析 & 摘要
    # ------------------------------------------------------------------

    def _parse_description(self, raw: str, frame_index: int,
                           timestamp: float) -> KeyFrameDescription:
        """从 VLM 原始回复中解析结构化字段"""
        lines = raw.strip().split("\n")
        scene_type = ""
        objects: list[str] = []
        text_in_frame = ""

        for line in lines:
            line = line.strip()
            if line.startswith("场景") or line.startswith("场景:"):
                scene_type = line.split(":", 1)[-1].strip() if ":" in line else line.split("场景", 1)[-1].strip()
            elif line.startswith("物体") or line.startswith("物体:"):
                obj_str = line.split(":", 1)[-1].strip() if ":" in line else line.split("物体", 1)[-1].strip()
                objects = [o.strip() for o in obj_str.replace("、", ",").split(",") if o.strip()]
                objects = list(dict.fromkeys(objects))[:15]
            elif line.startswith("文字") or line.startswith("文字:"):
                text_in_frame = line.split(":", 1)[-1].strip() if ":" in line else line.split("文字", 1)[-1].strip()

        return KeyFrameDescription(
            frame_index=frame_index,
            timestamp=timestamp,
            description=raw,
            objects=objects,
            scene_type=scene_type,
            text_in_frame=text_in_frame if text_in_frame not in ("无", "无。", "") else "",
            confidence=0.85,
        )

    def _generate_summary(self, keyframes: list[KeyFrameDescription]) -> str:
        """生成视频整体内容概要"""
        if not keyframes:
            return "无关键帧信息"

        # 统计场景
        scene_counts: dict[str, int] = {}
        for kf in keyframes:
            if kf.scene_type:
                scene_counts[kf.scene_type] = scene_counts.get(kf.scene_type, 0) + 1

        main_scene = max(scene_counts, key=scene_counts.get) if scene_counts else "未知"

        # 汇总物体
        all_objects: list[str] = []
        for kf in keyframes:
            all_objects.extend(kf.objects)
        top_objects = list(dict.fromkeys(all_objects))[:10]

        return (
            f"视频共 {len(keyframes)} 个关键帧。"
            f"主要场景: {main_scene}。"
            f"常见物体: {', '.join(top_objects) if top_objects else '无'}。"
        )
