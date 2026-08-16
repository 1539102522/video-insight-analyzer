#!/usr/bin/env python3
"""
第一层证据提取 —— 命令行入口

用法:
    python run_first_layer.py --input video.mp4

    python run_first_layer.py --input video.mp4 --no-visual --no-audio
    python run_first_layer.py --input video.mp4 --output result.json
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# 必须在导入 torch / easyocr / ctranslate2 等库之前设置。
# Windows 下多个库各自捆绑 libiomp5md.dll（Intel OpenMP 运行时），
# 重复初始化会触发 "OMP: Error #15" 并导致程序卡死。
# ---------------------------------------------------------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# hf-mirror 无法代理 HuggingFace 的 Xet(CAS) 存储，禁用 Xet 后
# faster-whisper 会改走普通 HTTP 下载，避免模型大文件 401 错误。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import argparse
import json
import logging
import sys
from pathlib import Path

# 确保能导入 first_layer 包
sys.path.insert(0, str(Path(__file__).resolve().parent))

from first_layer.pipeline import FirstLayerPipeline, extract_evidence
from first_layer.evidence_bundle import EvidenceBundle


def setup_logging(verbose: bool = True):
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def bundle_to_dict(bundle: EvidenceBundle) -> dict:
    """将 EvidenceBundle 转为可序列化的字典"""
    return bundle.model_dump(mode="json")


def main():
    parser = argparse.ArgumentParser(
        description="短视频第一层证据提取：ASR + OCR + 视觉理解 + 音频识别",
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="输入视频路径 (MP4/MOV)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="输出 JSON 路径（默认自动保存为 <视频名>_evidence.json）",
    )
    parser.add_argument(
        "--print-json", action="store_true",
        help="同时在终端打印完整 JSON",
    )
    parser.add_argument(
        "--keyframe-interval", type=float, default=2.0,
        help="关键帧间隔（秒），默认 2.0",
    )
    parser.add_argument(
        "--asr-model", type=str, default="medium",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="ASR 模型大小，默认 medium",
    )
    parser.add_argument(
        "--asr-device", type=str, default="auto",
        help="ASR 推理设备 (cpu/cuda/auto)",
    )
    parser.add_argument(
        "--asr-model-dir", type=str, default=None,
        help="ASR 本地模型目录（自备 faster-whisper 模型文件，跳过在线下载）",
    )
    parser.add_argument(
        "--no-hf-mirror", action="store_true",
        help="不使用 hf-mirror 镜像，直连官方 huggingface.co 下载 ASR 模型",
    )
    parser.add_argument(
        "--ocr-engine", type=str, default="easyocr",
        choices=["easyocr", "paddleocr"],
        help="OCR 引擎，默认 easyocr",
    )
    parser.add_argument(
        "--ocr-lang", type=str, default="ch",
        help="OCR 语言，默认 ch",
    )
    parser.add_argument(
        "--ocr-model-dir", type=str, default=None,
        help="OCR 本地模型目录（默认 ~/.EasyOCR/model）",
    )
    parser.add_argument(
        "--no-ocr-download", action="store_true",
        help="禁止 OCR 自动下载模型（配合 --ocr-model-dir 使用）",
    )
    parser.add_argument(
        "--work-dir", type=str, default=None,
        help="临时工作目录（存放提取的音频/关键帧）",
    )
    parser.add_argument(
        "--visual-backend", type=str, default="clip", choices=["clip", "vlm"],
        help="视觉理解后端：clip（轻量默认）/ vlm（InternVL2 等，需模型）",
    )
    parser.add_argument(
        "--vlm-model", type=str, default="OpenGVLab/InternVL2-2B",
        help="VLM 模型名或本地目录",
    )
    parser.add_argument(
        "--audio-backend", type=str, default="fingerprint",
        choices=["fingerprint", "shazamio", "dejavu"],
        help="音频歌曲识别后端，默认 fingerprint（本地指纹，无曲库时只报音乐片段）",
    )
    parser.add_argument(
        "--model-timeout", type=float, default=None,
        help="单个模型执行超时（秒），默认不超时。例如 --model-timeout 300",
    )
    parser.add_argument(
        "--no-asr", action="store_true",
        help="禁用 ASR",
    )
    parser.add_argument(
        "--no-ocr", action="store_true",
        help="禁用 OCR",
    )
    parser.add_argument(
        "--no-visual", action="store_true",
        help="禁用视觉理解",
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="禁用音频识别",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="减少日志输出",
    )

    args = parser.parse_args()

    # 校验输入
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 视频文件不存在: {args.input}", file=sys.stderr)
        sys.exit(1)

    setup_logging(not args.quiet)
    logger = logging.getLogger("main")

    logger.info("输入视频: %s", input_path)
    logger.info("配置: ASR=%s OCR=%s Visual=%s Audio=%s",
                not args.no_asr, not args.no_ocr,
                not args.no_visual, not args.no_audio)

    # 运行管线
    try:
        bundle = extract_evidence(
            video_path=str(input_path),
            keyframe_interval=args.keyframe_interval,
            asr_model=args.asr_model,
            asr_device=args.asr_device,
            asr_model_dir=args.asr_model_dir,
            asr_use_hf_mirror=not args.no_hf_mirror,
            ocr_engine=args.ocr_engine,
            ocr_lang=args.ocr_lang,
            ocr_model_dir=args.ocr_model_dir,
            ocr_download_enabled=not args.no_ocr_download,
            visual_backend=args.visual_backend,
            visual_model=args.vlm_model,
            audio_backend=args.audio_backend,
            enable_asr=not args.no_asr,
            enable_ocr=not args.no_ocr,
            enable_visual=not args.no_visual,
            enable_audio=not args.no_audio,
            model_timeout=args.model_timeout,
            verbose=not args.quiet,
        )
    except Exception as e:
        logger.error("管线执行失败: %s", e, exc_info=True)
        sys.exit(1)

    # 输出
    result_dict = bundle_to_dict(bundle)

    if args.output:
        output_path = Path(args.output)
    else:
        # 默认保存到项目 outputs/ 目录下 <视频名>_evidence.json
        output_path = Path("outputs") / (input_path.stem + "_evidence.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("结果已保存到: %s", output_path)
    if args.print_json:
        print(json.dumps(result_dict, ensure_ascii=False, indent=2))

    # 打印摘要
    print("\n" + "=" * 50, file=sys.stderr)
    print("证据提取摘要:", file=sys.stderr)
    print(f"  视频: {input_path.name}", file=sys.stderr)
    print(f"  时长: {bundle.video_duration:.1f}s", file=sys.stderr)
    print(f"  ASR: {'✓' if bundle.asr else '✗'} "
          f"({len(bundle.asr.full_text) if bundle.asr else 0} 字)", file=sys.stderr)
    print(f"  OCR: {'✓' if bundle.ocr else '✗'} "
          f"({len(bundle.ocr.segments) if bundle.ocr else 0} 区域)", file=sys.stderr)
    print(f"  视觉: {'✓' if bundle.visual else '✗'} "
          f"({len(bundle.visual.keyframes) if bundle.visual else 0} 帧)", file=sys.stderr)
    print(f"  音频: {'✓' if bundle.audio else '✗'} "
          f"({'有音乐' if bundle.audio and bundle.audio.has_music else '无音乐'})", file=sys.stderr)
    print(f"  分类提示: {bundle.category_hint.value}", file=sys.stderr)
    print(f"  错误数: {len(bundle.errors)}", file=sys.stderr)
    print("=" * 50, file=sys.stderr)


if __name__ == "__main__":
    main()
