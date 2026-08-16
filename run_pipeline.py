#!/usr/bin/env python3
"""
一键流水线：第一层证据提取 + 第二层 LLM 分析

用法:
    # 默认：AI 自动判断视频类别（歌曲/美食/美文/其他，三类主线深加工）
    python run_pipeline.py --input videos/视频.mp4 --no-visual --no-hf-mirror --no-ocr-download

    # 弹窗选一个类别，判断是否属于该类（是/不是）
    python run_pipeline.py --input videos/视频.mp4 --no-visual --no-hf-mirror --no-ocr-download --menu

    # 命令行直接判断是否相关（返回 是/不是）
    python run_pipeline.py --input videos/视频.mp4 --no-visual --no-hf-mirror --no-ocr-download --check 美食

    # 完整结构分析（概述/分类/标题/风险等）
    python run_pipeline.py --input videos/视频.mp4 --no-visual --no-hf-mirror --no-ocr-download --analyze

    # 自定义问题
    python run_pipeline.py --input videos/视频.mp4 --no-visual --no-hf-mirror --no-ocr-download --ask "视频里的菜是什么菜系？"

    # 只跑第一层，不调用 LLM
    python run_pipeline.py --input videos/视频.mp4 --no-visual --no-hf-mirror --no-ocr-download --evidence-only

    # 使用老师 HPC 上自部署的 Qwen3（先在 .env 配 HPC_QWEN_TOKEN，并建好 SSH 隧道）
    python run_pipeline.py --input videos/视频.mp4 --llm-backend hpc_qwen

输出文件（默认输出到 outputs/ 目录）:
    outputs/<视频名>_evidence.json           第一层证据
    outputs/<视频名>_evidence_analysis.json  第二层 LLM 分析（含融合/质检/整理）
    outputs/<视频名>_evidence_analysis.md    Markdown 报告
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# 确保能导入 first_layer / second_layer 包
sys.path.insert(0, str(Path(__file__).resolve().parent))

from first_layer.pipeline import extract_evidence
from first_layer.media_preprocessor import MediaPreprocessor
from second_layer.llm_analyzer import (
    BACKEND_HPC_QWEN,
    BACKEND_OPENAI,
    DEFAULT_BASE_URL,
    DEFAULT_HPC_QWEN_URL,
    DEFAULT_MODEL,
    analyze_evidence,
    check_relevance,
    ask_question,
    classify_evidence,
    load_dotenv,
)
from second_layer.orchestrator import (
    export_markdown,
    run_second_layer as run_second_layer_orchestrated,
)

# 可判断的类别（歌曲/美食/美文 三类主线，其他由 AI 自动归类）
CATEGORIES = ["歌曲", "美食", "美文"]


def ask_category_console() -> str | None:
    """控制台菜单选择类别"""
    choices = CATEGORIES + ["自动判断（让 AI 选）"]
    print("\n" + "=" * 40)
    print("请选择要判断的类别:")
    for i, c in enumerate(choices, 1):
        print(f"  {i}. {c}")
    print("  0. 取消")
    while True:
        try:
            raw = input("请输入序号: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw == "0":
            return None
        try:
            idx = int(raw)
            if 1 <= idx <= len(CATEGORIES):
                return CATEGORIES[idx - 1]
            if idx == len(CATEGORIES) + 1:
                return "auto"
        except ValueError:
            pass
        print("输入无效，请重新输入")


def ask_category_gui() -> str | None:
    """tkinter 弹窗选择类别，返回类别名 / 'auto' / None（取消）"""
    import tkinter as tk

    root = tk.Tk()
    root.title("选择判断类别")
    try:
        root.attributes("-topmost", True)
        root.resizable(False, False)
    except Exception:
        pass

    tk.Label(root, text="请选择要判断的类别：",
             font=("Microsoft YaHei", 12)).pack(anchor="w", padx=20, pady=(16, 8))

    var = tk.StringVar(value=CATEGORIES[0])
    for c in CATEGORIES:
        tk.Radiobutton(root, text=c, variable=var, value=c,
                       font=("Microsoft YaHei", 11)).pack(anchor="w", padx=30)
    tk.Radiobutton(root, text="自动判断（让 AI 选）", variable=var, value="auto",
                   font=("Microsoft YaHei", 11)).pack(anchor="w", padx=30, pady=(6, 0))

    result: dict[str, str | None] = {"value": None}

    def on_ok():
        result["value"] = var.get()
        root.destroy()

    def on_cancel():
        result["value"] = None
        root.destroy()

    frame = tk.Frame(root)
    frame.pack(pady=(14, 16))
    tk.Button(frame, text="开始判断", width=12, command=on_ok).pack(side="left", padx=8)
    tk.Button(frame, text="取消", width=12, command=on_cancel).pack(side="left", padx=8)

    root.update_idletasks()
    w, h = root.winfo_reqwidth(), root.winfo_reqheight()
    x = (root.winfo_screenwidth() - w) // 2
    y = max(0, (root.winfo_screenheight() - h) // 3)
    root.geometry(f"+{x}+{y}")
    root.mainloop()
    return result["value"]


def ask_category() -> str | None:
    """优先弹窗选择；tkinter 不可用时回退控制台菜单。返回 None 表示取消。"""
    try:
        import tkinter  # noqa: F401
        return ask_category_gui()
    except Exception:
        return ask_category_console()


def main():
    parser = argparse.ArgumentParser(
        description="一键流水线：第一层提取 + 第二层 LLM 分析",
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="输入视频路径（建议放 videos/ 目录）")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="JSON 输出目录（默认 outputs/）")

    # ---- 第一层参数（与 run_first_layer.py 一致）----
    parser.add_argument("--keyframe-interval", type=float, default=2.0)
    parser.add_argument("--asr-model", type=str, default="medium")
    parser.add_argument("--asr-beam", type=int, default=5, choices=[1, 2, 5],
                        help="ASR 解码束宽（1 最快，5 最准）")
    parser.add_argument("--asr-threads", type=int, default=4,
                        help="ASR CPU 线程数（默认 4，CPU 核数多可调大提速）")
    parser.add_argument("--asr-device", type=str, default="auto")
    parser.add_argument("--asr-model-dir", type=str, default=None)
    parser.add_argument("--no-hf-mirror", action="store_true",
                        help="不使用 hf-mirror 镜像，直连官方 huggingface.co")
    parser.add_argument("--ocr-engine", type=str, default="easyocr")
    parser.add_argument("--ocr-lang", type=str, default="ch")
    parser.add_argument("--ocr-max-side", type=int, default=0,
                        help="OCR 帧最长边限制（0=原图；如 1280 可提速 OCR）")
    parser.add_argument("--ocr-model-dir", type=str, default=None)
    parser.add_argument("--no-ocr-download", action="store_true",
                        help="禁止 OCR 联网下载模型（本地模型已就绪时使用）")
    parser.add_argument("--visual-backend", type=str, default="clip",
                        choices=["vlm", "clip"],
                        help="视觉理解后端：clip（轻量默认，~338MB 已就绪）；vlm（InternVL2-2B，最准但需下 5GB + einops/timm）")
    parser.add_argument("--vlm-model", type=str, default="OpenGVLab/InternVL2-2B",
                        help="VLM 模型名或目录（默认自动找项目内 models/InternVL2-2B）")
    parser.add_argument("--vlm-max-frames", type=int, default=12,
                        help="视觉理解最大帧数（均匀抽样，长视频提速，默认 12）")
    parser.add_argument("--no-asr", action="store_true")
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--no-visual", action="store_true")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--disable-model", type=str, default=None,
                        help="按模型 id 关闭（逗号分隔，如 asr,audio；可与前端选择联动）")
    parser.add_argument("--audio-backend", type=str, default="fingerprint",
                        choices=["fingerprint", "shazamio", "dejavu"],
                        help="音频歌曲识别后端：fingerprint=本地指纹（默认，无曲库时只报音乐片段）；shazamio=Shazam 在线识别")
    parser.add_argument("--max-duration-minutes", type=float, default=10.0,
                        help="视频时长上限（分钟），默认 10；超限直接报错，设 0 关闭检查")
    parser.add_argument("--model-timeout", type=float, default=None)
    parser.add_argument("--quiet", action="store_true")

    # ---- 第二层参数 ----
    parser.add_argument("--llm-backend", type=str, default=None,
                        choices=[BACKEND_OPENAI, BACKEND_HPC_QWEN],
                        help="LLM 后端：openai=云端 API（DeepSeek 等，默认，也可环境变量 LLM_BACKEND）；"
                             "hpc_qwen=老师 HPC 自部署 Qwen3（需 SSH 隧道 + 本次令牌）")
    parser.add_argument("--api-key", type=str, default=None,
                        help="openai 后端：API Key（或环境变量 LLM_API_KEY）；"
                             "hpc_qwen 后端：本次作业的 Bearer 令牌（或环境变量 HPC_QWEN_TOKEN）")
    parser.add_argument("--base-url", type=str, default=None,
                        help=f"openai 后端：接口地址（默认 {DEFAULT_BASE_URL}）；"
                             f"hpc_qwen 后端：隧道地址（默认 {DEFAULT_HPC_QWEN_URL}）")
    parser.add_argument("--model", type=str, default=None,
                        help=f"模型名（默认 {DEFAULT_MODEL}，hpc_qwen 后端忽略此参数）")
    parser.add_argument("--check", type=str, default=None,
                        help="只判断视频是否与关键词相关（如 --check 美食），返回 是/不是")
    parser.add_argument("--menu", action="store_true",
                        help="弹窗选择类别后判断是否属于该类（是/不是）")
    parser.add_argument("--ask", type=str, default=None,
                        help="自定义问题（如 --ask 视频里的菜是什么菜系）")
    parser.add_argument("--analyze", action="store_true",
                        help="完整结构分析（概述/分类/标题/风险等）")
    parser.add_argument("--evidence-only", action="store_true",
                        help="只跑第一层证据提取，不调用 LLM")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="第二层质检未通过时的最大修正重试次数（默认 2，仍不过则标记 unknown）")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="第二层 LLM 温度（默认 0 稳定；越高越随机）")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger("pipeline")

    # 加载项目根目录 .env 配置（LLM_API_KEY / LLM_BASE_URL / LLM_MODEL）
    load_dotenv()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 视频文件不存在: {args.input}", file=sys.stderr)
        sys.exit(1)

    # ---- 输入校验（对应方案图 "MP4/MOV · ≤10分钟"）----
    ext = input_path.suffix.lower()
    if ext not in (".mp4", ".mov", ".mkv", ".avi", ".flv", ".webm", ".ts"):
        print(f"错误: 不支持的视频格式 {ext!r}（支持 MP4/MOV 等常见格式）", file=sys.stderr)
        sys.exit(1)
    if ext not in (".mp4", ".mov"):
        print(f"提示: 格式 {ext} 不在推荐范围（MP4/MOV），仍尝试处理", file=sys.stderr)
    try:
        media_info = MediaPreprocessor().probe(input_path)
        if media_info.duration > 0 and args.max_duration_minutes > 0:
            limit = args.max_duration_minutes * 60
            if media_info.duration > limit:
                print(
                    f"错误: 视频时长 {media_info.duration / 60:.1f} 分钟超过上限 "
                    f"{args.max_duration_minutes} 分钟。\n"
                    "请先剪辑短视频，或用 --max-duration-minutes 调高上限（设 0 关闭检查）。",
                    file=sys.stderr,
                )
                sys.exit(1)
    except Exception as e:
        logger.warning("时长探测失败（跳过时长校验）: %s", e)

    # ---- 分析模式 ----
    if sum(bool(x) for x in (args.check, args.ask, args.analyze, args.menu)) > 1:
        print("错误: --check / --ask / --analyze / --menu 只能指定一个", file=sys.stderr)
        sys.exit(1)

    # --menu：弹窗选类别后判断（是/不是）；
    # 默认（不指定任何分析参数）→ AI 自动判断视频类别
    use_menu = args.menu and not args.evidence_only
    selected: str | None = None
    if use_menu:
        selected = ask_category()
        if selected is None:
            print("已取消判断", file=sys.stderr)
            return
        logger.info("选择的类别: %s", "自动判断" if selected == "auto" else selected)

    # 后端与凭据检查放在第一层之前：避免跑完一分钟证据提取才发现没配好
    if not args.evidence_only:
        # 后端选择：命令行 --llm-backend > 环境变量 LLM_BACKEND > 默认云端 API
        backend = args.llm_backend or os.environ.get("LLM_BACKEND") or BACKEND_OPENAI

        if backend == BACKEND_HPC_QWEN:
            # 老师 HPC 自部署 Qwen3：走 SSH 隧道 + Bearer 令牌（手册第 6 章）
            base_url = args.base_url or os.environ.get("HPC_QWEN_URL") or DEFAULT_HPC_QWEN_URL
            token = args.api_key or os.environ.get("HPC_QWEN_TOKEN")
            if not token:
                print(
                    "错误: 使用 hpc_qwen 后端需要本次作业的 Bearer 令牌。\n"
                    "请用 --api-key 传入令牌，或在 .env 中配置 HPC_QWEN_TOKEN=xxx。\n"
                    "（令牌在 HPC 上启动 api 后随机生成，每次作业都不同；\n"
                    "  并确认 SSH 隧道已建立: ssh -N -L 8000:计算节点:8000 用户@登录机）",
                    file=sys.stderr,
                )
                sys.exit(1)
            api_key = token
            model = args.model or DEFAULT_MODEL  # hpc 后端忽略 model，仅用于日志显示
        else:
            api_key = args.api_key or os.environ.get("LLM_API_KEY")
            if not api_key:
                print(
                    "错误: 未提供 LLM API Key。\n"
                    "请用 --api-key 传入，或设置环境变量 LLM_API_KEY，\n"
                    "或在项目根目录 .env 文件中配置 LLM_API_KEY=sk-xxx",
                    file=sys.stderr,
                )
                sys.exit(1)
            base_url = args.base_url or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL
            model = args.model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL

    # =====================================================================
    # 第一层：证据提取
    # =====================================================================
    # 通用开关：--disable-model id1,id2（与 --no-* 等价，便于前端/扩展模型使用）
    enable_map: dict[str, bool] = {}
    if args.disable_model:
        for sid in args.disable_model.split(","):
            sid = sid.strip()
            if sid:
                enable_map[sid] = False
    logger.info("=" * 60)
    logger.info("【第一层】证据提取: %s", input_path.name)
    logger.info("=" * 60)
    try:
        bundle = extract_evidence(
            video_path=str(input_path),
            keyframe_interval=args.keyframe_interval,
            asr_model=args.asr_model,
            asr_device=args.asr_device,
            asr_model_dir=args.asr_model_dir,
            asr_use_hf_mirror=not args.no_hf_mirror,
            asr_beam=args.asr_beam,
            asr_cpu_threads=args.asr_threads,
            ocr_engine=args.ocr_engine,
            ocr_lang=args.ocr_lang,
            ocr_model_dir=args.ocr_model_dir,
            ocr_download_enabled=not args.no_ocr_download,
            ocr_max_side=args.ocr_max_side,
            visual_backend=args.visual_backend,
            visual_model=args.vlm_model,
            vlm_max_frames=args.vlm_max_frames,
            audio_backend=args.audio_backend,
            enable_asr=not args.no_asr,
            enable_ocr=not args.no_ocr,
            enable_visual=not args.no_visual,
            enable_audio=not args.no_audio,
            enable_map=enable_map,
            model_timeout=args.model_timeout,
            verbose=not args.quiet,
        )
    except Exception as e:
        logger.error("第一层执行失败: %s", e, exc_info=True)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / (input_path.stem + "_evidence.json")
    evidence: dict = bundle.model_dump(mode="json")
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("证据已保存: %s", evidence_path)

    if args.evidence_only:
        print(f"\n证据文件: {evidence_path}", file=sys.stderr)
        return

    # =====================================================================
    # 第二层：LLM 分析
    # =====================================================================
    logger.info("=" * 60)
    logger.info("【第二层】LLM 分析 [%s]: %s @ %s", backend, model, base_url)
    logger.info("=" * 60)

    try:
        if args.check:
            result = check_relevance(
                evidence, args.check, api_key, base_url, model,
                temperature=0.0,
                backend=backend,
            )
        elif args.ask:
            result = ask_question(
                evidence, args.ask, api_key, base_url, model,
                temperature=0.3,
                backend=backend,
            )
        elif args.analyze:
            result = analyze_evidence(
                evidence, api_key=api_key, base_url=base_url, model=model,
                backend=backend,
            )
        elif use_menu:
            # 弹窗模式：自动判断 → 分类；否则 是/不是 判断
            if selected == "auto":
                # 自动判断走完整编排：融合 → 分类 → 质检重试 → 专用整理
                result = run_second_layer_orchestrated(
                    evidence, api_key, base_url, model, backend=backend,
                    temperature=args.temperature, max_retries=args.max_retries, organize=True,
                    timeout=int(args.model_timeout) if args.model_timeout else 120,
                )
            else:
                result = check_relevance(
                    evidence, selected, api_key, base_url, model,
                    temperature=0.0,
                    backend=backend,
                )
        else:
            # 默认：AI 自动判断视频类别（融合 → 分类 → 质检重试 → 专用整理）
            result = run_second_layer_orchestrated(
                evidence, api_key, base_url, model, backend=backend,
                temperature=args.temperature, max_retries=args.max_retries, organize=True,
                timeout=int(args.model_timeout) if args.model_timeout else 120,
            )
    except Exception as e:
        logger.error("LLM 分析失败: %s", e)
        sys.exit(1)

    out_path = evidence_path.with_name(evidence_path.stem + "_analysis.json")
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("分析结果已保存: %s", out_path)

    # 导出 Markdown 报告（对应方案图"导出 Markdown/JSON"）
    try:
        md_path = out_path.with_name(out_path.stem + ".md")
        md_path.write_text(export_markdown(result, evidence), encoding="utf-8")
        logger.info("Markdown 报告已保存: %s", md_path)
    except Exception as e:
        logger.warning("Markdown 导出失败（忽略）: %s", e)

    # ---- 打印结论 ----
    print("\n" + "=" * 50, file=sys.stderr)
    if args.check or (use_menu and selected != "auto"):
        keyword = args.check or selected
        print("判断结果:", file=sys.stderr)
        print(f"  「{keyword}」相关? {result.get('answer', '')}", file=sys.stderr)
        print(f"  理由: {result.get('reason', '')}", file=sys.stderr)
        print(f"  使用模型: {', '.join(result.get('models_used', []))}", file=sys.stderr)
    elif args.ask:
        print("问题:", result.get("question", ""), file=sys.stderr)
        print("答案:", result.get("answer", ""), file=sys.stderr)
        print("依据:", result.get("reason", ""), file=sys.stderr)
        print(f"  使用模型: {', '.join(result.get('models_used', []))}", file=sys.stderr)
    elif args.analyze:
        print("LLM 视频分析:", file=sys.stderr)
        print(f"  概述: {result.get('summary', '')}", file=sys.stderr)
        print(f"  分类: {result.get('category', '')}", file=sys.stderr)
        print(f"  建议标题: {result.get('title_suggestion', '')}", file=sys.stderr)
        print(f"  原创性: {result.get('is_original', '')}", file=sys.stderr)
        print(f"  关键实体: {result.get('key_entities', [])}", file=sys.stderr)
        print(f"  风险: {result.get('risk_flags', [])}", file=sys.stderr)
        print(f"  使用模型: {', '.join(result.get('models_used', []))}", file=sys.stderr)
    else:
        # 默认自动分类 / 菜单选了自动判断
        print("AI 自动分类结果:", file=sys.stderr)
        print(f"  类别: {result.get('category', '')}", file=sys.stderr)
        print(f"  理由: {result.get('reason', '')}", file=sys.stderr)
        fusion = result.get("fusion") or {}
        suff = fusion.get("sufficiency") or {}
        print(f"  证据充分性: {suff.get('level', '?')}（去重 {fusion.get('duplicate_count', 0)} 条，冲突 {fusion.get('conflict_count', 0)} 处）", file=sys.stderr)
        qc = result.get("qc") or {}
        print(f"  质量控制: {'通过' if qc.get('passed') else '未通过'}（尝试 {qc.get('attempts', 1)} 次）", file=sys.stderr)
        org = result.get("organized")
        if isinstance(org, dict) and org.get("schema"):
            print(f"  结构化整理: {org.get('schema')} {'✓ Schema 校验通过' if org.get('schema_valid') else '⚠ Schema 校验未通过'}", file=sys.stderr)
        print(f"  使用模型: {', '.join(result.get('models_used', []))}", file=sys.stderr)
    print("=" * 50, file=sys.stderr)


if __name__ == "__main__":
    main()
