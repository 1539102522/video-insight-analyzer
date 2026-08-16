#!/usr/bin/env python3
"""
第二层：LLM 综合分析 —— 命令行入口

用法:
    # 推荐：环境变量方式
    $env:LLM_API_KEY = "sk-xxx"
    python run_second_layer.py --evidence video_evidence.json

    # 或参数方式
    python run_second_layer.py --evidence video_evidence.json --api-key sk-xxx

    # 换其他 OpenAI 兼容服务（如通义千问/Kimi/GLM/OpenAI）
    python run_second_layer.py --evidence video_evidence.json --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --model qwen-plus
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# 确保能导入 second_layer 包
sys.path.insert(0, str(Path(__file__).resolve().parent))

from second_layer.llm_analyzer import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    analyze_evidence,
    check_relevance,
    ask_question,
    load_dotenv,
)


def main():
    parser = argparse.ArgumentParser(
        description="第二层：用大模型分析第一层证据 JSON",
    )
    parser.add_argument(
        "--evidence", "-e", type=str, default=None,
        help="第一层证据 JSON 路径（默认 <视频名>_evidence.json）",
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="输入视频路径（仅用于推导默认证据文件名）",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="分析结果输出路径（默认 <证据名>_analysis.json）",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="LLM API Key（或环境变量 LLM_API_KEY）",
    )
    parser.add_argument(
        "--base-url", type=str, default=None,
        help=f"OpenAI 兼容接口地址（默认 {DEFAULT_BASE_URL}，或环境变量 LLM_BASE_URL）",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"模型名（默认 {DEFAULT_MODEL}，或环境变量 LLM_MODEL）",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="采样温度，默认 0.3",
    )
    parser.add_argument(
        "--check", type=str, default=None,
        help="只判断视频是否与关键词相关（如 --check 美食），返回 是/不是",
    )
    parser.add_argument(
        "--ask", type=str, default=None,
        help="自定义问题（如 --ask 视频里的菜是什么菜系）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger("main")

    # 加载项目根目录 .env 配置（LLM_API_KEY / LLM_BASE_URL / LLM_MODEL）
    load_dotenv()

    # 确定证据文件
    if args.evidence:
        evidence_path = Path(args.evidence)
    elif args.input:
        stem = Path(args.input).stem
        evidence_path = Path("outputs") / (stem + "_evidence.json")
    else:
        evidence_path = Path("outputs") / "video_evidence.json"

    if not evidence_path.exists():
        print(f"错误: 证据文件不存在: {evidence_path}", file=sys.stderr)
        print("请先运行: python run_first_layer.py --input video.mp4", file=sys.stderr)
        sys.exit(1)

    # API 配置
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

    # 分析
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    logger.info("证据文件: %s", evidence_path)
    logger.info("LLM: %s @ %s", model, base_url)

    if args.check and args.ask:
        print("错误: --check 和 --ask 不能同时使用", file=sys.stderr)
        sys.exit(1)

    try:
        if args.check:
            result = check_relevance(
                evidence, args.check, api_key, base_url, model,
                temperature=0.0,
            )
        elif args.ask:
            result = ask_question(
                evidence, args.ask, api_key, base_url, model,
                temperature=args.temperature,
            )
        else:
            result = analyze_evidence(
                evidence,
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=args.temperature,
            )
    except Exception as e:
        logger.error("LLM 分析失败: %s", e)
        sys.exit(1)

    # 保存
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = evidence_path.with_name(evidence_path.stem + "_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("分析结果已保存到: %s", out_path)

    # 打印关键结论
    print("\n" + "=" * 50, file=sys.stderr)
    if args.check:
        print("判断结果:", file=sys.stderr)
        print(f"  「{result.get('keyword', '')}」相关? {result.get('answer', '')}", file=sys.stderr)
        print(f"  理由: {result.get('reason', '')}", file=sys.stderr)
    elif args.ask:
        print("问题:", result.get("question", ""), file=sys.stderr)
        print("答案:", result.get("answer", ""), file=sys.stderr)
        print("依据:", result.get("reason", ""), file=sys.stderr)
    else:
        print("LLM 视频分析:", file=sys.stderr)
        print(f"  概述: {result.get('summary', '')}", file=sys.stderr)
        print(f"  分类: {result.get('category', '')}", file=sys.stderr)
        print(f"  建议标题: {result.get('title_suggestion', '')}", file=sys.stderr)
        print(f"  原创性: {result.get('is_original', '')}", file=sys.stderr)
        print(f"  关键实体: {result.get('key_entities', [])}", file=sys.stderr)
        print(f"  风险: {result.get('risk_flags', [])}", file=sys.stderr)
    print("=" * 50, file=sys.stderr)


if __name__ == "__main__":
    main()
