"""
第二层：LLM 综合分析（已解耦：按情况选择后端）
把第一层提取的结构化证据 JSON 交给大模型，产出视频内容深度分析。

支持两种后端（backend 参数切换）：
1. openai    —— 云端 OpenAI 兼容接口（DeepSeek / 通义千问 / Kimi / GLM 等），
                通过 base_url + api_key + model 配置；
2. hpc_qwen  —— 老师 HPC 上自部署的 Qwen3（Slurm + tmux + SSH 隧道 + Bearer 令牌），
                对应《Qwen3 HPC 部署与交互完整使用手册》第 6 章网页 API。
                通过 base_url（隧道地址）+ api_key（本次令牌）配置。
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


def load_dotenv(path: str | Path | None = None) -> None:
    """加载 .env 文件到环境变量（不覆盖已存在的值）。

    用法：把 LLM_API_KEY=sk-xxx 写入项目根目录的 .env 文件。
    代码分享给别人时，各自配置自己的 .env 即可，互不干扰，
    无需修改代码、也不依赖本机注册表。
    """
    if path is None:
        # 默认：项目根目录（second_layer 的上一级）
        path = Path(__file__).resolve().parent.parent / ".env"
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)

# 默认使用 DeepSeek（国内直连、OpenAI 兼容），可用环境变量切换
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

# ---- LLM 后端类型 ----
BACKEND_OPENAI = "openai"      # 云端 OpenAI 兼容 API（DeepSeek 等）
BACKEND_HPC_QWEN = "hpc_qwen"  # 老师 HPC 上自部署的 Qwen3（网页 API）

# ---- HPC Qwen3 网页 API 配置（对应老师部署手册第 6 章）----
DEFAULT_HPC_QWEN_URL = "http://127.0.0.1:8000"  # 本机 SSH 隧道端口，端口不同用 --base-url 覆盖
HPC_QWEN_MESSAGE_KEY = "message"  # /api/chat 请求体里消息的字段名；拿到 api_server.py 后按实际接口调整
HPC_QWEN_MAX_MESSAGE = 7800       # 手册：单条消息最大 8192 字符，留余量

# 提示词已收口到 second_layer/prompts.py（支持网页编辑 + 场景推荐 + 用户自定义覆盖）。
# 这里保留同名默认常量以兼容函数签名默认值，运行时请使用 get_* 系列函数读取最新配置。
from .prompts import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT as SYSTEM_PROMPT,
    DEFAULT_CHECK_TEMPLATE as CHECK_SYSTEM_TEMPLATE,
    DEFAULT_QUESTION_TEMPLATE as QUESTION_SYSTEM_TEMPLATE,
    DEFAULT_CLASSIFY_TEMPLATE as CLASSIFY_SYSTEM_TEMPLATE,
    get_system_prompt,
    get_check_template,
    get_question_template,
    get_classify_template,
)


# ---------------------------------------------------------------------------
# 证据 → 提示文本
# ---------------------------------------------------------------------------

def build_evidence_text(bundle: dict[str, Any]) -> str:
    """将 EvidenceBundle 的 JSON 转成适合 LLM 阅读的文本"""
    parts: list[str] = []

    duration = bundle.get("video_duration") or 0
    parts.append(f"视频时长: {float(duration):.1f}s")

    # ASR
    asr = bundle.get("asr") or {}
    asr_name = asr.get("model_name") or "ASR"
    if asr.get("segments"):
        segs = [
            f"  [{s['start_time']:.1f}s-{s['end_time']:.1f}s] {s['text']} "
            f"(置信度 {s.get('confidence', 0):.2f})"
            for s in asr["segments"]
        ]
        parts.append(f"【ASR 语音识别 · 模型 {asr_name}】\n" + "\n".join(segs))
    else:
        parts.append(
            f"【ASR 语音识别 · 模型 {asr_name}】\n"
            "  无（视频可能没有语音，或被 VAD 静音过滤）"
        )

    # OCR
    ocr = bundle.get("ocr") or {}
    ocr_name = ocr.get("model_name") or "OCR"
    if ocr.get("segments"):
        by_time: dict[float, list[str]] = defaultdict(list)
        for s in ocr["segments"]:
            by_time[round(float(s.get("timestamp", 0)), 1)].append(s["text"])
        lines = [
            f"  [{t:.1f}s] " + " | ".join(texts)
            for t, texts in sorted(by_time.items())
        ]
        parts.append(f"【OCR 画面文字 · 模型 {ocr_name}】\n" + "\n".join(lines))
    else:
        parts.append(f"【OCR 画面文字 · 模型 {ocr_name}】\n  无")

    # 视觉理解
    visual = bundle.get("visual") or {}
    visual_name = visual.get("model_name") or "视觉理解"
    if visual.get("keyframes"):
        lines = []
        for k in visual["keyframes"]:
            extra = []
            if k.get("scene_type"):
                extra.append(f"场景={k['scene_type']}")
            if k.get("objects"):
                extra.append("物体=" + ",".join(k["objects"]))
            if k.get("text_in_frame"):
                extra.append(f"画面文字={k['text_in_frame']}")
            suffix = "，".join(extra)
            lines.append(
                f"  [{float(k.get('timestamp', 0)):.1f}s] {k.get('description', '')}"
                + (f"（{suffix}）" if suffix else "")
            )
        parts.append(f"【视觉理解 · 模型 {visual_name}】\n" + "\n".join(lines))
    else:
        parts.append(f"【视觉理解 · 模型 {visual_name}】\n  无（未启用视觉模块）")

    # 音频识别
    audio = bundle.get("audio") or {}
    audio_name = audio.get("model_name") or "音频识别"
    if audio.get("songs"):
        songs = [
            f"  {s.get('title', '?')} - {s.get('artist', '')} "
            f"(置信度 {s.get('confidence', 0):.2f})"
            for s in audio["songs"]
        ]
        parts.append(
            f"【音频识别 · 模型 {audio_name}】\n识别到候选歌曲:\n" + "\n".join(songs)
        )
    elif audio.get("has_music"):
        segs = audio.get("music_segments") or []
        if segs:
            seg_text = "，".join(
                f"{s[0]:.1f}s-{s[1]:.1f}s"
                if isinstance(s, (list, tuple)) and len(s) >= 2 else str(s)
                for s in segs
            )
            line = f"检测到 {len(segs)} 段音乐（{seg_text}），指纹库未匹配到具体歌曲"
        else:
            line = "检测到音乐片段，指纹库未匹配到具体歌曲"
        parts.append(f"【音频识别 · 模型 {audio_name}】\n  {line}")
    else:
        parts.append(f"【音频识别 · 模型 {audio_name}】\n  未检测到音乐")

    return "\n\n".join(parts)


def collect_models(bundle: dict[str, Any]) -> list[str]:
    """列出本次分析实际产出证据的模型"""
    models: list[str] = []
    checks = [
        ("asr", "segments"),
        ("ocr", "segments"),
        ("visual", "keyframes"),
    ]
    for key, content_field in checks:
        block = bundle.get(key) or {}
        if not block.get(content_field):
            continue
        models.append(block.get("model_name") or key)
    audio = bundle.get("audio") or {}
    if audio.get("audio_duration"):
        models.append(audio.get("model_name") or "audio")
    return models


# ---------------------------------------------------------------------------
# LLM 调用
# ---------------------------------------------------------------------------

def parse_json_object(text: str) -> dict[str, Any] | None:
    """从 LLM 输出中提取 JSON 对象（容忍 Markdown 代码块和前后废话）"""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def call_openai_compatible(
    user_content: str,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    timeout: int = 120,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    """调用 OpenAI 兼容的 chat/completions 接口，要求返回 JSON"""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
    }

    last_content = ""
    # 先尝试 JSON 模式，接口不支持或解析失败时回退为普通文本模式
    for attempt, fmt in enumerate(({"type": "json_object"}, None)):
        if fmt is not None:
            payload["response_format"] = fmt
        else:
            payload.pop("response_format", None)

        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if fmt is not None and resp.status_code == 400:
            logger.warning("接口不支持 response_format=json_object，改用文本模式重试")
            continue
        resp.raise_for_status()

        content = resp.json()["choices"][0]["message"]["content"]
        last_content = content
        parsed = parse_json_object(content)
        if parsed is not None:
            return parsed
        if fmt is None:
            break
        logger.warning("JSON 模式返回内容无法解析，改用文本模式重试")

    raise ValueError(f"LLM 返回内容无法解析为 JSON: {last_content[:300]!r}")


def _extract_text_from_chunk(obj: Any) -> str:
    """尽力从 HPC 流式响应的一个 JSON 分片里提取文字（兼容多种字段名）"""
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return ""
    # OpenAI 风格：choices[0].delta/message.content
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            for part in ("delta", "message"):
                inner = first.get(part)
                if isinstance(inner, dict) and isinstance(inner.get("content"), str):
                    return inner["content"]
    for key in ("content", "text", "token", "delta", "answer", "message"):
        val = obj.get(key)
        if isinstance(val, str):
            return val
    return ""


def _post_hpc_chat(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: int,
) -> str:
    """POST /api/chat 并解析响应（兼容 SSE 流式 / JSON / 纯文本三种格式）"""
    resp = requests.post(
        url + "/api/chat", json=payload, headers=headers,
        timeout=timeout, stream=True,
    )
    if resp.status_code == 409:
        raise RuntimeError(
            "HPC Qwen 返回 409：模型正忙（同一时间只允许一个生成请求），请稍后重试"
        )
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "event-stream" in content_type or "text/stream" in content_type:
        # SSE：逐行解析 data: {...}
        parts: list[str] = []
        for raw_line in resp.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = raw_line.strip()
            if not line or line.startswith(":") or line.startswith("event:"):
                continue
            if line.startswith("data:"):
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    parts.append(data)
                    continue
                piece = _extract_text_from_chunk(obj)
                if piece:
                    parts.append(piece)
            else:
                # 非 SSE 前缀的逐行文本（部分简易服务直接推文本行）
                parts.append(line)
        return "".join(parts)

    raw = resp.text
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        text = _extract_text_from_chunk(obj)
        if text:
            return text
    return raw


def call_hpc_qwen(
    user_content: str,
    token: str,
    base_url: str = DEFAULT_HPC_QWEN_URL,
    temperature: float = 0.3,
    timeout: int = 120,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    """调用老师 HPC 上自部署 Qwen3 的网页 API（SSH 隧道 + Bearer 令牌）

    对应《Qwen3 HPC 部署与交互完整使用手册》第 6 章：
    - GET  /api/health  健康检查（确认作业/隧道/令牌）
    - POST /api/clear   清空服务器上下文（每次分析独立，不串历史）
    - POST /api/chat    流式生成（请求体字段名按 HPC_QWEN_MESSAGE_KEY 配置）

    注意：令牌每次启动作业都会变；作业最长 20 分钟；同一时间只允许一个请求。
    """
    url = base_url.rstrip("/")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # 1) 健康检查：确认作业运行中、隧道已建立、令牌有效
    try:
        resp = requests.get(url + "/api/health", headers=headers, timeout=15)
        resp.raise_for_status()
        health = resp.json()
        logger.info(
            "HPC Qwen 健康检查通过: 模型=%s 节点=%s",
            health.get("model", "?"), health.get("node", "?"),
        )
    except Exception as e:
        raise RuntimeError(
            "无法连接 HPC 上的 Qwen API，请逐项检查:\n"
            "  1) HPC 作业是否在运行（登录节点执行: tmux attach -t qwen-api）\n"
            "  2) 本地 SSH 隧道是否建立: ssh -N -L 8000:计算节点:8000 用户@登录机\n"
            "  3) 令牌是否为本次作业最新生成（.env 的 HPC_QWEN_TOKEN）\n"
            f"  4) 健康检查地址: {url}/api/health\n"
            f"原始错误: {e}"
        )

    # 2) 清空服务器历史上下文，保证每次分析互不干扰（老接口没有该路由则忽略）
    try:
        resp = requests.post(url + "/api/clear", headers=headers, timeout=15)
        if resp.status_code >= 400 and resp.status_code != 404:
            logger.warning("/api/clear 返回 %s（忽略，继续）", resp.status_code)
    except requests.RequestException as e:
        logger.warning("调用 /api/clear 失败（忽略）: %s", e)

    # 3) 发送消息：系统提示 + 证据合成一条（服务器自行维护上下文）
    message = f"{system_prompt}\n\n{user_content}"
    if len(message) > HPC_QWEN_MAX_MESSAGE:
        logger.warning(
            "证据文本过长（%d 字符），超过 HPC 单条消息上限，截断最早的部分",
            len(message),
        )
        message = "（证据较长，已截断最早的部分）\n" + message[-HPC_QWEN_MAX_MESSAGE:]
    payload = {HPC_QWEN_MESSAGE_KEY: message}

    full_text = _post_hpc_chat(url, headers, payload, timeout)
    parsed = parse_json_object(full_text)
    if parsed is not None:
        return parsed

    # 4) 一次补救：明确要求只输出 JSON（自部署模型不支持 response_format）
    retry_msg = (
        message + "\n\n【重要】请忽略示例中的说明文字，只输出一个合法的 JSON 对象，"
        "不要输出任何解释或其他文字。"
    )
    full_text = _post_hpc_chat(url, headers, {HPC_QWEN_MESSAGE_KEY: retry_msg}, timeout)
    parsed = parse_json_object(full_text)
    if parsed is None:
        raise ValueError(f"HPC Qwen 返回内容无法解析为 JSON: {full_text[:300]!r}")
    return parsed


def call_llm(
    user_content: str,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    timeout: int = 120,
    system_prompt: str = SYSTEM_PROMPT,
    backend: str = BACKEND_OPENAI,
) -> dict[str, Any]:
    """按 backend 分派：openai=云端 API（DeepSeek 等）；hpc_qwen=老师 HPC 自部署 Qwen3"""
    if backend == BACKEND_HPC_QWEN:
        # HPC 模式：api_key 位置传入的是本次作业的 Bearer 令牌
        return call_hpc_qwen(
            user_content, api_key, base_url=base_url,
            temperature=temperature, timeout=timeout,
            system_prompt=system_prompt,
        )
    if backend != BACKEND_OPENAI:
        raise ValueError(
            f"未知 LLM 后端: {backend!r}（可选 {BACKEND_OPENAI} / {BACKEND_HPC_QWEN}）"
        )
    return call_openai_compatible(
        user_content, api_key, base_url, model, temperature, timeout, system_prompt,
    )


def analyze_evidence(
    evidence: dict[str, Any],
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    timeout: int = 120,
    backend: str = BACKEND_OPENAI,
) -> dict[str, Any]:
    """对第一层证据做 LLM 综合分析"""
    user_content = build_evidence_text(evidence)
    logger.info("证据文本长度: %d 字符", len(user_content))
    logger.debug("发送给 LLM 的内容:\n%s", user_content)
    logger.info("调用 LLM [%s]: %s @ %s", backend, model, base_url)

    result = call_llm(
        user_content, api_key, base_url, model, temperature, timeout,
        system_prompt=get_system_prompt(),
        backend=backend,
    )

    # 附带视频元信息，方便追溯
    result.setdefault("video_path", evidence.get("video_path", ""))
    result.setdefault("video_duration", evidence.get("video_duration", 0))
    result.setdefault("models_used", collect_models(evidence))
    return result


def check_relevance(
    evidence: dict[str, Any],
    keyword: str,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    timeout: int = 120,
    backend: str = BACKEND_OPENAI,
) -> dict[str, Any]:
    """快速判断视频是否与关键词相关，只返回 是/不是"""
    user_content = build_evidence_text(evidence)
    system_prompt = get_check_template().format(keyword=keyword)
    logger.info("相关性判断: 「%s」", keyword)
    result = call_llm(
        user_content, api_key, base_url, model, temperature, timeout,
        system_prompt=system_prompt,
        backend=backend,
    )
    result.setdefault("keyword", keyword)
    result.setdefault("video_path", evidence.get("video_path", ""))
    result.setdefault("video_duration", evidence.get("video_duration", 0))
    result.setdefault("models_used", collect_models(evidence))
    return result


def ask_question(
    evidence: dict[str, Any],
    question: str,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    timeout: int = 120,
    backend: str = BACKEND_OPENAI,
) -> dict[str, Any]:
    """基于证据回答任意问题"""
    user_content = build_evidence_text(evidence) + "\n\n【用户问题】\n" + question
    logger.info("回答问题: %s", question)
    result = call_llm(
        user_content, api_key, base_url, model, temperature, timeout,
        system_prompt=get_question_template(),
        backend=backend,
    )
    result.setdefault("question", question)
    result.setdefault("video_path", evidence.get("video_path", ""))
    result.setdefault("video_duration", evidence.get("video_duration", 0))
    result.setdefault("models_used", collect_models(evidence))
    return result


def classify_evidence(
    evidence: dict[str, Any],
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    timeout: int = 120,
    backend: str = BACKEND_OPENAI,
) -> dict[str, Any]:
    """自动分类：歌曲 / 美食 / 美文 / 其他"""
    user_content = build_evidence_text(evidence)
    logger.info("自动分类: 歌曲/美食/美文/其他")
    result = call_llm(
        user_content, api_key, base_url, model, temperature, timeout,
        system_prompt=get_classify_template(),
        backend=backend,
    )
    result.setdefault("video_path", evidence.get("video_path", ""))
    result.setdefault("video_duration", evidence.get("video_duration", 0))
    result.setdefault("models_used", collect_models(evidence))
    return result
