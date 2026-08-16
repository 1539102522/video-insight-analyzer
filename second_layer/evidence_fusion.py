"""
证据融合：对第一层证据做去重、冲突标记与信息充分性评分。

对应方案图第二层的"证据融合"模块：
- 消除重复（OCR 同时间戳重复文本、ASR 重复文本）
- 冲突标记（同一时间窗内语音与画面文字不一致）
- 判断信息是否充分（四类证据块齐不齐）
"""

from __future__ import annotations

import re
from typing import Any

logger_placeholder = None  # 保持模块零依赖


def _norm(s: str) -> str:
    """文本归一化：去空白与标点，用于重复/冲突比较"""
    return re.sub(
        r"[\s\u3000，。、！？,.!?;；:：'\"「」『』（）()【】\[\]\-—_·|~/~]",
        "",
        s,
    ).lower()


def _near(t1: float, t2: float, tol: float = 0.5) -> bool:
    return abs(t1 - t2) <= tol


def fuse_evidence(bundle: dict[str, Any]) -> dict[str, Any]:
    """输入 EvidenceBundle JSON 字典，返回融合后的字典（含 fusion 元数据）。

    原始 bundle 不被修改；融合后字典结构与原 bundle 兼容，
    可直接喂给 build_evidence_text / classify_evidence。
    """
    fused: dict[str, Any] = dict(bundle)
    removed: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    # ---- ASR 去重：相同文本只保留第一条 ----
    asr = bundle.get("asr") or {}
    segs = list(asr.get("segments") or [])
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for s in segs:
        key = _norm(s.get("text", ""))
        if not key:
            continue
        if key in seen:
            removed.append({"source": "asr", "reason": "重复文本", "item": s})
            continue
        seen.add(key)
        kept.append(s)
    if len(kept) != len(segs):
        fused["asr"] = {**asr, "segments": kept}

    # ---- OCR 去重：同一时间点（0.5s 桶）相同文本保留置信度最高的一条 ----
    ocr = bundle.get("ocr") or {}
    osegs = list(ocr.get("segments") or [])
    buckets: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for s in osegs:
        t = round(float(s.get("timestamp", 0)) * 2) / 2
        key = (t, _norm(s.get("text", "")))
        if not key[1]:
            continue
        buckets.setdefault(key, []).append(s)
    okept: list[dict[str, Any]] = []
    for items in buckets.values():
        best = max(items, key=lambda x: float(x.get("confidence", 0) or 0))
        okept.append(best)
        for other in items:
            if other is not best:
                removed.append(
                    {"source": "ocr", "reason": "同时间戳重复文本", "item": other}
                )
    okept.sort(key=lambda x: float(x.get("timestamp", 0)))
    if len(okept) != len(osegs):
        fused["ocr"] = {**ocr, "segments": okept}

    # ---- 冲突标记：同一时间窗内 ASR 与 OCR 文本明显不一致 ----
    for a in (fused.get("asr") or {}).get("segments", []):
        a_start = float(a.get("start_time", 0))
        a_text = _norm(a.get("text", ""))
        if not a_text:
            continue
        for o in (fused.get("ocr") or {}).get("segments", []):
            t = float(o.get("timestamp", 0))
            o_text = _norm(o.get("text", ""))
            if not o_text:
                continue
            if (
                _near(a_start, t, 1.5)
                and o_text != a_text
                and o_text not in a_text
                and a_text not in o_text
            ):
                conflicts.append(
                    {
                        "time": round(t, 1),
                        "asr": a.get("text", ""),
                        "ocr": o.get("text", ""),
                        "note": "同一时间窗内语音与画面文字不一致，以画面文字（OCR）为准",
                    }
                )
                break  # 每个 ASR 片段只记一条

    # ---- 信息充分性评分 ----
    audio_block = fused.get("audio") or {}
    has_asr = bool((fused.get("asr") or {}).get("segments"))
    has_ocr = bool((fused.get("ocr") or {}).get("segments"))
    has_visual = bool((fused.get("visual") or {}).get("keyframes"))
    has_audio = bool(audio_block.get("has_music")) or bool(audio_block.get("audio_duration"))
    present = sum([has_asr, has_ocr, has_visual, has_audio])
    missing: list[str] = []
    if not has_asr:
        missing.append("ASR（无语音文本）")
    if not has_ocr:
        missing.append("OCR（无画面文字）")
    if not has_visual:
        missing.append("视觉（无关键帧描述）")
    if not has_audio:
        missing.append("音频（无音频信息）")
    score = present / 4.0
    level = "充分" if score >= 0.75 else ("一般" if score >= 0.5 else "不足")

    fused["fusion"] = {
        "duplicates_removed": removed,
        "duplicate_count": len(removed),
        "conflicts": conflicts,
        "conflict_count": len(conflicts),
        "sufficiency": {
            "score": round(score, 2),
            "level": level,
            "present_blocks": present,
            "missing": missing,
        },
    }
    return fused
