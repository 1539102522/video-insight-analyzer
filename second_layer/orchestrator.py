"""
第二层编排：证据融合 → LLM 分类 → 质量控制（有限重试/降级 unknown）→ 类别专用整理。

对应方案图第二层的完整流程（含"通过/不通过 → 返回修正"回路）。
"""

from __future__ import annotations

import logging
from typing import Any

from .evidence_fusion import fuse_evidence
from .llm_analyzer import (
    BACKEND_OPENAI,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    build_evidence_text,
    call_llm,
    classify_evidence,
    collect_models,
)
from .prompts import get_classify_template
from .organizers import organize_result
from .quality_control import quality_control

logger = logging.getLogger(__name__)


def _finalize(result: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    """补齐元信息字段（与 classify_evidence 一致）"""
    result.setdefault("video_path", evidence.get("video_path", ""))
    result.setdefault("video_duration", evidence.get("video_duration", 0))
    result.setdefault("models_used", collect_models(evidence))
    return result


def run_second_layer(
    evidence: dict[str, Any],
    api_key: str,
    base_url: str | None = None,
    model: str | None = None,
    backend: str = BACKEND_OPENAI,
    temperature: float = 0.0,
    timeout: int = 120,
    max_retries: int = 2,
    organize: bool = True,
) -> dict[str, Any]:
    """第二层完整编排，返回包含 fusion/qc/organized 的最终结果字典。

    流程：
    1. 证据融合（去重/冲突/充分性）
    2. LLM 自动分类
    3. 质量控制：不通过 → 把错误反馈给 LLM 重试（最多 max_retries 次）
       仍不通过 → 降级标记 category=unknown
    4. 按类别调用专用整理器，产出结构化结果
    """
    # 入口默认值补齐（None 会覆盖下游函数默认参数）
    if base_url is None:
        base_url = DEFAULT_BASE_URL
    if model is None:
        model = DEFAULT_MODEL

    # 1) 证据融合
    fused = fuse_evidence(evidence)
    evidence_text = build_evidence_text(fused)
    sufficiency = fused.get("fusion", {}).get("sufficiency", {})
    logger.info(
        "证据融合完成: 去重 %d 条, 冲突 %d 条, 充分性 %s(%.0f%%)",
        fused.get("fusion", {}).get("duplicate_count", 0),
        fused.get("fusion", {}).get("conflict_count", 0),
        sufficiency.get("level", "?"),
        float(sufficiency.get("score", 0)) * 100,
    )

    # 2) 首次分类
    result = classify_evidence(
        fused, api_key, base_url, model, temperature, timeout, backend=backend
    )
    qc = quality_control(result, evidence_text)
    attempts = 1
    qc_history: list[dict[str, Any]] = [
        {
            "attempt": attempts,
            "category": result.get("category"),
            "reason": result.get("reason"),
            "errors": list(qc["errors"]),
            "warnings": list(qc.get("warnings", [])),
            "passed": qc["passed"],
        }
    ]

    # 3) 质检 → 反馈重试回路
    while not qc["passed"] and attempts <= max_retries:
        logger.warning(
            "质检未通过（第 %d/%d 次）: %s", attempts, max_retries, qc["errors"]
        )
        feedback = "，".join(qc["errors"])
        user_content = (
            evidence_text
            + "\n\n【上次输出被质检驳回，请修正后重新输出 JSON】\n"
            + "问题：" + feedback
        )
        try:
            result = call_llm(
                user_content, api_key, base_url, model, temperature, timeout,
                system_prompt=get_classify_template(),
                backend=backend,
            )
        except Exception as e:
            logger.error("质检修正重试失败: %s", e)
            break
        result = _finalize(result, fused)
        qc = quality_control(result, evidence_text)
        attempts += 1
        qc_history.append(
            {
                "attempt": attempts,
                "category": result.get("category"),
                "reason": result.get("reason"),
                "errors": list(qc["errors"]),
                "warnings": list(qc.get("warnings", [])),
                "passed": qc["passed"],
            }
        )

    if not qc["passed"]:
        # 有限重试后仍不通过：降级为 unknown，保证系统不空转
        logger.warning("质检 %d 次仍未通过，降级标记为 unknown: %s", attempts, qc["errors"])
        result = _finalize(
            {
                "category": "unknown",
                "reason": "证据不足或多次质检未通过，无法确定类别",
            },
            fused,
        )
        qc = quality_control(result, evidence_text)

    # 4) 类别专用整理
    organized: dict[str, Any] | None = None
    category = result.get("category")
    if organize and category and category != "unknown":
        organized = organize_result(
            category, fused, api_key, base_url, model,
            backend=backend, temperature=0.2, timeout=timeout,
        )
    result["organized"] = organized
    result["fusion"] = fused.get("fusion", {})
    result["qc"] = {
        "passed": qc["passed"],
        "errors": qc["errors"],
        "warnings": qc.get("warnings", []),
        "violations": qc["violations"],
        "attempts": attempts,
        "downgraded_to_unknown": category == "unknown" and attempts > 1,
        "history": qc_history,
    }
    return result


# ---------------------------------------------------------------------------
# Markdown 导出（对应方案图"导出 Markdown / JSON"）
# ---------------------------------------------------------------------------

def export_markdown(result: dict[str, Any], evidence: dict[str, Any]) -> str:
    """把分析结果导出为 Markdown 报告文本"""
    lines: list[str] = []
    lines.append(f"# 视频分析报告：{result.get('video_path', '')}\n")
    lines.append(f"- 时长：{float(result.get('video_duration', 0)):.1f} 秒")
    lines.append(f"- 类别：**{result.get('category', 'unknown')}**")
    lines.append(f"- 使用模型：{', '.join(result.get('models_used', []) or ['无'])}\n")

    fusion = result.get("fusion") or {}
    suff = fusion.get("sufficiency") or {}
    lines.append("## 证据融合")
    lines.append(
        f"- 信息充分性：{suff.get('level', '?')}（{float(suff.get('score', 0)) * 100:.0f}%）"
    )
    if suff.get("missing"):
        lines.append(f"- 缺失证据：{', '.join(suff['missing'])}")
    if fusion.get("duplicate_count"):
        lines.append(f"- 去重：移除 {fusion['duplicate_count']} 条重复证据")
    if fusion.get("conflict_count"):
        lines.append(f"- 冲突：发现 {fusion['conflict_count']} 处语音与画面文字不一致\n")

    lines.append("## 分类理由")
    lines.append(f"> {result.get('reason', '')}\n")

    organized = result.get("organized")
    if isinstance(organized, dict) and organized.get("schema") not in (None, "error"):
        lines.append(f"## 结构化整理（schema: {organized.get('schema')}）")
        if organized.get("schema_valid") is False:
            lines.append(f"> ⚠ Schema 校验未通过：{organized.get('schema_error', '')}\n")
        for key in ("dish_name", "ingredients", "steps", "song_name", "artist",
                    "version", "original_text", "proofread_candidates", "author",
                    "summary"):
            val = organized.get(key)
            if val in (None, "", []):
                continue
            lines.append(f"### {key}")
            if isinstance(val, list):
                for item in val:
                    lines.append(
                        f"- {json_dump_compact(item) if isinstance(item, dict) else item}"
                    )
            else:
                lines.append(str(val))
        unknown = organized.get("unknown_fields")
        if unknown:
            lines.append(f"\n**未知（证据未确认）**：{', '.join(unknown)}")
        prov = organized.get("provenance") or {}
        if prov:
            lines.append("\n**信息来源出处**：")
            for field, entry in prov.items():
                if isinstance(entry, dict):
                    src = entry.get("source", "未知")
                    icon = "📹 视频" if src == "视频" else ("🌐 网络" if src == "网络" else "❓ 未知")
                    ev = entry.get("evidence", "")
                    lines.append(f"- {field}：{icon}" + (f"（{ev}）" if ev else ""))
        if organized.get("notes"):
            lines.append(f"\n**说明**：{organized['notes']}")

    qc = result.get("qc") or {}
    lines.append("\n## 质量控制")
    lines.append(f"- 通过：{'是' if qc.get('passed') else '否'}（共尝试 {qc.get('attempts', 1)} 次）")
    if qc.get("errors"):
        lines.append("- 错误：" + "；".join(qc["errors"]))
    if qc.get("warnings"):
        lines.append("- 提示：" + "；".join(qc["warnings"]))
    if qc.get("violations"):
        lines.append(f"- 违规风险词：{', '.join(qc['violations'])}")
    for h in qc.get("history") or []:
        mark = "✅ 通过" if h.get("passed") else "❌ 未通过"
        lines.append(
            f"\n### 第 {h.get('attempt', '?')} 次质检（{mark}）"
        )
        lines.append(f"- 输出类别：{h.get('category', '?')}")
        if h.get("reason"):
            lines.append(f"- 输出理由：{h['reason']}")
        if h.get("errors"):
            lines.append("- 发现问题：" + "；".join(h["errors"]))
            lines.append("- 处理：把问题反馈给 LLM，要求修正后重新输出")
        if h.get("warnings"):
            lines.append("- 提示：" + "；".join(h["warnings"]))
    if qc.get("downgraded_to_unknown"):
        lines.append("\n> ⚠ 多次质检未通过，已降级标记为 unknown")

    if evidence.get("errors"):
        lines.append("\n## 第一层模块错误")
        for e in evidence["errors"]:
            lines.append(f"- {json_dump_compact(e)}")

    return "\n".join(lines)


def json_dump_compact(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)
