"""
质量控制层：校验 LLM 输出的合法性、证据引用真实性与违规内容。

对应方案图第二层的"质量控制"模块：
- JSON 结构校验（category/reason 字段合法）
- 证据引用检查（reason 必须标明来源模型；引用文本必须真实存在于证据中）
- 违规内容检查（敏感词，作为风险标记而非一票否决）
- 不通过时给出可反馈给 LLM 的错误清单（供有限重试）
"""

from __future__ import annotations

import json
import re
from typing import Any

# 合法类别（与 CLASSIFY_SYSTEM_TEMPLATE 保持一致；unknown 为降级标记）
VALID_CATEGORIES = {"歌曲", "美食", "美文", "其他", "unknown"}

# 证据来源标记：reason 至少要提到其中一个，才算"标明来源"
EVIDENCE_MARKERS = ("ASR", "OCR", "CLIP", "视觉", "音频", "whisper", "easyocr", "VLM")

# 违规敏感词（示例级，可按需扩充；命中只标记风险，不阻断分析）
VIOLATION_WORDS = ("赌博", "色情", "诈骗", "枪支", "毒品", "暴力血腥")


def _norm_text(s: str) -> str:
    """去空白与标点，用于宽松的引文匹配（容忍简繁/标点差异）"""
    return re.sub(r"[\s\u3000，。、！？,.!?;；:：'\"「」『』（）()【】\[\]\-—_·|~/~]", "", s)


def validate_classify_result(
    result: dict[str, Any], evidence_text: str
) -> tuple[list[str], list[str]]:
    """校验分类结果，返回 (errors, warnings)。

    errors：硬性错误（会触发反馈重试）——类别非法 / reason 为空 / 未标明证据来源。
    warnings：软性提示（不阻断）——引文未逐字命中（可能是简繁或标点差异）。
    """
    errors: list[str] = []
    warnings: list[str] = []

    category = result.get("category")
    if not category or str(category) not in VALID_CATEGORIES:
        errors.append(
            f"category 非法: {category!r}（应为 歌曲/美食/美文/其他 之一）"
        )

    reason = (result.get("reason") or "").strip()
    if not reason:
        errors.append("reason 为空")
    else:
        if not any(m in reason for m in EVIDENCE_MARKERS):
            errors.append(
                "reason 未标明证据来源（应提到 ASR/OCR/视觉/音频 等证据块）"
            )
        # 引用真实性：非贪婪提取引号内文本，归一化后在证据中查找；
        # 失配只警告不阻断（简繁/标点差异会导致逐字匹配失败，不应误杀正确结果）
        quoted = re.findall(r"['「『\"](.{2,60}?)['」』\"]", reason)
        norm_evidence = _norm_text(evidence_text)
        missing: list[str] = []
        for q in quoted:
            nq = _norm_text(q)
            if len(nq) >= 2 and nq in norm_evidence:
                continue
            missing.append(q)
        if quoted and len(missing) == len(quoted):
            warnings.append(
                "reason 的引文未在证据中逐字找到（可能是简繁/标点差异）: "
                + "、".join(missing[:3])
            )

    return errors, warnings


def check_violation(text: str) -> list[str]:
    """违规敏感词检查，返回命中词列表"""
    hits: list[str] = []
    for w in VIOLATION_WORDS:
        if w in text:
            hits.append(w)
    return hits


def quality_control(result: dict[str, Any], evidence_text: str) -> dict[str, Any]:
    """质检入口：返回 {passed, errors, warnings, violations}"""
    errors, warnings = validate_classify_result(result, evidence_text)
    all_text = evidence_text + json.dumps(result, ensure_ascii=False)
    violations = check_violation(all_text)
    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "violations": violations,
    }
