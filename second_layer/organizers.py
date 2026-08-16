"""
类别专用整理器（可扩展注册表模式）。

三类主线 + 其他：
- 歌曲 → 歌名 / 歌手 / 版本 / 未知状态
- 美食 → 结构化菜谱（菜名 / 食材 / 步骤）
- 美文 → 可阅读文稿（原文 / 校订候选 / 作者）
- 其他 → 暂不深入处理

每条主要信息都带出处（provenance）：
    source = "视频"（证据原文中出现） / "网络"（知识推断，非视频证据） / "未知"
    evidence = 出处依据

扩展新类别只需两步：
1. 定义 Pydantic Schema（继承 BaseModel）
2. register_organizer("类别名", schema, 提示词)
   并在 llm_analyzer.CLASSIFY_SYSTEM_TEMPLATE 中加入该类别定义，
   编排器会自动调用对应整理器。
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Type

from pydantic import BaseModel, Field

from .llm_analyzer import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    build_evidence_text,
    call_llm,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 结构化 Schema
# ---------------------------------------------------------------------------


class ProvenanceEntry(BaseModel):
    """单条信息的出处标记"""
    source: Literal["视频", "网络", "未知"] = Field(default="未知",
        description="视频=证据原文中出现；网络=知识推断；未知=无法确认")
    evidence: str = Field(default="", description="出处依据")


class RecipeStep(BaseModel):
    step: int = Field(..., description="步骤序号")
    description: str = Field(..., description="步骤描述")
    timestamp: float | None = Field(default=None, description="对应视频秒数")


class RecipeResult(BaseModel):
    """美食：结构化菜谱"""
    dish_name: str = Field(default="", description="菜名")
    ingredients: list[str] = Field(default_factory=list, description="食材清单")
    steps: list[RecipeStep] = Field(default_factory=list, description="步骤")
    unknown_fields: list[str] = Field(default_factory=list,
        description="证据中无法确认的字段名（未知状态）")
    provenance: dict[str, ProvenanceEntry] = Field(default_factory=dict,
        description="每个字段的出处")
    notes: str = Field(default="", description="证据不足时的说明")


class SongResult(BaseModel):
    """歌曲：歌名 / 歌手 / 版本 / 未知状态"""
    song_name: str = Field(default="", description="歌名")
    artist: str = Field(default="", description="歌手")
    version: str = Field(default="", description="版本信息（原版/翻唱/remix等）")
    unknown_fields: list[str] = Field(default_factory=list,
        description="证据中无法确认的字段名（未知状态）")
    provenance: dict[str, ProvenanceEntry] = Field(default_factory=dict,
        description="每个字段的出处")
    notes: str = Field(default="", description="证据不足时的说明")


class ProseResult(BaseModel):
    """美文：可阅读文稿"""
    original_text: str = Field(default="", description="原文（逐字来自证据）")
    proofread_candidates: list[str] = Field(default_factory=list,
        description="校订候选（每条：错字/标点 → 校订后文本）")
    author: str = Field(default="", description="作者")
    unknown_fields: list[str] = Field(default_factory=list,
        description="证据中无法确认的字段名（未知状态）")
    provenance: dict[str, ProvenanceEntry] = Field(default_factory=dict,
        description="每个字段的出处")
    notes: str = Field(default="", description="证据不足时的说明")


class OtherResult(BaseModel):
    """其他：暂不深入处理"""
    summary: str = Field(default="", description="内容概述")
    notes: str = Field(default="该视频不属于歌曲/美食/美文三类，暂不深入处理")


# ---------------------------------------------------------------------------
# 整理器提示词（含出处标注规则）
# ---------------------------------------------------------------------------

_PROVENANCE_RULE = (
    "出处标注规则（必须遵守）：每条主要信息都要在 provenance 中标明出处——\n"
    "  - source=\"视频\"：信息在证据原文中直接出现（写明证据依据，如 OCR 识别到...）；\n"
    "  - source=\"网络\"：证据中没有直接出现，但根据常识/知识可以合理推断"
    "（写明推断依据，如 ASR 歌词'在這個世界多少人走下去'可辨认出是《稻香》）；\n"
    "  - source=\"未知\"：既无直接证据、也无法合理推断时才用。\n"
    "不要过度保守：只要根据证据内容能合理推断出答案，就给出推断结果并标 source=\"网络\"；\n"
    "禁止编造证据中没有、也无法推断的信息；完全无法确认的字段留空并写进 unknown_fields 和 notes。"
)

ORGANIZER_PROMPTS: dict[str, str] = {
    "歌曲": """你是音乐内容整理助手。请基于证据，提取视频中歌曲的歌名、歌手、版本信息。

要求：
1. 只输出一个合法 JSON 对象（不要 Markdown 代码块）：
{"song_name": "歌名", "artist": "歌手", "version": "版本信息",
 "unknown_fields": ["证据中无法确认的字段名"],
 "provenance": {"song_name": {"source": "视频|网络|未知", "evidence": "依据"},
                "artist": {...}, "version": {...}},
 "notes": "证据不足时的说明"}
2. """ + _PROVENANCE_RULE + """
3. 用中文回答。""",
    "美食": """你是美食内容整理助手。请基于证据，把视频中的美食信息整理成结构化菜谱。

要求：
1. 只输出一个合法 JSON 对象（不要 Markdown 代码块）：
{"dish_name": "菜名", "ingredients": ["食材1", "食材2"],
 "steps": [{"step": 1, "description": "步骤描述", "timestamp": 对应视频秒数或 null}],
 "unknown_fields": ["证据中无法确认的字段名"],
 "provenance": {"dish_name": {"source": "视频|网络|未知", "evidence": "依据"},
                "ingredients": {...}, "steps": {...}},
 "notes": "证据不足时的说明"}
2. 菜名候选规则：若视觉证据中出现"候选菜品"，从中挑选出现次数最多且置信度最高的作为菜名候选
   （例如 CLIP 视觉识别到'麻辣烫(0.57)'、'水煮鱼(0.55)'），dish_name 填该菜名，
   provenance 标 source="视频"，evidence 写"CLIP 视觉识别候选菜品 xxx(0.xx)"；
   若视频各时段展示多个不同菜品、无单一菜品主导（多菜品混剪），
   dish_name 填"多菜品混剪（A、B、C 等）"，provenance 标 source="视频"并在 notes 说明；
   仅当候选菜品互相矛盾或最高置信度低于 0.3 时，dish_name 才留空并标"未知"。
3. """ + _PROVENANCE_RULE + """
4. 用中文回答。""",
    "美文": """你是美文内容整理助手。请基于证据，把视频中的文字内容整理成可阅读文稿。

要求：
1. 只输出一个合法 JSON 对象（不要 Markdown 代码块）：
{"original_text": "原文（逐字来自证据，OCR/ASR 原文拼接）",
 "proofread_candidates": ["原文有误处 → 校订后文本"],
 "author": "作者（证据中出现才填）",
 "unknown_fields": ["证据中无法确认的字段名"],
 "provenance": {"original_text": {"source": "视频", "evidence": "依据"},
                "author": {"source": "视频|网络|未知", "evidence": "依据"}},
 "notes": "证据不足时的说明"}
2. """ + _PROVENANCE_RULE + """
3. 用中文回答。""",
    "其他": """你是短视频内容整理助手。请基于证据，给出该视频（不属于歌曲/美食/美文）的一句话概述。

要求：
1. 只输出一个合法 JSON 对象（不要 Markdown 代码块）：
{"summary": "视频内容一句话概述", "notes": "该视频不属于歌曲/美食/美文三类，暂不深入处理"}
2. 严格基于证据，禁止编造；用中文回答。""",
}

# ---------------------------------------------------------------------------
# 注册表（扩展点）
# ---------------------------------------------------------------------------

ORGANIZER_REGISTRY: dict[str, dict[str, Any]] = {
    "歌曲": {"schema": SongResult, "prompt": ORGANIZER_PROMPTS["歌曲"]},
    "美食": {"schema": RecipeResult, "prompt": ORGANIZER_PROMPTS["美食"]},
    "美文": {"schema": ProseResult, "prompt": ORGANIZER_PROMPTS["美文"]},
    "其他": {"schema": OtherResult, "prompt": ORGANIZER_PROMPTS["其他"]},
}

_DEFAULT_CATEGORY = "其他"


def register_organizer(
    category: str, schema: Type[BaseModel], prompt: str
) -> None:
    """注册新的类别整理器（扩展点）。

    Example:
        class VlogResult(BaseModel): ...
        register_organizer("Vlog", VlogResult, "你是Vlog整理助手……")
    """
    ORGANIZER_REGISTRY[category] = {"schema": schema, "prompt": prompt}
    logger.info("已注册整理器: %s -> %s", category, schema.__name__)


def supported_categories() -> list[str]:
    return list(ORGANIZER_REGISTRY.keys())


def organize_result(
    category: str,
    evidence: dict[str, Any],
    api_key: str,
    base_url: str | None = None,
    model: str | None = None,
    backend: str = "openai",
    temperature: float = 0.2,
    timeout: int = 120,
) -> dict[str, Any]:
    """按类别调用对应整理器（未注册类别自动落到"其他"），返回带 schema 名的结果。"""
    # None 会覆盖 call_llm 的默认参数，这里补齐
    if base_url is None:
        base_url = DEFAULT_BASE_URL
    if model is None:
        model = DEFAULT_MODEL
    spec = ORGANIZER_REGISTRY.get(category) or ORGANIZER_REGISTRY[_DEFAULT_CATEGORY]
    user_content = build_evidence_text(evidence) + f"\n\n【已判定类别】{category}"
    try:
        data = call_llm(
            user_content, api_key, base_url, model, temperature, timeout,
            system_prompt=spec["prompt"],
            backend=backend,
        )
    except Exception as e:
        logger.warning("整理器调用失败（类别=%s）: %s", category, e)
        return {"schema": "error", "error": str(e)}

    # Pydantic Schema 校验
    schema_model: Type[BaseModel] = spec["schema"]
    try:
        schema_model.model_validate(data)
        data["schema_valid"] = True
    except Exception as e:
        logger.warning("整理结果 Schema 校验失败（类别=%s）: %s", category, e)
        data["schema_valid"] = False
        data["schema_error"] = str(e)[:300]
    data["schema"] = category if category in ORGANIZER_REGISTRY else _DEFAULT_CATEGORY
    return data
