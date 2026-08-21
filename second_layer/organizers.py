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
from .prompts import get_organizer_prompt

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
# 注册表（扩展点）—— 整理器提示词已收口到 second_layer/prompts.py
# ---------------------------------------------------------------------------

ORGANIZER_REGISTRY: dict[str, dict[str, Any]] = {
    "歌曲": {"schema": SongResult},
    "美食": {"schema": RecipeResult},
    "美文": {"schema": ProseResult},
    "其他": {"schema": OtherResult},
}

_DEFAULT_CATEGORY = "其他"


def register_organizer(
    category: str, schema: Type[BaseModel], prompt: str | None = None
) -> None:
    """注册新的类别整理器（扩展点）。

    Example:
        class VlogResult(BaseModel): ...
        register_organizer("Vlog", VlogResult)

    注意：prompt 参数已废弃，整理器提示词统一从 prompts.py 读取（支持网页编辑）。
    """
    ORGANIZER_REGISTRY[category] = {"schema": schema}
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
            system_prompt=get_organizer_prompt(category),
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
