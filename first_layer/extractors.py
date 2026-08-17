"""
第一层模型提取器注册表（可扩展）。

新增一种模型只需三步：
1. 写一个提取器类：提供 extract(audio_path) 或 extract_from_dir(dir, interval)，
   返回一个证据对象（pydantic 模型或普通对象均可）；
2. 在 first_layer/pipeline.py 顶部调用 register_extractor() 注册
   （填 id / 显示名 / 依赖输入 / 构造参数生成函数）；
3. 管线会自动并行调度它；Web 界面的模型选项（GET /api/models）也会自动出现，
   上传时可用 --disable-model 或前端勾选框开关它。

本模块只依赖标准库，保证后端 API 能轻量导入（不加载任何模型库）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ExtractorSpec:
    """一个可插拔的第一层提取器描述"""

    id: str                                   # 唯一 id（asr / ocr / visual / audio / ...）
    name: str                                 # 显示名
    input_key: str = "keyframes"              # 依赖的预处理产物：audio / keyframes / none
    result_field: str = ""                    # 写入 EvidenceBundle 的字段名（默认同 id）
    default_enabled: bool = True
    backends: list[str] = field(default_factory=list)  # 可选后端列表（供前端下拉）
    backend: str = ""                         # 当前后端（默认取 backends[0]）
    backend_labels: dict[str, str] = field(default_factory=dict)  # 后端 id -> 显示名（前端下拉展示用）

    # ---- 以下由 pipeline.py 注册时填充（避免本模块反向依赖具体提取器类）----
    cls: Any = None                           # 提取器类
    make_kwargs: Callable[[Any], dict[str, Any]] | None = None  # (pipeline) -> 构造参数
    run_audio: Callable[[Any, str], Any] | None = None          # 自定义执行：音频输入
    run_keyframes: Callable[[Any, str, float], Any] | None = None  # 自定义执行：关键帧输入


EXTRACTOR_REGISTRY: dict[str, ExtractorSpec] = {}


def register_extractor(spec: ExtractorSpec) -> None:
    """注册一个第一层提取器（扩展点）"""
    if not spec.result_field:
        spec.result_field = spec.id
    if not spec.backend and spec.backends:
        spec.backend = spec.backends[0]
    EXTRACTOR_REGISTRY[spec.id] = spec


def get_extractors() -> dict[str, ExtractorSpec]:
    """返回注册表副本（id -> spec）"""
    return dict(EXTRACTOR_REGISTRY)


def describe_registry() -> list[dict[str, Any]]:
    """纯元数据描述（不含类引用），供 Web 界面 GET /api/models 使用"""
    return [
        {
            "id": s.id,
            "name": s.name,
            "input_key": s.input_key,
            "default_enabled": s.default_enabled,
            "backends": s.backends,
            "backend": s.backend,
            "backend_labels": s.backend_labels,
        }
        for s in EXTRACTOR_REGISTRY.values()
    ]
