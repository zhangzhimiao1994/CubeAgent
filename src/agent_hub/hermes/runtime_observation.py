"""Shared runtime-observation detection for Hermes+ boundaries."""

from __future__ import annotations

import re

_RUNTIME_OBSERVATION_RE = re.compile(
    r"^Run [a-z_]+ with mode=[^,]+, workflow=[^.\s]+"
    r"\.(?: Scheduler notices: .+)?$"
)
_LOCALIZED_RUNTIME_OBSERVATION_RE = re.compile(
    r"本次(?:对话学习|运行观察)记录了一个(?:成功经验|失败教训|中性观察|运行观察)："
    r".*工作流以 .+ 模式(?:成功完成|运行失败)"
)


def is_runtime_observation_lesson(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return _RUNTIME_OBSERVATION_RE.fullmatch(value.strip()) is not None


def is_runtime_observation_text(value: object) -> bool:
    if not isinstance(value, str):
        return False
    text = " ".join(value.split())
    return (
        is_runtime_observation_lesson(text)
        or _LOCALIZED_RUNTIME_OBSERVATION_RE.search(text) is not None
    )
