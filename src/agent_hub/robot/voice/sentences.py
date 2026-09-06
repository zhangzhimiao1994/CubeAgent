from __future__ import annotations

_TERMINATORS = frozenset("。！？!?；;\n")


def split_speakable(text: str) -> tuple[str, str]:
    last = -1
    for index, char in enumerate(text):
        if char in _TERMINATORS:
            last = index
    if last < 0:
        return "", text
    return text[: last + 1], text[last + 1 :]


__all__ = ["split_speakable"]
