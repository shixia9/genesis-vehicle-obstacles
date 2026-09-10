"""Small, deterministic language adapter for the visual/control demo.

This is deliberately not an LLM.  It keeps an English visual phrase when the
user already provides one and translates a few common Chinese demo words so a
single terminal sentence can reach YOLO-World.  The production LLM adapter
will replace this function while preserving the same prompt boundary.
"""

from __future__ import annotations

import re


_REPLACEMENTS = (
    ("黄色", "yellow"),
    ("绿色", "green"),
    ("红色", "red"),
    ("蓝色", "blue"),
    ("紫色", "purple"),
    ("橙色", "orange"),
    ("小车", "car"),
    ("汽车", "car"),
    ("车辆", "vehicle"),
    ("圆柱", "cylinder"),
    ("柱子", "pillar"),
    ("柱", "pillar"),
    ("平台", "platform"),
    ("箱子", "box"),
    ("方块", "block"),
    ("雕塑", "sculpture"),
    ("雕像", "sculpture"),
    ("物体", "object"),
)

_FILLER = (
    "行驶到",
    "驾驶到",
    "开到",
    "移动到",
    "去",
    "到",
    "附近",
    "旁边",
    "那里",
    "这里",
    "目标",
    "一个",
    "一個",
    "的",
    "the",
    "a",
    "an",
    "drive",
    "go",
    "navigate",
    "near",
    "to",
)


def instruction_to_visual_prompt(instruction: str) -> str:
    """Convert one demo instruction into a text prompt for YOLO-World.

    Unknown words are retained rather than mapped to a closed-set class.  The
    function is intentionally conservative: it provides a useful first demo,
    while relation parsing and multilingual paraphrasing remain LLM work.
    """

    text = str(instruction).strip()
    if not text:
        raise ValueError("instruction must not be empty")
    for source, target in _REPLACEMENTS:
        text = text.replace(source, f" {target} ")
    for filler in _FILLER:
        if filler.isascii():
            text = re.sub(rf"\b{re.escape(filler)}\b", " ", text, flags=re.IGNORECASE)
        else:
            text = text.replace(filler, " ")
    text = re.sub(r"[，。！？、,.!?;；:：()（）\[\]{}]", " ", text)
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text)
    if words:
        return " ".join(words).strip().lower()
    # Preserve an unknown non-English expression for a future multilingual
    # backend instead of silently guessing a known object.
    return str(instruction).strip()
