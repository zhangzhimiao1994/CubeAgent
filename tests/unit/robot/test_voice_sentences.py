from __future__ import annotations

from agent_hub.robot.voice.sentences import split_speakable


def test_split_speakable_keeps_incomplete_sentence() -> None:
    spoken, rest = split_speakable("你好，我还在想")
    assert spoken == ""
    assert rest == "你好，我还在想"


def test_split_speakable_takes_complete_chinese_sentences() -> None:
    spoken, rest = split_speakable("你好。我在。下一句还没说完")
    assert spoken == "你好。我在。"
    assert rest == "下一句还没说完"


def test_split_speakable_accepts_latin_terminators() -> None:
    spoken, rest = split_speakable("Hello! More")
    assert spoken == "Hello!"
    assert rest == " More"
