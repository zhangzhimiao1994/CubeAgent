"""Production-control helpers for AI short-drama multimedia runs."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass

_CHARACTER_ID_RE = re.compile(r"\bCHAR_[A-Z0-9]{2,16}_[0-9]{3}\b")
_LOOK_ID_RE = re.compile(r"\bLOOK_[0-9]{3}\b")
_ROLE_NAME_RE = re.compile(
    r"(?P<role>女主|男主|女二|男二|反派|配角|主角|医生|护士|外卖员|店长|老板|总裁|学生|老师|母亲|父亲|警察|记者|导演|制片人)"
    r"[：:、，,\s]*(?P<name>[\u4e00-\u9fff]{2,4})"
)
_NAME_ROLE_RE = re.compile(
    r"(?:^|[\n|｜,，。；;:：\s])"
    r"(?:[*_`#>\-\d.、)\s]*)?"
    r"(?P<name>[\u4e00-\u9fff]{2,4})"
    r"(?:[*_`#>\s]*)?"
    r"[｜|,，、:：\s（(]{0,8}"
    r"(?P<role>女主|男主|女二|男二|反派|配角|主角|医生|护士|外卖员|店长|老板|总裁|学生|老师|母亲|父亲|警察|记者|导演|制片人)"
)
_ACTION_NAME_RE = re.compile(
    r"(?P<name>[\u4e00-\u9fff]{2,4})(?:穿|拿|握|说|走|站|进入|冲入|换成|换上|继续|发现|追查|回头|抬头)"
)
_SCENE_RE = re.compile(r"(?:EP\d+_SC\d+|第[一二三四五六七八九十\d]+场|SC\d+)[：:][^。]*")
_LOOK_SWITCH_TERMS = (
    "第二天",
    "次日",
    "几天后",
    "多年后",
    "回家",
    "到家",
    "换成",
    "换上",
    "晚宴",
    "睡衣",
    "居家服",
    "战斗",
    "受伤",
    "雨夜",
    "湿身",
    "伪装",
    "参加活动",
)
_CONTINUITY_TERMS = ("同一夜", "同一天", "继续", "随后", "紧接", "仍然")
_NAME_STOPWORDS = frozenset(
    {
        "第二天",
        "同一夜",
        "同一天",
        "白大褂",
        "居家服",
        "外卖箱",
        "第一场",
        "第二场",
        "第三场",
    }
)
_NAME_REJECT_TERMS = ("他", "她", "它", "继续", "第二", "同一", "换成", "回家")
_NAME_COSTUME_REJECT_TERMS = (
    "白大褂",
    "居家服",
    "冲锋",
    "西装",
    "长裙",
    "针织",
    "雨衣",
    "战斗",
    "黄色",
    "藏蓝",
    "浅灰",
    "低马尾",
    "外卖箱",
    "断佩",
    "银针",
    "急诊",
    "夜班",
    "执行",
    "资深",
    "公司",
)
_PRIMARY_ROLE_TERMS = frozenset(("女主", "男主", "女二", "男二", "反派", "配角", "主角"))
_PINYIN_INITIALS = {
    "林": "L",
    "小": "X",
    "满": "M",
    "苏": "S",
    "清": "Q",
    "月": "Y",
    "渊": "Y",
    "念": "N",
    "烬": "J",
    "赵": "Z",
    "天": "T",
    "霸": "B",
    "沈": "S",
    "墨": "M",
    "秦": "Q",
    "岚": "L",
    "母": "M",
    "晚": "W",
    "晴": "Q",
}


@dataclass(frozen=True, slots=True)
class CharacterIdentity:
    """Stable identity definition: who the character is, independent of costume."""

    character_id: str
    display_name: str
    role_type: str | None
    identity_prompt: str
    identity_traits: tuple[str, ...]
    forbidden_drift: tuple[str, ...]
    master_reference_artifact_ids: tuple[str, ...]
    embedding_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CharacterLook:
    """Changeable look/costume definition for one character."""

    look_id: str
    character_id: str
    name: str
    scene_applicability: tuple[str, ...]
    costume_traits: tuple[str, ...]
    accessories: tuple[str, ...]
    hair_makeup_variations: tuple[str, ...]
    forbidden_identity_changes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SceneCharacterState:
    scene_id: str
    character_id: str
    look_id: str
    continuity_reason: str
    inherited_from_scene_id: str | None


@dataclass(frozen=True, slots=True)
class RhythmBeat:
    start_ms: int
    duration_ms: int
    purpose: str


@dataclass(frozen=True, slots=True)
class ProductionDirection:
    project_id: str | None
    target_duration_seconds: int
    director_statement: str
    producer_constraints: tuple[str, ...]
    rhythm_beats: tuple[RhythmBeat, ...]
    continuity_rules: tuple[str, ...]
    qc_rules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IdentityValidationResult:
    character_id: str
    artifact_id: str
    identity_score: float | None
    threshold: float
    passed: bool
    issues: tuple[str, ...]
    provider: str


class DeterministicIdentityValidator:
    """Small deterministic validator used by tests and non-provider smoke checks."""

    def __init__(self, *, scores: Mapping[str, float]) -> None:
        self._scores = dict(scores)

    def validate(
        self,
        character_id: str,
        artifact_id: str,
        *,
        threshold: float,
    ) -> IdentityValidationResult:
        score = self._scores.get(artifact_id)
        if score is None:
            return IdentityValidationResult(
                character_id=character_id,
                artifact_id=artifact_id,
                identity_score=None,
                threshold=threshold,
                passed=False,
                issues=("未配置该资产的身份相似度分数，不能静默放行。",),
                provider="deterministic_not_configured",
            )
        passed = score >= threshold
        return IdentityValidationResult(
            character_id=character_id,
            artifact_id=artifact_id,
            identity_score=score,
            threshold=threshold,
            passed=passed,
            issues=()
            if passed
            else (f"身份相似度 {score:.2f} 低于阈值 {threshold:.2f}，疑似人物漂移。",),
            provider="deterministic",
        )


@dataclass(frozen=True, slots=True)
class ProductionPlan:
    character_identities: tuple[CharacterIdentity, ...]
    looks: tuple[CharacterLook, ...]
    scene_states: tuple[SceneCharacterState, ...]
    direction: ProductionDirection


def build_identity_lock_prompt(
    identity: CharacterIdentity,
    *,
    look: CharacterLook | None = None,
    pose: str | None = None,
    scene: str | None = None,
    shot: str | None = None,
) -> str:
    """Build the character-identity-first prompt section used by asset/video generation."""

    lines = [
        f"CHARACTER_ID: {identity.character_id}",
        f"CHARACTER_NAME: {identity.display_name}",
    ]
    if identity.role_type:
        lines.append(f"ROLE_TYPE: {identity.role_type}")
    lines.extend(
        [
            "IDENTITY LOCK:",
            (
                "保持 master identity reference 的同一张脸、脸型、五官比例、眼距、眼型、鼻型、"
                "嘴型、下颌线、肤色、年龄特征、发际线、基础发型逻辑、身体比例和可辨识特征。"
            ),
            f"身份描述：{identity.identity_prompt}",
            f"不可漂移特征：{_join_traits(identity.forbidden_drift)}",
        ]
    )
    if look is not None:
        lines.extend(
            [
                "LOOK / COSTUME:",
                f"LOOK_ID: {look.look_id}",
                f"LOOK_NAME: {look.name}",
                f"服装元素：{_join_traits(look.costume_traits)}",
                f"配饰：{_join_traits(look.accessories)}",
                f"妆发状态：{_join_traits(look.hair_makeup_variations)}",
                (
                    "只允许修改服装、鞋履、首饰、包、帽子、妆发状态、雨夜/战斗/工作等当前 Look 字段；"
                    "不得改脸、改年龄感、改身体比例、改肤色、改发际线或重设人物身份。"
                ),
                (
                    "如果使用服装参考图，只提取版型、颜色、材质、纹理、配饰和穿着方式；"
                    "不要继承服装参考图中的脸、发型、肤色、年龄、身材或模特身份。"
                ),
            ]
        )
        if look.forbidden_identity_changes:
            lines.append(f"Look 禁止项：{_join_traits(look.forbidden_identity_changes)}")
    if pose:
        lines.extend(("POSE:", pose.strip()))
    if scene:
        lines.extend(("SCENE:", scene.strip()))
    if shot:
        lines.extend(("SHOT:", shot.strip()))
    lines.extend(
        [
            "NEGATIVE_DRIFT_RULES:",
            "优先级：Character Identity / Face > Character Body > Hairstyle base > Costume / Look > Pose > Scene。",
            "禁止把上一场失败图继续编辑成下一场；每次换装都必须从原始 Identity Reference 出发。",
            "QC_EXPECTATIONS:",
            "输出后必须检查是否仍是同一 Character ID、当前 Look 是否正确、是否与其他角色撞脸。",
        ]
    )
    return "\n".join(lines)


def production_metadata_for_label(label: str, prompt: str) -> dict[str, str]:
    """Extract compact production metadata from a generated asset label and prompt."""

    metadata: dict[str, str] = {}
    match = _CHARACTER_ID_RE.search(prompt)
    if match is not None:
        metadata["character_id"] = match.group(0)
    look_match = _LOOK_ID_RE.search(prompt)
    if look_match is not None:
        metadata["look_id"] = look_match.group(0)
    category = production_category_for_label(label)
    if category is not None:
        metadata["production_category"] = category
    return metadata


def production_category_for_label(label: str) -> str | None:
    normalized = unicodedata.normalize("NFKC", label).casefold()
    if any(term in normalized for term in ("角色锁定", "角色资产", "character", "定妆")):
        return "character_identity"
    if any(term in normalized for term in ("服装", "妆造", "costume", "look")):
        return "character_look"
    if any(term in normalized for term in ("分镜", "storyboard")):
        return "storyboard"
    if any(term in normalized for term in ("动作", "pose", "action")):
        return "action"
    if any(term in normalized for term in ("特效", "vfx", "effect")):
        return "vfx"
    if any(term in normalized for term in ("场景", "scene")):
        return "scene"
    if any(term in normalized for term in ("道具", "prop")):
        return "prop"
    if any(term in normalized for term in ("镜头", "camera", "lens")):
        return "camera"
    if any(term in normalized for term in ("表演", "节奏", "style", "performance")):
        return "performance_style"
    if any(term in normalized for term in ("资产", "asset")):
        return "production_asset"
    return None


def build_production_plan(script_text: str, *, request_text: str = "") -> ProductionPlan:
    """Derive a compact production plan from script/request text.

    This is intentionally deterministic and conservative. It extracts obvious
    role/name pairs and continuity markers; ambiguous choices stay as defaults
    instead of inventing unsupported wardrobe changes.
    """

    text = unicodedata.normalize("NFKC", f"{request_text}\n{script_text}")
    identities = _extract_character_identities(text)
    looks: list[CharacterLook] = []
    for identity in identities:
        looks.append(_default_look(identity, index=1, name="基础造型"))
    scene_states = _resolve_scene_states(text, identities, looks)
    direction = ProductionDirection(
        project_id=None,
        target_duration_seconds=60,
        director_statement=(
            "按短剧导演和制片标准控制节奏：角色身份稳定、造型连续、资产干净、"
            "镜头服务剧情推进，避免把设定图画成剧照或海报。"
        ),
        producer_constraints=(
            "昂贵生成任务必须可断点恢复",
            "失败资产只重试对应 Character ID / Look ID / Shot",
            "资产先审核再进入分镜和视频",
        ),
        rhythm_beats=(
            RhythmBeat(start_ms=0, duration_ms=3000, purpose="Hook"),
            RhythmBeat(start_ms=3000, duration_ms=7000, purpose="建立人物和冲突"),
            RhythmBeat(start_ms=10000, duration_ms=25000, purpose="核心变化和动作推进"),
            RhythmBeat(start_ms=35000, duration_ms=17000, purpose="对抗、反转或情绪兑现"),
            RhythmBeat(start_ms=52000, duration_ms=8000, purpose="收束和下一集钩子"),
        ),
        continuity_rules=(
            "连续时间默认继承上一场 Look",
            "只有换装、回家、第二天、活动、受伤、雨夜/湿身、战斗等信号才切换 Look",
            "场景变化不等于人物身份变化",
        ),
        qc_rules=(
            "抽帧检查身份一致性",
            "检查服装/道具/伤痕/湿身等连续性",
            "检查黑帧、冻结、字幕遮挡、音画同步和节奏",
        ),
    )
    return ProductionPlan(
        character_identities=tuple(identities),
        looks=tuple(looks),
        scene_states=tuple(scene_states),
        direction=direction,
    )


def _join_traits(values: tuple[str, ...]) -> str:
    return "、".join(item.strip() for item in values if item.strip()) or "未指定"


def _extract_character_identities(text: str) -> list[CharacterIdentity]:
    seen: dict[str, str] = {}
    ordered: list[tuple[str, str, str]] = []
    for match in _ROLE_NAME_RE.finditer(text):
        role = match.group("role")
        name = match.group("name")
        if name in seen or _invalid_character_name(name):
            continue
        seen[name] = role
        snippet = _nearby_sentence(text, match.start("name"))
        ordered.append((name, role, snippet))
    for match in _NAME_ROLE_RE.finditer(text):
        role = match.group("role")
        name = match.group("name")
        if role not in _PRIMARY_ROLE_TERMS:
            continue
        if name in seen or _invalid_character_name(name):
            continue
        seen[name] = role
        snippet = _nearby_sentence(text, match.start("name"))
        ordered.append((name, role, snippet))
    if not ordered:
        for match in _ACTION_NAME_RE.finditer(text):
            name = match.group("name")
            if name in seen or name in _NAME_STOPWORDS or any(term in name for term in _NAME_REJECT_TERMS):
                continue
            seen[name] = "角色"
            snippet = _nearby_sentence(text, match.start())
            ordered.append((name, "角色", snippet))
    return [
        CharacterIdentity(
            character_id=_character_id(name, index + 1),
            display_name=name,
            role_type=role,
            identity_prompt=snippet or f"{role}{name}",
            identity_traits=_identity_traits_from_snippet(snippet, role=role, name=name),
            forbidden_drift=("换脸", "改变年龄感", "改变五官比例", "改变发际线", "与其他角色撞脸"),
            master_reference_artifact_ids=(),
            embedding_refs=(),
        )
        for index, (name, role, snippet) in enumerate(ordered)
    ]


def _invalid_character_name(name: str) -> bool:
    cleaned = name.strip()
    return (
        not cleaned
        or cleaned in _NAME_STOPWORDS
        or any(term in cleaned for term in _NAME_REJECT_TERMS)
        or any(term in cleaned for term in _NAME_COSTUME_REJECT_TERMS)
        or cleaned.endswith(("服", "衣", "箱", "针", "佩", "褂", "裙"))
    )


def _identity_traits_from_snippet(snippet: str, *, role: str, name: str) -> tuple[str, ...]:
    traits = [f"{role}{name}", "身份与脸部长期稳定"]
    for keyword in ("医生", "外卖员", "店长", "学生", "总裁", "修仙", "高武", "甜品店", "白大褂"):
        if keyword in snippet and keyword not in traits:
            traits.append(keyword)
    return tuple(traits[:6])


def _default_look(identity: CharacterIdentity, *, index: int, name: str) -> CharacterLook:
    return CharacterLook(
        look_id=f"LOOK_{index:03d}",
        character_id=identity.character_id,
        name=name,
        scene_applicability=(),
        costume_traits=_look_traits_from_identity(identity),
        accessories=(),
        hair_makeup_variations=("沿用身份参考发型逻辑",),
        forbidden_identity_changes=("不得改脸", "不得改变体态", "不得继承服装参考模特身份"),
    )


def _look_traits_from_identity(identity: CharacterIdentity) -> tuple[str, ...]:
    prompt = identity.identity_prompt
    traits: list[str] = []
    if "白大褂" in prompt or "医生" in prompt:
        traits.append("白大褂/医疗职业状态")
    if "外卖" in prompt:
        traits.append("外卖员工作状态")
    if "甜品" in prompt or "店长" in prompt:
        traits.append("甜品店工作服")
    for keyword in (
        "居家服",
        "睡衣",
        "晚宴服",
        "风衣",
        "冲锋衣",
        "西装",
        "长裙",
        "针织衫",
        "雨衣",
        "战斗服",
        "校服",
        "制服",
    ):
        if keyword in prompt and keyword not in traits:
            traits.append(keyword)
    return tuple(traits[:5]) or ("符合角色身份的基础服装",)


def _resolve_scene_states(
    text: str,
    identities: list[CharacterIdentity],
    looks: list[CharacterLook],
) -> list[SceneCharacterState]:
    if not identities:
        return []
    scenes = _SCENE_RE.findall(text)
    if not scenes:
        scenes = [text]
    last_look_by_character = {identity.character_id: "LOOK_001" for identity in identities}
    next_look_index_by_character = {identity.character_id: 2 for identity in identities}
    states: list[SceneCharacterState] = []
    last_scene_by_character: dict[str, str] = {}
    for index, scene in enumerate(scenes, start=1):
        scene_id = _scene_id(scene, index)
        scene_characters = _scene_characters(scene, identities)
        for identity in scene_characters:
            switch_reason = _look_switch_reason(scene)
            inherited_from = last_scene_by_character.get(identity.character_id)
            if switch_reason and not _is_continuity_scene(scene):
                look_id = f"LOOK_{next_look_index_by_character[identity.character_id]:03d}"
                next_look_index_by_character[identity.character_id] += 1
                last_look_by_character[identity.character_id] = look_id
                looks.append(
                    CharacterLook(
                        look_id=look_id,
                        character_id=identity.character_id,
                        name=switch_reason,
                        scene_applicability=(scene_id,),
                        costume_traits=(switch_reason,),
                        accessories=(),
                        hair_makeup_variations=("沿用身份参考发型逻辑",),
                        forbidden_identity_changes=("不得改脸", "不得改变年龄感", "不得继承服装参考模特身份"),
                    )
                )
                reason = f"切换造型：{switch_reason}"
            else:
                look_id = last_look_by_character[identity.character_id]
                reason = "连续时间继承上一场造型" if inherited_from else "首次出场使用基础造型"
            states.append(
                SceneCharacterState(
                    scene_id=scene_id,
                    character_id=identity.character_id,
                    look_id=look_id,
                    continuity_reason=reason,
                    inherited_from_scene_id=inherited_from,
                )
            )
            last_scene_by_character[identity.character_id] = scene_id
    return states


def _scene_characters(scene: str, identities: list[CharacterIdentity]) -> list[CharacterIdentity]:
    matched = [identity for identity in identities if identity.display_name in scene]
    if matched:
        return matched
    if any(pronoun in scene for pronoun in ("她", "他", "TA", "ta")):
        return identities[:1]
    return identities


def _look_switch_reason(scene: str) -> str | None:
    for term in _LOOK_SWITCH_TERMS:
        if term in scene:
            return term
    return None


def _is_continuity_scene(scene: str) -> bool:
    return any(term in scene for term in _CONTINUITY_TERMS)


def _scene_id(scene: str, index: int) -> str:
    match = re.match(r"(EP\d+_SC\d+|SC\d+|第[一二三四五六七八九十\d]+场)", scene)
    return match.group(1) if match is not None else f"SC{index:03d}"


def _nearby_sentence(text: str, offset: int) -> str:
    start = max(text.rfind("。", 0, offset), text.rfind("\n", 0, offset), 0)
    end_candidates = [value for value in (text.find("。", offset), text.find("\n", offset)) if value != -1]
    end = min(end_candidates) if end_candidates else min(len(text), offset + 180)
    return " ".join(text[start:end].strip("。\n :：，,").split())[:180]


def _character_id(name: str, index: int) -> str:
    initials = "".join(_PINYIN_INITIALS.get(character, "C") for character in name[:3])
    return f"CHAR_{initials or 'C'}_{index:03d}"
