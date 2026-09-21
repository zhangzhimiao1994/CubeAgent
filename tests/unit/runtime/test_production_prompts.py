from __future__ import annotations

from agent_hub.runtime.production import (
    CharacterIdentity,
    CharacterLook,
    DeterministicIdentityValidator,
    build_production_plan,
    build_identity_lock_prompt,
    production_metadata_for_label,
)


def test_identity_lock_prioritizes_face_before_costume() -> None:
    identity = CharacterIdentity(
        character_id="CHAR_LXM_001",
        display_name="林小满",
        role_type="女主",
        identity_prompt="26岁甜品店店长，圆杏眼，柔和下颌线，黑色低马尾",
        identity_traits=("圆杏眼", "柔和下颌线", "黑色低马尾"),
        forbidden_drift=("换脸", "变年龄", "换发际线"),
        master_reference_artifact_ids=(),
        embedding_refs=(),
    )
    look = CharacterLook(
        look_id="LOOK_001",
        character_id="CHAR_LXM_001",
        name="甜品店工作服",
        scene_applicability=("EP01_SC01",),
        costume_traits=("浅粉围裙", "白衬衫"),
        accessories=("胸牌",),
        hair_makeup_variations=("低马尾保持",),
        forbidden_identity_changes=("不要继承服装参考模特的脸",),
    )

    prompt = build_identity_lock_prompt(identity, look=look)

    assert prompt.index("IDENTITY LOCK") < prompt.index("LOOK / COSTUME")
    assert "CHAR_LXM_001" in prompt
    assert "只允许修改服装" in prompt
    assert "不要继承服装参考图中的脸" in prompt


def test_production_metadata_extracts_character_and_look() -> None:
    metadata = production_metadata_for_label(
        "角色锁定资产：林小满",
        "CHARACTER_ID: CHAR_LXM_001\nLOOK_ID: LOOK_002\nIDENTITY LOCK",
    )

    assert metadata == {
        "character_id": "CHAR_LXM_001",
        "look_id": "LOOK_002",
        "production_category": "character_identity",
    }


def test_two_characters_get_separate_ids_and_assets() -> None:
    plan = build_production_plan("女主苏清月，医生。男主林渊，外卖员。第一场：雨夜相遇。")

    assert [item.display_name for item in plan.character_identities] == ["苏清月", "林渊"]
    assert len({item.character_id for item in plan.character_identities}) == 2
    assert all(not item.character_id.startswith("LOOK_") for item in plan.character_identities)


def test_time_jump_switches_look_without_changing_identity() -> None:
    plan = build_production_plan(
        "EP01_SC01：苏清月穿白大褂值夜班。"
        "EP01_SC02：同一夜，她继续追查。"
        "EP01_SC03：第二天回家，她换成居家服。"
    )

    states = [state for state in plan.scene_states if state.character_id == "CHAR_SQY_001"]
    assert states[0].look_id == states[1].look_id
    assert states[2].look_id != states[1].look_id


def test_name_first_cast_lines_create_stable_identity_and_looks() -> None:
    plan = build_production_plan(
        "**苏清月**｜女主，26岁，急诊医生，白大褂、低马尾、银针。\n"
        "**林渊**｜男主，22岁，外卖员，藏蓝冲锋衣、黄色外卖箱、青玉断佩。\n"
        "EP01_SC01：苏清月穿白大褂值夜班，林渊冒雨送餐。\n"
        "EP01_SC02：同一夜，她继续追查。\n"
        "EP01_SC03：第二天回家，她换成浅灰居家服。"
    )

    assert [item.display_name for item in plan.character_identities[:2]] == ["苏清月", "林渊"]
    assert [item.character_id for item in plan.character_identities[:2]] == [
        "CHAR_SQY_001",
        "CHAR_LY_002",
    ]
    su_qingyue = plan.character_identities[0]
    su_looks = [look for look in plan.looks if look.character_id == su_qingyue.character_id]
    assert su_looks[0].look_id == "LOOK_001"
    assert any("白大褂" in "、".join(look.costume_traits) for look in su_looks)
    assert len(su_looks) >= 2
    states = [state for state in plan.scene_states if state.character_id == "CHAR_SQY_001"]
    assert states[0].look_id == states[1].look_id
    assert states[2].look_id != states[1].look_id


def test_low_identity_score_blocks_asset() -> None:
    validator = DeterministicIdentityValidator(scores={"artifact-1": 0.41})
    result = validator.validate("CHAR_LXM_001", "artifact-1", threshold=0.78)

    assert not result.passed
    assert result.identity_score == 0.41
    assert result.issues


def test_missing_identity_score_is_not_silent_success() -> None:
    validator = DeterministicIdentityValidator(scores={})
    result = validator.validate("CHAR_LXM_001", "missing-artifact", threshold=0.78)

    assert not result.passed
    assert result.identity_score is None
    assert result.provider == "deterministic_not_configured"
