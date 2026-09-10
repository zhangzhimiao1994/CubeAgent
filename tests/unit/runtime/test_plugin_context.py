from agent_hub.runtime.plugin_context import (
    requested_plugin_context_payload,
    requested_plugin_context_text,
    requested_plugin_names,
)


def test_requested_plugin_names_parse_comma_string_and_dedupe() -> None:
    names = requested_plugin_names(
        {
            "requested_plugins": "Runway, github, runway, app-123@openai-curated-remote",
        }
    )

    assert names == ("Runway", "github", "app-123@openai-curated-remote")


def test_requested_plugin_context_rejects_empty_and_unsafe_items() -> None:
    names = requested_plugin_names(
        {
            "requested_plugins": (
                "",
                "  ",
                "bad\x00name",
                "<Runway>",
                7,
                "Higgsfield",
            )
        }
    )

    assert names == ("Runway", "Higgsfield")


def test_requested_plugin_context_payload_marks_intent_not_availability() -> None:
    payload = requested_plugin_context_payload({"requested_plugins": ("runway",)})

    assert payload["requested_plugins"] == ("runway",)
    assert "not proof" in str(payload["policy"])
    assert "installed" in str(payload["policy"])


def test_requested_plugin_context_text_is_bounded_prompt_block() -> None:
    text = requested_plugin_context_text({"requested_plugins": "runway,github"})

    assert text.startswith("<REQUESTED_PLUGIN_CONTEXT>")
    assert "runway" in text
    assert "github" in text
    assert "Never claim a plugin was used" in text
    assert text.endswith("</REQUESTED_PLUGIN_CONTEXT>")
