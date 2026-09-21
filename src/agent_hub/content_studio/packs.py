"""Manifest-backed Content Studio Pack loading and validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from agent_hub.content_studio import PackManifest, PackRegistry

CORE_VERSION: Final[str] = "0.1.0"
PACK_MANIFEST_SCHEMA: Final[str] = "content_studio.pack.v1"
PACK_TYPES: Final[frozenset[str]] = frozenset(
    {"domain", "format", "platform", "channel", "style"}
)


class PackManifestError(ValueError):
    """Raised when a Content Studio Pack manifest is invalid or incompatible."""


def bundled_pack_manifest_dir() -> Path:
    return Path(__file__).with_name("bundled_packs")


def load_pack_registry(
    manifest_dir: str | Path | None = None,
    *,
    core_version: str = CORE_VERSION,
) -> PackRegistry:
    root = Path(manifest_dir) if manifest_dir is not None else bundled_pack_manifest_dir()
    manifests = _load_manifests(root, core_version=core_version)
    registry = PackRegistry(
        domain_packs=manifests["domain"],
        format_packs=manifests["format"],
        platform_packs=manifests["platform"],
        channel_packs=manifests["channel"],
        style_packs=manifests["style"],
    )
    return registry


def _load_manifests(
    manifest_dir: Path, *, core_version: str
) -> dict[str, dict[str, PackManifest]]:
    if not manifest_dir.is_dir():
        raise PackManifestError(f"manifest_dir does not exist: {manifest_dir}")
    try:
        parsed_core = Version(core_version)
    except InvalidVersion as error:
        raise PackManifestError(f"core_version is invalid: {core_version}") from error

    manifests: dict[str, dict[str, PackManifest]] = {pack_type: {} for pack_type in PACK_TYPES}
    for path in sorted(manifest_dir.glob("*.json")):
        manifest = _load_manifest(path, core_version=parsed_core)
        bucket = manifests[manifest.pack_type]
        if manifest.name in bucket:
            raise PackManifestError(f"duplicate {manifest.pack_type} pack name: {manifest.name}")
        bucket[manifest.name] = manifest

    for pack_type in sorted(PACK_TYPES):
        if not manifests[pack_type]:
            raise PackManifestError(f"missing {pack_type} pack manifest")
    return manifests


def _load_manifest(path: Path, *, core_version: Version) -> PackManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PackManifestError(f"{path.name}: invalid JSON") from error
    if not isinstance(raw, Mapping):
        raise PackManifestError(f"{path.name}: manifest must be an object")
    return _parse_manifest(path.name, raw, core_version=core_version)


def _parse_manifest(
    source: str,
    raw: Mapping[object, object],
    *,
    core_version: Version,
) -> PackManifest:
    schema = _required_string(raw, "manifest", source=source)
    if schema != PACK_MANIFEST_SCHEMA:
        raise PackManifestError(f"{source}: manifest schema must be {PACK_MANIFEST_SCHEMA}")

    pack_type = _required_string(raw, "pack_type", source=source)
    if pack_type not in PACK_TYPES:
        raise PackManifestError(f"{source}: pack_type is unsupported: {pack_type}")

    name = _required_string(raw, "name", source=source)
    _validate_pack_name(name, source=source)
    version = _required_string(raw, "version", source=source)
    _validate_version(version, field="version", source=source)
    schema_version = _required_string(raw, "schema_version", source=source)
    _validate_version(schema_version, field="schema_version", source=source)
    compatible_core = _required_string(raw, "compatible_core", source=source)
    _validate_compatible_core(compatible_core, core_version=core_version, source=source)

    settings = raw.get("settings")
    if not isinstance(settings, Mapping):
        raise PackManifestError(f"{source}: settings must be an object")
    normalized_settings = _validate_settings(pack_type, settings, source=source)

    return PackManifest(
        pack_type=pack_type,
        name=name,
        version=version,
        schema_version=schema_version,
        compatible_core=compatible_core,
        settings=normalized_settings,
    )


def _required_string(raw: Mapping[object, object], field: str, *, source: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise PackManifestError(f"{source}: {field} must be a non-empty string")
    return value


def _validate_pack_name(value: str, *, source: str) -> None:
    if not value.replace("_", "").replace("-", "").isalnum() or value != value.lower():
        raise PackManifestError(f"{source}: name must be lowercase alphanumeric with _ or -")


def _validate_version(value: str, *, field: str, source: str) -> None:
    try:
        Version(value)
    except InvalidVersion as error:
        raise PackManifestError(f"{source}: {field} must be a valid version") from error


def _validate_compatible_core(value: str, *, core_version: Version, source: str) -> None:
    try:
        specifier = SpecifierSet(value)
    except InvalidSpecifier as error:
        raise PackManifestError(f"{source}: compatible_core must be a valid version range") from error
    if core_version not in specifier:
        raise PackManifestError(
            f"{source}: compatible_core {value} does not include core {core_version}"
        )


def _validate_settings(
    pack_type: str,
    settings: Mapping[object, object],
    *,
    source: str,
) -> dict[str, object]:
    normalized = _json_settings_copy(settings, source=source)
    if pack_type == "domain":
        normalized["source_priority"] = _string_tuple(settings, "source_priority", source=source)
        normalized["fact_rules"] = _string_tuple(settings, "fact_rules", source=source)
        return normalized
    if pack_type == "format":
        normalized["structure"] = _setting_string(settings, "structure", source=source)
        return normalized
    if pack_type == "platform":
        aspect_ratio = _setting_string(settings, "aspect_ratio", source=source)
        width = _setting_int(settings, "width", source=source)
        height = _setting_int(settings, "height", source=source)
        min_seconds = _setting_int(settings, "min_seconds", source=source)
        max_seconds = _setting_int(settings, "max_seconds", source=source)
        target_seconds = _setting_int(settings, "target_seconds", source=source)
        hook_seconds = _setting_int(settings, "hook_seconds", source=source)
        visual_change_seconds = _int_tuple(
            settings, "visual_change_seconds", length=2, source=source
        )
        _validate_platform_constraints(
            aspect_ratio=aspect_ratio,
            width=width,
            height=height,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            target_seconds=target_seconds,
            hook_seconds=hook_seconds,
            visual_change_seconds=visual_change_seconds,
            source=source,
        )
        normalized.update(
            {
                "aspect_ratio": aspect_ratio,
                "width": width,
                "height": height,
                "min_seconds": min_seconds,
                "max_seconds": max_seconds,
                "target_seconds": target_seconds,
                "codec": _setting_string(settings, "codec", source=source),
                "subtitle_required": _setting_bool(settings, "subtitle_required", source=source),
                "hook_seconds": hook_seconds,
                "visual_change_seconds": visual_change_seconds,
            }
        )
        return normalized
    if pack_type == "channel":
        normalized["persona"] = _setting_string(settings, "persona", source=source)
        normalized["audience"] = _setting_string(settings, "audience", source=source)
        normalized["banned_phrases"] = _string_tuple(settings, "banned_phrases", source=source)
        return normalized
    if pack_type == "style":
        normalized["subtitle_style"] = _setting_string(settings, "subtitle_style", source=source)
        normalized["visual_language"] = _setting_string(settings, "visual_language", source=source)
        normalized["transition"] = _setting_string(settings, "transition", source=source)
        return normalized
    raise PackManifestError(f"{source}: unsupported pack_type: {pack_type}")


def _json_settings_copy(settings: Mapping[object, object], *, source: str) -> dict[str, object]:
    try:
        copied = json.loads(json.dumps(settings, ensure_ascii=False))
    except (TypeError, ValueError) as error:
        raise PackManifestError(f"{source}: settings must contain JSON values") from error
    if not isinstance(copied, dict):
        raise PackManifestError(f"{source}: settings must be an object")
    return {str(key): value for key, value in copied.items()}


def _validate_platform_constraints(
    *,
    aspect_ratio: str,
    width: int,
    height: int,
    min_seconds: int,
    max_seconds: int,
    target_seconds: int,
    hook_seconds: int,
    visual_change_seconds: tuple[int, ...],
    source: str,
) -> None:
    if not min_seconds <= target_seconds <= max_seconds:
        raise PackManifestError(
            f"{source}: settings.min_seconds <= target_seconds <= max_seconds is required"
        )
    expected_ratio = _aspect_ratio(width, height)
    if aspect_ratio != expected_ratio:
        raise PackManifestError(
            f"{source}: settings.aspect_ratio must match width/height ({expected_ratio})"
        )
    if visual_change_seconds[0] > visual_change_seconds[1]:
        raise PackManifestError(f"{source}: settings.visual_change_seconds must be ascending")
    if hook_seconds >= target_seconds:
        raise PackManifestError(f"{source}: settings.hook_seconds must be less than target_seconds")


def _aspect_ratio(width: int, height: int) -> str:
    def gcd(a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return a

    divisor = gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def _setting_string(settings: Mapping[object, object], field: str, *, source: str) -> str:
    value = settings.get(field)
    if not isinstance(value, str) or not value.strip():
        raise PackManifestError(f"{source}: settings.{field} must be a non-empty string")
    return value


def _setting_int(settings: Mapping[object, object], field: str, *, source: str) -> int:
    value = settings.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PackManifestError(f"{source}: settings.{field} must be a positive integer")
    return value


def _setting_bool(settings: Mapping[object, object], field: str, *, source: str) -> bool:
    value = settings.get(field)
    if not isinstance(value, bool):
        raise PackManifestError(f"{source}: settings.{field} must be a boolean")
    return value


def _string_tuple(settings: Mapping[object, object], field: str, *, source: str) -> tuple[str, ...]:
    value = settings.get(field)
    if not isinstance(value, list) or not value:
        raise PackManifestError(f"{source}: settings.{field} must be a non-empty string list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise PackManifestError(f"{source}: settings.{field} must contain strings")
        result.append(item)
    if len(set(result)) != len(result):
        raise PackManifestError(f"{source}: settings.{field} must not contain duplicates")
    return tuple(result)


def _int_tuple(
    settings: Mapping[object, object], field: str, *, length: int, source: str
) -> tuple[int, ...]:
    value = settings.get(field)
    if not isinstance(value, list) or len(value) != length:
        raise PackManifestError(f"{source}: settings.{field} must contain {length} integers")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise PackManifestError(f"{source}: settings.{field} must contain positive integers")
        result.append(item)
    return tuple(result)


__all__ = [
    "CORE_VERSION",
    "PACK_MANIFEST_SCHEMA",
    "PackManifestError",
    "bundled_pack_manifest_dir",
    "load_pack_registry",
]
