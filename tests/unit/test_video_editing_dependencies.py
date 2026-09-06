from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_docker_runtime_installs_ffmpeg_for_video_composition() -> None:
    dockerfile = read("Dockerfile")
    runtime_stage = dockerfile.split("FROM ${PYTHON_IMAGE} AS runtime", maxsplit=1)[1]

    assert "apt-get install" in runtime_stage
    assert "ffmpeg" in runtime_stage


def test_native_installer_declares_ffmpeg_for_supported_package_managers() -> None:
    installer = read("deploy/native/install-packages.sh")
    package_setup = installer.split("# Python 3.12 is installed by uv", maxsplit=1)[0]

    assert "ffmpeg" in package_setup


def test_native_installer_does_not_make_dnf_base_install_depend_on_ffmpeg() -> None:
    installer = read("deploy/native/install-packages.sh")
    dnf_branch = installer.split('elif [[ "$manager" == "dnf" ]]', maxsplit=1)[1]
    package_setup = dnf_branch.split("# Python 3.12 is installed by uv", maxsplit=1)[0]

    assert "packages+=(ffmpeg)" not in package_setup
    assert "install_ffmpeg" in installer
    assert "ffmpeg is required for compose_video" in installer
