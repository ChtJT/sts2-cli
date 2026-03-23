#!/usr/bin/env python3
"""Shared runtime helpers for STS2 agent tooling."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "Sts2Headless" / "Sts2Headless.csproj"
LIB_DIR = ROOT / "lib"

_STRIP_KEYS = {
    "description",
    "after_upgrade",
    "enchantment",
    "enchantment_amount",
    "affliction",
    "affliction_amount",
    "id",
    "draw_pile_count",
    "discard_pile_count",
    "upgraded",
    "act_name",
}


def read_target_framework(project_path: Path = PROJECT) -> str:
    """Read the SDK target framework from the project file."""
    try:
        root = ET.parse(project_path).getroot()
        for elem in root.iter():
            if elem.tag.endswith("TargetFramework") and elem.text:
                return elem.text.strip()
    except (ET.ParseError, FileNotFoundError):
        pass
    return "net10.0"


TARGET_FRAMEWORK = read_target_framework()


def built_dll_path() -> Path:
    return ROOT / "Sts2Headless" / "bin" / "Debug" / TARGET_FRAMEWORK / "Sts2Headless.dll"


def compact_json(obj: Any, depth: int = 0) -> Any:
    if isinstance(obj, dict):
        result: Dict[str, Any] = {}
        for key, value in obj.items():
            if key in _STRIP_KEYS:
                continue
            if key == "context" and obj.get("decision") == "combat_play":
                continue
            if key == "player" and obj.get("decision") == "combat_play":
                if isinstance(value, dict):
                    potions = value.get("potions")
                    if potions:
                        result["potions"] = compact_json(potions, depth + 1)
                continue
            if key == "relics" and isinstance(value, list) and depth > 0:
                result[key] = [
                    compact_json(item, depth + 1) if isinstance(item, dict) else item
                    for item in value
                ]
                continue
            result[key] = compact_json(value, depth + 1)
        return result
    if isinstance(obj, list):
        return [compact_json(item, depth + 1) for item in obj]
    return obj


def sanitize_json(obj: Any) -> Any:
    if isinstance(obj, str):
        return "".join(ch for ch in obj if ch >= " " or ch in "\n\t")
    if isinstance(obj, dict):
        return {key: sanitize_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(item) for item in obj]
    return obj


def _expand_game_dir(path: str) -> Iterable[Path]:
    base = Path(path).expanduser()
    if base.name.startswith("data_sts2_"):
        yield base
        return
    if (base / "sts2.dll").is_file():
        yield base
        return
    if base.name == "Resources":
        yield base / "data_sts2_macos_arm64"
        yield base / "data_sts2_macos_x86_64"
        return
    if base.name == "SlayTheSpire2.app":
        resources = base / "Contents" / "Resources"
        yield resources / "data_sts2_macos_arm64"
        yield resources / "data_sts2_macos_x86_64"
        return
    yield base


def candidate_game_dirs(explicit: Optional[str] = None) -> Iterable[Path]:
    seen = set()

    def emit(path: Path) -> Iterable[Path]:
        resolved = str(path)
        if resolved in seen:
            return []
        seen.add(resolved)
        return [path]

    for raw in (explicit, os.environ.get("STS2_GAME_DIR")):
        if raw:
            for candidate in _expand_game_dir(raw):
                for item in emit(candidate):
                    yield item

    system = platform.system()
    if system == "Darwin":
        resources = Path.home() / "Library" / "Application Support" / "Steam" / "steamapps" / "common" / "Slay the Spire 2" / "SlayTheSpire2.app" / "Contents" / "Resources"
        for name in ("data_sts2_macos_arm64", "data_sts2_macos_x86_64"):
            for item in emit(resources / name):
                yield item

        volumes = Path("/Volumes")
        if volumes.exists():
            for volume in sorted(volumes.iterdir()):
                steam_resources = volume / "SteamLibrary" / "steamapps" / "common" / "Slay the Spire 2" / "SlayTheSpire2.app" / "Contents" / "Resources"
                for name in ("data_sts2_macos_arm64", "data_sts2_macos_x86_64"):
                    for item in emit(steam_resources / name):
                        yield item
    elif system == "Linux":
        for steam_root in ("~/.steam/steam", "~/.local/share/Steam"):
            base = Path(steam_root).expanduser() / "steamapps" / "common" / "Slay the Spire 2"
            for item in emit(base):
                yield item
    elif system == "Windows":
        base = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2")
        for item in emit(base):
            yield item


def find_game_dir(explicit: Optional[str] = None) -> Optional[str]:
    for candidate in candidate_game_dirs(explicit):
        if candidate.is_dir() and (candidate / "sts2.dll").is_file():
            return str(candidate)
    return None


def find_dotnet(explicit: Optional[str] = None) -> Optional[str]:
    candidates = [
        explicit,
        os.environ.get("DOTNET"),
        str(Path.home() / ".dotnet-arm64" / "dotnet"),
        str(Path.home() / ".dotnet" / "dotnet"),
        "/usr/local/share/dotnet/dotnet",
        "dotnet",
    ]
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return candidate
    return None


class Sts2Process:
    """Thin JSON client around the headless simulator process."""

    def __init__(
        self,
        game_dir: Optional[str] = None,
        dotnet: Optional[str] = None,
        no_build: bool = True,
        forward_stderr: bool = True,
    ) -> None:
        self.explicit_game_dir = game_dir
        self.explicit_dotnet = dotnet
        self.no_build = no_build
        self.forward_stderr = forward_stderr
        self.proc: Optional[subprocess.Popen[str]] = None
        self._stderr_thread: Optional[threading.Thread] = None

    def start(self) -> Dict[str, Any]:
        dotnet = find_dotnet(self.explicit_dotnet)
        if not dotnet:
            raise RuntimeError("Could not find dotnet. Set --dotnet or add it to PATH.")

        game_dir = find_game_dir(self.explicit_game_dir)
        if not game_dir:
            raise RuntimeError(
                "Could not find Slay the Spire 2 data directory. "
                "Pass --game-dir or set STS2_GAME_DIR."
            )

        if not (LIB_DIR / "sts2.dll").is_file():
            raise RuntimeError(
                "Missing lib/sts2.dll. Run ./setup.sh '<game data dir>' before starting the agent."
            )

        command = [dotnet, "run"]
        if self.no_build and built_dll_path().is_file():
            command.append("--no-build")
        command.extend(["--project", str(PROJECT)])

        env = os.environ.copy()
        env["STS2_GAME_DIR"] = game_dir
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(ROOT),
            env=env,
        )

        if self.forward_stderr and self.proc.stderr is not None:
            def _forward() -> None:
                assert self.proc is not None and self.proc.stderr is not None
                for line in self.proc.stderr:
                    print(f"[STS2] {line.rstrip()}")

            self._stderr_thread = threading.Thread(target=_forward, daemon=True)
            self._stderr_thread.start()

        ready = self.read()
        if ready is None:
            raise RuntimeError("Failed to start simulator: no ready message received.")
        return ready

    def read(self) -> Optional[Dict[str, Any]]:
        if not self.proc or not self.proc.stdout:
            return None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            return sanitize_json(json.loads(line))

    def send(self, command: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("STS2 process is not running.")
        self.proc.stdin.write(json.dumps(command, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        return self.read()

    def close(self) -> None:
        if not self.proc:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self.proc.stdin.flush()
        except OSError:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        finally:
            self.proc = None

    def __enter__(self) -> "Sts2Process":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
