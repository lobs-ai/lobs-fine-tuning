"""
Resolve squad's install location and gateway config.

Squad's home is `~/.squad/` (overridable via SQUAD_HOME). Inside it:
- `config`         — KEY=VALUE env-style file written by `squad onboard`,
                      contains SQUAD_REPO, SQUAD_PORT, SQUAD_TOKEN, SQUAD_URL.
- `squads.json`    — registered squad gateways.
- `<name>/squad.db` — per-squad sqlite gateway DB (chat_messages, tool_calls).
- `<name>/data/`   — workspace, ssh, etc.

Per-agent JSONL transcripts (the *runner* writes these) currently land
under `~/.lobs/agents/{agentType}/sessions/`. That path is hard-coded in
squad's `session-transcript.ts` and is NOT configurable from squad. We
treat it as a separate, parametric input to the harvester.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SquadConfig:
    home: Path
    repo: Path | None
    squads_json_path: Path
    config_path: Path
    extras: dict[str, str]

    def gateway_db(self, squad_name: str = "default") -> Path:
        return self.home / squad_name / "squad.db"


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        k, _, v = s.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_squad_config(home: Path | None = None) -> SquadConfig:
    home = home or Path(os.path.expanduser(os.environ.get("SQUAD_HOME", "~/.squad")))
    cfg_path = home / "config"
    extras = _parse_env_file(cfg_path)
    repo_str = os.environ.get("SQUAD_REPO") or extras.get("SQUAD_REPO")
    repo = Path(os.path.expanduser(repo_str)) if repo_str else None
    return SquadConfig(
        home=home,
        repo=repo,
        squads_json_path=home / "squads.json",
        config_path=cfg_path,
        extras=extras,
    )
