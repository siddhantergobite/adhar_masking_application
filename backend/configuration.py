from __future__ import annotations

import os
import re
from pathlib import Path


_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INLINE_COMMENT = re.compile(r"\s+#.*$")


def load_project_environment() -> None:
    """Load the project .env before settings and batch paths are initialized.

    python-dotenv is optional in this project, so retain a small stdlib fallback
    for ordinary KEY=value entries when that package is not installed.
    Existing process environment variables always take precedence.
    """
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_simple_env_file(env_path)
    else:
        load_dotenv(env_path, override=False)


def _load_simple_env_file(env_path: Path) -> None:
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not _ENV_KEY.fullmatch(key) or key in os.environ:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = _INLINE_COMMENT.sub("", value).rstrip()
        os.environ[key] = value
