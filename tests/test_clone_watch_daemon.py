from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
import sys
import logging
import tomllib


def _load_clone_watch_daemon_module():
    repo_dir = Path(__file__).resolve().parents[1]
    daemon_path = repo_dir / "bin" / "clone-watch-daemon"
    loader = SourceFileLoader("clone_watch_daemon", str(daemon_path))
    spec = importlib.util.spec_from_loader("clone_watch_daemon", loader)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summarize_clone_stderr_prefers_real_error_over_clone_note():
    m = _load_clone_watch_daemon_module()
    stderr = "\n".join(
        [
            "[clone] Note: working tree has changes outside autopush allowlist; leaving them uncommitted.",
            "git@github.com: Permission denied (publickey).",
            "fatal: Could not read from remote repository.",
        ]
    )
    assert m.summarize_clone_stderr(stderr).startswith("git@github.com: Permission denied (publickey).")


def test_summarize_clone_stderr_falls_back_to_last_non_noise_line():
    m = _load_clone_watch_daemon_module()
    stderr = "\n".join(
        [
            "[clone] Note: working tree has changes outside autopush allowlist; leaving them uncommitted.",
            "some other error line",
        ]
    )
    assert m.summarize_clone_stderr(stderr) == "some other error line"


def test_summarize_clone_stderr_handles_git_index_lock():
    m = _load_clone_watch_daemon_module()
    stderr = "fatal: Unable to create '/home/<REDACTED_USER>/File-Window/.git/index.lock': File exists."
    assert "index.lock" in m.summarize_clone_stderr(stderr)


def test_clone_watch_daemon_reads_pr_on_sync_from_config():
    m = _load_clone_watch_daemon_module()
    repo_dir = Path(__file__).resolve().parents[1]
    cfg_path = repo_dir / ".clone.toml"
    conf = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    expected = conf.get("pr_on_sync", conf.get("global", {}).get("pr_on_sync", False))
    stub = type("Stub", (), {"config_path": cfg_path, "logger": logging.getLogger("test")})()
    cfg = m.CloneWatchDaemon._load_config(stub)
    assert cfg["pr_on_sync"] == expected
