from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


def _load_clone_module():
    repo_dir = Path(__file__).resolve().parents[1]
    clone_path = repo_dir / "bin" / "clone"
    loader = SourceFileLoader("clone_bin", str(clone_path))
    spec = importlib.util.spec_from_loader("clone_bin", loader)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_match_any_glob_does_not_strip_dotfiles():
    m = _load_clone_module()
    assert m._match_any_glob(".clone.toml", [".clone.toml"])
    assert m._match_any_glob("./.clone.toml", [".clone.toml"])
