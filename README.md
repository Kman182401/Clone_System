# Clone System (Current Runtime Snapshot)

This repository captures the **currently running clone system** from the local machine.
It includes only the active sync tooling and excludes older or unused variants to avoid confusion.

Included:
- `bin/clone` — one‑shot sync runner with PII/secret scanning and smart file analysis
- `bin/clone-watch` — debounced inotify watcher
- `bin/clone-watch-daemon` — systemd‑friendly sync daemon with health endpoint
- `bin/clone-status` — status dashboard
- `lib/` — AI relevance + PII detection helpers used by `bin/clone`
- `tests/` — unit tests for clone matching and daemon helpers
- `.clone.toml` — runtime configuration (redacted where needed)
- `ops/systemd/user/file-window-clone-watch.service` — the **current** systemd user unit

Excluded (intentionally):
- Regression copies (`File-Window_regression_cqr*`) and any other legacy clones
- Runtime state files: `.clone-watch-*.json`, `.clone-watch.lock`, `.sync_status`
- Caches, logs, and virtualenvs

## PII Redaction

All detected personal identifiers are replaced with placeholders, for example:
- `<REDACTED_USER>`
- `<REDACTED_GITHUB_USER>`

If you need additional redaction patterns, add them in `lib/pii_detector.py`.

## Quick Start

Requirements:
- Python 3.11+
- `watchdog` (for `clone-watch-daemon`)
- `networkx` (optional; improves smart analysis)

Install (example):
```bash
python3 -m pip install watchdog networkx
```

Run one‑shot sync:
```bash
bin/clone --scan-secrets --push --rebase
```

Run the daemon directly:
```bash
bin/clone-watch-daemon --no-notify
```

## systemd (User Service)

The **currently running** service on this machine is:
- `file-window-clone-watch.service`

The unit file is preserved under:
- `ops/systemd/user/file-window-clone-watch.service`

If you want to install it elsewhere, copy the file into:
`~/.config/systemd/user/` and run:
```bash
systemctl --user daemon-reload
systemctl --user enable --now file-window-clone-watch.service
```

Update the `ExecStart` and `WorkingDirectory` fields if your repo lives in a
different location.
