"""Single-device helper: point the collector's output at a local draft app.

On the classic two-machine setup the collector writes ``outbox/*.sync.zip``
bundles that get copied (rsync/scp/tailscale) to the Windows PC's
``draft_app/sync_inbox/``. When both apps live on the same machine there is
nothing to copy: the collector can write its bundles straight into the draft
app's inbox, and the draft app auto-ingests them from there.

This helper records that target path in ``config_profile.json``
(``sync.outbox_dir``).

Usage:
    python single_device_sync.py --show      # print the current outbox target
    python single_device_sync.py --link      # bundles go to ..\\draft_app\\sync_inbox
    python single_device_sync.py --unlink    # back to the collector's own outbox/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from config_manager import PROFILE_PATH, load_profile, save_profile
from runtime_paths import PROJECT_ROOT


def default_draft_inbox() -> Path:
    """The draft app's inbox, assuming the standard sibling-folder layout."""
    return PROJECT_ROOT.parent / "draft_app" / "sync_inbox"


def configured_outbox_dir(profile: Mapping[str, Any]) -> Path:
    """The effective bundle output directory (mirrors collector_daemon)."""
    sync = profile.get("sync", {}) or {}
    raw = str(sync.get("outbox_dir", "") or "").strip()
    if not raw:
        return PROJECT_ROOT / "outbox"
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def local_outbox() -> Path:
    return PROJECT_ROOT / "outbox"


def show() -> None:
    profile = load_profile(PROFILE_PATH)
    inbox = default_draft_inbox()
    linked = configured_outbox_dir(profile) != local_outbox()
    print(f"profile            : {PROFILE_PATH}")
    print(f"outbox target      : {configured_outbox_dir(profile)}")
    state = "found" if inbox.is_dir() else "not found"
    print(f"draft app inbox    : {inbox} ({state})")
    print(f"single-device link : {'active' if linked else 'not active'}")


def link() -> None:
    profile = load_profile(PROFILE_PATH)
    inbox = default_draft_inbox()
    inbox.mkdir(parents=True, exist_ok=True)
    keep = inbox / ".gitkeep"
    if not keep.exists():
        keep.touch()
    sync = dict(profile.get("sync", {}) or {})
    sync["outbox_dir"] = str(inbox)
    profile["sync"] = sync
    save_profile(profile, PROFILE_PATH)
    print(f"Collector bundles now go straight to: {inbox}")
    print("The draft app auto-ingests them (Settings -> auto-ingest server data).")


def unlink() -> None:
    profile = load_profile(PROFILE_PATH)
    sync = dict(profile.get("sync", {}) or {})
    sync["outbox_dir"] = ""
    profile["sync"] = sync
    save_profile(profile, PROFILE_PATH)
    print(f"Collector bundles go to the local folder again: {local_outbox()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--show", action="store_true", help="Print the current outbox target.")
    group.add_argument(
        "--link",
        action="store_true",
        help="Point the outbox at the sibling draft app's sync_inbox.",
    )
    group.add_argument(
        "--unlink", action="store_true", help="Restore the collector's own outbox/ folder."
    )
    args = parser.parse_args()
    if args.show:
        show()
    elif args.link:
        link()
    else:
        unlink()


if __name__ == "__main__":
    main()
