#!/usr/bin/env bash
# Start Draft Lab v4 and open http://127.0.0.1:8765. Pass --host 0.0.0.0 to share on a trusted LAN.
cd "$(dirname "$0")" && exec .venv/bin/python -m draft4.server "$@"
