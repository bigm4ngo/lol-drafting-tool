#!/usr/bin/env bash
# League Draft Lab v3.1 — Ubuntu data collector setup.
# Run once on the home server (Ubuntu). Creates a venv, installs deps,
# initializes static data + config, and installs a systemd service.
set -euo pipefail

cd "$(dirname "$0")"

echo "==> Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "==> Refreshing champion + static data (current patch)..."
python data_dragon_maps.py
python -c "from static_data import StaticDataCatalog; StaticDataCatalog.load(refresh=True)"

echo "==> Creating default config_profile.json if needed..."
python - <<'PY'
from config_manager import load_profile
load_profile()
print("config_profile.json ready")
PY

if [ ! -f config.env ]; then
  echo "==> No config.env found. Create it with your Riot development key:"
  echo "    echo 'RIOT_API_KEY=RGAPI-...' > config.env"
else
  echo "==> config.env already present."
fi

echo "==> Installing systemd service..."
mkdir -p ~/.config/systemd/user
# Inject the real absolute collector path into the unit.
APP_DIR="$(pwd)"
sed -e "s|@APP_DIR@|$APP_DIR|g" systemd/lol-draft-collector.service \
  > ~/.config/systemd/user/lol-draft-collector.service
systemctl --user daemon-reload 2>/dev/null || true
systemctl --user enable lol-draft-collector.service 2>/dev/null || true
systemctl --user start lol-draft-collector.service 2>/dev/null || true

echo
echo "Setup complete. Quick checks:"
echo "  python collector_daemon.py --status"
echo "  systemctl --user status lol-draft-collector"
echo
echo "The collector auto-boots at 6am and shuts down at 2am via your existing"
echo "server power schedule. Bundles land in $(pwd)/outbox/"
echo "Set the Riot key in config.env, then systemctl --user restart lol-draft-collector."
