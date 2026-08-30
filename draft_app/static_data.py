"""Patch-aware Data Dragon names and image URLs for items, runes and spells."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

LOGGER = logging.getLogger("static_data")
from runtime_paths import PROJECT_ROOT
CACHE_PATH = PROJECT_ROOT / "data" / "static_data.json"
VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
CDN_ROOT = "https://ddragon.leagueoflegends.com/cdn"


@dataclass(frozen=True, slots=True)
class StaticEntry:
    item_id: int
    name: str
    icon_url: str


class StaticDataCatalog:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.version = str(payload.get("version", ""))
        self.items = {int(k): v for k, v in payload.get("items", {}).items()}
        self.spells = {int(k): v for k, v in payload.get("spells", {}).items()}
        self.runes = {int(k): v for k, v in payload.get("runes", {}).items()}

    def name(self, kind: str, identifier: int) -> str:
        mapping = {"item": self.items, "spell": self.spells, "rune": self.runes}[kind]
        return str(mapping.get(int(identifier), {}).get("name", f"{kind.title()} {identifier}"))

    def icon_url(self, kind: str, identifier: int) -> str:
        mapping = {"item": self.items, "spell": self.spells, "rune": self.runes}[kind]
        return str(mapping.get(int(identifier), {}).get("icon_url", ""))

    def completed_item_ids(self) -> set[int]:
        return {item_id for item_id, item in self.items.items() if bool(item.get("completed"))}

    def item_tags(self, identifier: int) -> frozenset[str]:
        """Return normalized Data Dragon item tags when available.

        Older caches from v2.3 do not contain tags; those safely return an empty
        set and can be upgraded with Data Watcher -> Refresh static data.
        """
        raw = self.items.get(int(identifier), {}).get("tags", [])
        return frozenset(str(tag).casefold() for tag in raw)

    def item_description(self, identifier: int) -> str:
        return str(self.items.get(int(identifier), {}).get("description", ""))

    @classmethod
    def from_cache(cls, path: Path = CACHE_PATH) -> "StaticDataCatalog":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def download(cls, path: Path = CACHE_PATH, timeout: float = 25.0) -> "StaticDataCatalog":
        session = requests.Session()
        session.headers["User-Agent"] = "LeagueDraftLab/2.4.1"
        versions = session.get(VERSIONS_URL, timeout=timeout)
        versions.raise_for_status()
        version = str(versions.json()[0])

        def get_json(url: str) -> Any:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()

        item_payload = get_json(f"{CDN_ROOT}/{version}/data/en_US/item.json")
        spell_payload = get_json(f"{CDN_ROOT}/{version}/data/en_US/summoner.json")
        rune_payload = get_json(f"{CDN_ROOT}/{version}/data/en_US/runesReforged.json")

        items: dict[str, Any] = {}
        for raw_id, item in item_payload.get("data", {}).items():
            item_id = int(raw_id)
            maps = item.get("maps", {})
            gold = item.get("gold", {})
            tags = {str(tag).casefold() for tag in item.get("tags", [])}
            completed = (
                bool(maps.get("11"))
                and bool(gold.get("purchasable", True))
                and int(gold.get("total", 0)) >= 900
                and not item.get("into")
                and "consumable" not in tags
                and "trinket" not in tags
            )
            image_name = item.get("image", {}).get("full", f"{item_id}.png")
            items[str(item_id)] = {
                "name": item.get("name", f"Item {item_id}"),
                "icon_url": f"{CDN_ROOT}/{version}/img/item/{image_name}",
                "completed": completed,
                "gold": int(gold.get("total", 0)),
                "tags": [str(tag) for tag in item.get("tags", [])],
                "description": str(item.get("plaintext", "") or ""),
            }

        spells: dict[str, Any] = {}
        for spell in spell_payload.get("data", {}).values():
            spell_id = int(spell.get("key", 0))
            if spell_id <= 0:
                continue
            image_name = spell.get("image", {}).get("full", "")
            spells[str(spell_id)] = {
                "name": spell.get("name", f"Spell {spell_id}"),
                "icon_url": f"{CDN_ROOT}/{version}/img/spell/{image_name}",
            }

        runes: dict[str, Any] = {}
        for style in rune_payload:
            runes[str(int(style["id"]))] = {
                "name": style["name"],
                "icon_url": f"{CDN_ROOT}/img/{style['icon']}",
            }
            for slot in style.get("slots", []):
                for rune in slot.get("runes", []):
                    runes[str(int(rune["id"]))] = {
                        "name": rune["name"],
                        "icon_url": f"{CDN_ROOT}/img/{rune['icon']}",
                    }

        payload = {"version": version, "items": items, "spells": spells, "runes": runes}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)
        LOGGER.info("Cached Data Dragon static data for patch %s.", version)
        return cls(payload)

    @classmethod
    def load(cls, *, refresh: bool = False, allow_download: bool = True) -> "StaticDataCatalog":
        if CACHE_PATH.exists() and not refresh:
            try:
                return cls.from_cache()
            except (OSError, ValueError, json.JSONDecodeError):
                LOGGER.exception("Static-data cache was invalid.")
        if allow_download:
            try:
                return cls.download()
            except requests.RequestException:
                LOGGER.exception("Could not refresh Data Dragon static data.")
                if CACHE_PATH.exists():
                    return cls.from_cache()
        return cls({"version": "", "items": {}, "spells": {}, "runes": {}})
