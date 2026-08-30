"""CommunityDragon champion metadata utilities.

This module downloads the current champion summary JSON, builds numerical-ID to
name/alias mappings, and stores a small local cache used by the scraper, engine,
and UI. CommunityDragon is preferred here because its "latest" dataset generally
tracks the live client more closely than patch-pinned Data Dragon assets.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import quote

import requests

LOGGER = logging.getLogger("champion_catalog")

from runtime_paths import PROJECT_ROOT
DEFAULT_CACHE_PATH = PROJECT_ROOT / "data" / "champion_map.json"
CDRAGON_CHAMPION_SUMMARY_URL = (
    "https://raw.communitydragon.org/latest/plugins/"
    "rcp-be-lol-game-data/global/default/v1/champion-summary.json"
)

# CommunityDragon has changed the locale directory used by the live ``latest``
# tree in the past. Keep the old locale-specific path as a secondary probe so
# older mirrors or patch-pinned deployments still work.
CDRAGON_CHAMPION_SUMMARY_URLS = (
    CDRAGON_CHAMPION_SUMMARY_URL,
    (
        "https://raw.communitydragon.org/latest/plugins/"
        "rcp-be-lol-game-data/global/en_us/v1/champion-summary.json"
    ),
)

DDRAGON_VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
DDRAGON_CDN_ROOT = "https://ddragon.leagueoflegends.com/cdn"
CDRAGON_RAW_ROOT = "https://raw.communitydragon.org/latest/"
CDRAGON_CHAMPION_CDN_TEMPLATE = (
    "https://cdn.communitydragon.org/latest/champion/{champion_id}/square"
)

# Riot's static datasets occasionally expose an internal champion alias as the
# display name. Champion ID 62 is the best-known example: the internal asset/API
# name is ``MonkeyKing`` while the player-facing champion name is ``Wukong``.
# Keep the internal alias searchable, but always present the official display
# name in the profile editor, logs, recommendations, and cached catalog.
CHAMPION_DISPLAY_NAME_OVERRIDES: dict[int, str] = {
    62: "Wukong",
}


def normalize_champion_name(value: str) -> str:
    """Return a punctuation-insensitive key for user-entered champion names."""
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _record_completeness(record: ChampionRecord) -> tuple[int, int, int]:
    """Rank how much usable data a duplicate record carries (higher wins)."""
    return (
        bool(record.alias.strip()),
        bool(record.square_portrait_path.strip()),
        bool(record.name.strip()),
    )


@dataclass(frozen=True, slots=True)
class ChampionRecord:
    champion_id: int
    name: str
    alias: str
    square_portrait_path: str = ""

    @property
    def cdn_square_url(self) -> str:
        return CDRAGON_CHAMPION_CDN_TEMPLATE.format(champion_id=self.champion_id)

    @property
    def raw_square_url(self) -> str:
        """Fallback image URL derived from the path in champion-summary.json."""
        path = self.square_portrait_path.strip()
        if not path:
            return self.cdn_square_url

        # CDragon JSON uses /lol-game-data/assets/... paths. Those map to the
        # rcp-be-lol-game-data plugin directory and are lower-cased on disk.
        prefix = "/lol-game-data/assets/"
        if path.casefold().startswith(prefix.casefold()):
            relative = path[len(prefix) :].lower()
            return (
                CDRAGON_RAW_ROOT
                + "plugins/rcp-be-lol-game-data/global/default/"
                + quote(relative, safe="/._-")
            )
        return CDRAGON_RAW_ROOT + quote(path.lstrip("/").lower(), safe="/._-")


class ChampionCatalog:
    """In-memory champion lookup with a JSON cache on disk."""

    def __init__(
        self,
        records: Iterable[ChampionRecord],
        *,
        source: str = "",
    ) -> None:
        normalised_records: list[ChampionRecord] = []
        for record in records:
            display_name = CHAMPION_DISPLAY_NAME_OVERRIDES.get(
                int(record.champion_id), record.name
            )
            normalised_records.append(
                ChampionRecord(
                    champion_id=int(record.champion_id),
                    name=display_name,
                    alias=record.alias,
                    square_portrait_path=record.square_portrait_path,
                )
            )

        # Collapse duplicated champion ids before anything consumes the catalog.
        # Some CommunityDragon responses repeat entries and older JSON caches
        # were saved with the duplicates intact, which made icon grids render
        # the same champion twice. One deterministic winner is kept per id.
        best_by_id: dict[int, ChampionRecord] = {}
        for record in normalised_records:
            existing = best_by_id.get(record.champion_id)
            if existing is None or _record_completeness(record) > _record_completeness(existing):
                best_by_id[record.champion_id] = record

        # Collapse distinct-id entries that share one display name.
        # CommunityDragon ships "variant" rows (e.g. the Jade clan block, ids
        # 60001+, whose display names duplicate classic champions such as
        # Annie, Lux, Wukong). They rendered as literally duplicated icons in
        # the pool picker and double-counted in recommendations. The real
        # champion always has the lowest id, so first-in-order wins below.
        best_by_name: dict[str, ChampionRecord] = {}
        for record in sorted(best_by_id.values(), key=lambda item: item.champion_id):
            key = normalize_champion_name(record.name)
            if key and key not in best_by_name:
                best_by_name[key] = record
        self.records = tuple(
            sorted(best_by_name.values(), key=lambda record: record.champion_id)
        )
        self.source = source
        self.by_id = {record.champion_id: record for record in self.records}
        self.by_name: dict[str, ChampionRecord] = {}
        for record in self.records:
            for value in (record.name, record.alias):
                if value:
                    self.by_name[normalize_champion_name(value)] = record

    def __len__(self) -> int:
        return len(self.records)

    def name_for_id(self, champion_id: int, default: str | None = None) -> str:
        record = self.by_id.get(int(champion_id))
        if record:
            return record.name
        return default if default is not None else f"Champion {champion_id}"

    def alias_for_id(self, champion_id: int, default: str = "") -> str:
        record = self.by_id.get(int(champion_id))
        return record.alias if record else default

    def id_for_name(self, name: str) -> int | None:
        record = self.by_name.get(normalize_champion_name(name))
        return record.champion_id if record else None

    def resolve(self, value: int | str) -> ChampionRecord | None:
        if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
            return self.by_id.get(int(value))
        return self.by_name.get(normalize_champion_name(str(value)))

    def square_url(self, champion_id: int) -> str:
        record = self.by_id.get(int(champion_id))
        return record.cdn_square_url if record else CDRAGON_CHAMPION_CDN_TEMPLATE.format(
            champion_id=int(champion_id)
        )

    def fallback_square_url(self, champion_id: int) -> str:
        record = self.by_id.get(int(champion_id))
        return record.raw_square_url if record else self.square_url(champion_id)

    def save(self, cache_path: Path = DEFAULT_CACHE_PATH) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": self.source or CDRAGON_CHAMPION_SUMMARY_URL,
            "champions": [asdict(record) for record in self.records],
        }
        temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(cache_path)

    @classmethod
    def from_cache(cls, cache_path: Path = DEFAULT_CACHE_PATH) -> "ChampionCatalog":
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        raw_records = payload.get("champions", payload)
        records = [
            ChampionRecord(
                champion_id=int(item["champion_id"]),
                name=str(item["name"]),
                alias=str(item.get("alias", "")),
                square_portrait_path=str(item.get("square_portrait_path", "")),
            )
            for item in raw_records
        ]
        source = str(payload.get("source", "")) if isinstance(payload, dict) else ""
        return cls(records, source=source)

    @staticmethod
    def _records_from_cdragon(payload: Any) -> list[ChampionRecord]:
        """Convert CommunityDragon's champion-summary list into records."""
        if not isinstance(payload, list):
            raise ValueError("Unexpected CommunityDragon champion summary format.")

        records: list[ChampionRecord] = []
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            champion_id = int(item.get("id", -1))
            # -1 is the client placeholder for an unselected champion.
            if champion_id <= 0:
                continue
            name = str(item.get("name", "")).strip()
            alias = str(item.get("alias", "")).strip()
            if not name:
                continue
            records.append(
                ChampionRecord(
                    champion_id=champion_id,
                    name=name,
                    alias=alias,
                    square_portrait_path=str(item.get("squarePortraitPath", "")),
                )
            )
        return records

    @staticmethod
    def _records_from_ddragon(payload: Any) -> list[ChampionRecord]:
        """Convert Data Dragon's champion.json mapping into records.

        Data Dragon is used only as a resilience fallback. It does not expose
        the same CommunityDragon portrait path, but the numerical champion ID,
        display name, and internal alias are enough for the draft engine and
        for the ID-based CommunityDragon portrait endpoint.
        """
        if not isinstance(payload, Mapping):
            raise ValueError("Unexpected Data Dragon champion format.")
        raw_data = payload.get("data", {})
        if not isinstance(raw_data, Mapping):
            raise ValueError("Data Dragon champion payload has no data mapping.")

        records: list[ChampionRecord] = []
        for alias, item in raw_data.items():
            if not isinstance(item, Mapping):
                continue
            raw_id = item.get("key")
            if raw_id in (None, ""):
                continue
            champion_id = int(raw_id)
            if champion_id <= 0:
                continue
            name = str(item.get("name", "")).strip()
            champion_alias = str(item.get("id", alias)).strip()
            if not name:
                continue
            records.append(
                ChampionRecord(
                    champion_id=champion_id,
                    name=name,
                    alias=champion_alias,
                )
            )
        return records

    @staticmethod
    def _validate_records(records: list[ChampionRecord], source: str) -> None:
        # A healthy modern champion catalog contains comfortably more than 100
        # entries. Refuse suspiciously small responses so a proxy error page or
        # partial CDN response can never overwrite a valid local cache.
        if len(records) < 100:
            raise ValueError(
                f"{source} returned only {len(records)} champions; refusing "
                "to replace a potentially valid cache."
            )

    @classmethod
    def download(
        cls,
        *,
        timeout: float = 20.0,
        session: requests.Session | None = None,
    ) -> "ChampionCatalog":
        http = session or requests.Session()
        headers = {"User-Agent": "LeagueDraftLab/2.1"}
        failures: list[str] = []

        # Primary path: CommunityDragon's current locale-neutral live dataset.
        # Secondary path: its historical en_us location.
        for url in CDRAGON_CHAMPION_SUMMARY_URLS:
            try:
                LOGGER.info("Downloading champion metadata from %s", url)
                response = http.get(url, timeout=timeout, headers=headers)
                response.raise_for_status()
                records = cls._records_from_cdragon(response.json())
                cls._validate_records(records, url)
                return cls(records, source=url)
            except (requests.RequestException, ValueError, TypeError, json.JSONDecodeError) as exc:
                failures.append(f"{url}: {exc}")
                LOGGER.warning("Champion metadata source failed: %s (%s)", url, exc)

        # Independent fallback: Riot's public Data Dragon catalog. This makes
        # startup resilient to a CommunityDragon outage or path migration.
        try:
            LOGGER.info("CommunityDragon unavailable; trying Data Dragon fallback.")
            versions_response = http.get(
                DDRAGON_VERSIONS_URL,
                timeout=timeout,
                headers=headers,
            )
            versions_response.raise_for_status()
            versions = versions_response.json()
            if not isinstance(versions, list) or not versions:
                raise ValueError("Data Dragon returned no available versions.")
            version = str(versions[0])
            url = f"{DDRAGON_CDN_ROOT}/{version}/data/en_US/champion.json"
            response = http.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()
            records = cls._records_from_ddragon(response.json())
            cls._validate_records(records, url)
            return cls(records, source=url)
        except (requests.RequestException, ValueError, TypeError, json.JSONDecodeError) as exc:
            failures.append(f"Data Dragon fallback: {exc}")
            LOGGER.warning("Data Dragon champion fallback failed: %s", exc)

        details = "\n  - ".join(failures)
        raise RuntimeError(
            "Unable to download champion metadata from any configured source."
            + (f"\n  - {details}" if details else "")
        )

    @classmethod
    def load(
        cls,
        cache_path: Path = DEFAULT_CACHE_PATH,
        *,
        refresh: bool = False,
        allow_download: bool = True,
    ) -> "ChampionCatalog":
        if not refresh and cache_path.exists():
            try:
                return cls.from_cache(cache_path)
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                LOGGER.exception("Champion cache is invalid; attempting a refresh.")

        if allow_download:
            try:
                catalog = cls.download()
                catalog.save(cache_path)
                LOGGER.info("Saved %d champions to %s", len(catalog), cache_path)
                return catalog
            except (requests.RequestException, RuntimeError, ValueError, OSError):
                LOGGER.exception("Champion metadata download failed.")
                if cache_path.exists():
                    LOGGER.warning("Falling back to the existing champion cache.")
                    return cls.from_cache(cache_path)
                # Do not make the whole desktop application crash merely because
                # both public static-data CDNs are temporarily unavailable. The
                # UI can still open and the user can retry the refresh later.
                LOGGER.error(
                    "No champion cache is available. Starting with an empty "
                    "catalog; use Data Watcher > Refresh static data once online."
                )
                return cls(())

        if cache_path.exists():
            return cls.from_cache(cache_path)
        return cls(())


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the CDragon champion map.")
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Output JSON cache path.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_records",
        help="Print the resulting ID/name map.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    catalog = ChampionCatalog.load(args.cache, refresh=True)
    LOGGER.info("Loaded %d champions.", len(catalog))
    if args.print_records:
        for record in catalog.records:
            print(f"{record.champion_id:>4}  {record.name} ({record.alias})")


if __name__ == "__main__":
    main()
