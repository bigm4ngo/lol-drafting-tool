"""Rate-limit-aware Riot Match-V5 collector.

Heavy work is intentionally separate from the live draft UI. The collector stores
raw, lossless-enough participant data so analytics can be rebuilt when weights or
algorithms change without redownloading every match.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import requests
from dotenv import load_dotenv

from config_manager import ENV_PATH, PROFILE_PATH, load_profile, reconcile_legacy_api_key
from data_dragon_maps import ChampionCatalog

from runtime_paths import PROJECT_ROOT
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "draft_data.sqlite3"
DEFAULT_LOG_PATH = PROJECT_ROOT / "scraper.log"
LOGGER = logging.getLogger("riot_scraper")

ROLE_MAP = {
    "TOP": "TOP", "JUNGLE": "JUNGLE", "MIDDLE": "MID", "MID": "MID",
    "BOTTOM": "ADC", "BOT": "ADC", "UTILITY": "SUPPORT", "SUPPORT": "SUPPORT",
}
VALID_ROLES = frozenset({"TOP", "JUNGLE", "MID", "ADC", "SUPPORT"})
HIGH_TIERS = frozenset({"MASTER", "GRANDMASTER", "CHALLENGER"})
TIER_ORDER = (
    "IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD",
    "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER",
)


def expand_target_tiers(target_elo: str) -> tuple[str, ...]:
    """Expand GUI cohorts such as EMERALD+ into concrete Riot API tiers."""
    target = str(target_elo).strip().upper()
    if target.endswith("+"):
        base = target[:-1]
        if base not in TIER_ORDER:
            raise ValueError(f"Unsupported Elo cohort: {target_elo}")
        return TIER_ORDER[TIER_ORDER.index(base):]
    if target not in TIER_ORDER:
        raise ValueError(f"Unsupported Elo tier: {target_elo}")
    return (target,)


class RiotApiError(RuntimeError):
    pass


class RiotAuthenticationError(RiotApiError):
    """The saved Riot API key was not accepted by the Riot API."""


class RiotForbiddenError(RiotApiError):
    pass


class RiotRateLimitError(RiotApiError):
    """Riot continued returning 429 after request-level retries."""

    def __init__(self, message: str, retry_after: float = 1.0) -> None:
        super().__init__(message)
        self.retry_after = max(1.0, float(retry_after))


@dataclass(slots=True)
class ScrapeSettings:
    target_elo: str
    platform: str
    regional_route: str
    queue: str
    queue_id: int
    divisions: list[str]
    players_per_run: int
    matches_per_player: int
    max_matches_per_run: int
    max_concurrent_requests: int
    min_game_duration_seconds: int
    request_timeout_seconds: float
    max_retries: int
    # 0 disables the cutoff. When set, Match-V5 listing only returns games
    # played within the last N days, so discovery stops re-listing stale
    # history and patch rotation does not dilute current-patch collection.
    matches_window_days: int

    @classmethod
    def from_profile(cls, profile: Mapping[str, Any]) -> "ScrapeSettings":
        scraper = profile.get("scraper", {})
        return cls(
            target_elo=str(profile.get("target_elo", "EMERALD")).upper(),
            platform=str(profile.get("region_platform", "OC1")).upper(),
            regional_route=str(profile.get("regional_route", "SEA")).upper(),
            queue=str(profile.get("queue", "RANKED_SOLO_5x5")),
            queue_id=int(profile.get("queue_id", 420)),
            divisions=[str(x).upper() for x in profile.get("divisions", ["I"])],
            players_per_run=max(1, int(scraper.get("players_per_run", 80))),
            matches_per_player=max(1, min(100, int(scraper.get("matches_per_player", 15)))),
            max_matches_per_run=max(1, int(scraper.get("max_matches_per_run", 600))),
            # Four in-flight requests are sufficient because the 100/120s window
            # is the real bottleneck and greatly reduce duplicate 429 bursts.
            max_concurrent_requests=max(1, min(4, int(scraper.get("max_concurrent_requests", 4)))),
            min_game_duration_seconds=max(0, int(scraper.get("min_game_duration_seconds", 900))),
            request_timeout_seconds=max(5.0, float(scraper.get("request_timeout_seconds", 20))),
            max_retries=max(1, int(scraper.get("max_retries", 5))),
            matches_window_days=max(0, int(scraper.get("matches_window_days", 0))),
        )


class SlidingWindowRateLimiter:
    """Conservative per-host guard beneath Riot's 20/s and 100/120s limits.

    The small safety margin absorbs clock differences, requests already in flight,
    and rolling-window boundary differences between this process and Riot's edge.
    """

    SHORT_LIMIT = 18
    SHORT_WINDOW = 1.05
    LONG_LIMIT = 95
    LONG_WINDOW = 121.0

    def __init__(self) -> None:
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()
        self._blocked_until = 0.0

    async def pause(self, seconds: float) -> None:
        async with self._lock:
            self._blocked_until = max(self._blocked_until, time.monotonic() + max(0.0, seconds))

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self.LONG_WINDOW:
                    self._timestamps.popleft()
                wait_for = max(0.0, self._blocked_until - now)
                recent = [stamp for stamp in self._timestamps if now - stamp < self.SHORT_WINDOW]
                if len(recent) >= self.SHORT_LIMIT:
                    wait_for = max(wait_for, recent[0] + self.SHORT_WINDOW - now)
                if len(self._timestamps) >= self.LONG_LIMIT:
                    wait_for = max(wait_for, self._timestamps[0] + self.LONG_WINDOW - now)
                if wait_for <= 0:
                    self._timestamps.append(now)
                    return
            await asyncio.sleep(wait_for + 0.04)


class AsyncRiotClient:
    def __init__(self, api_key: str, *, timeout: float, max_retries: int) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._limiters: dict[str, SlidingWindowRateLimiter] = {}
        self._local = threading.local()
        self._rate_limit_hits = 0
        self._last_retry_after = 0.0

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({
                "X-Riot-Token": self.api_key,
                "Accept": "application/json",
                "User-Agent": "LeagueDraftLab/3.0.0",
            })
            self._local.session = session
        return session

    @staticmethod
    def _host(route: str) -> str:
        return f"{route.lower()}.api.riotgames.com"

    def _get_sync(self, url: str, params: Mapping[str, Any] | None) -> requests.Response:
        return self._session().get(url, params=params, timeout=self.timeout)

    async def get_json(
        self,
        route: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        allow_404: bool = False,
    ) -> Any:
        host = self._host(route)
        limiter = self._limiters.setdefault(host, SlidingWindowRateLimiter())
        url = f"https://{host}{path}"

        for attempt in range(1, self.max_retries + 1):
            await limiter.acquire()
            try:
                response = await asyncio.to_thread(self._get_sync, url, params)
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise RiotApiError(f"Network failure for {path}: {exc}") from exc
                delay = min(30.0, 2 ** (attempt - 1))
                LOGGER.warning("Network error %s; retrying in %.1fs", exc, delay)
                await asyncio.sleep(delay)
                continue

            status = response.status_code
            if status == 200:
                try:
                    return response.json()
                except requests.JSONDecodeError as exc:
                    raise RiotApiError(f"Invalid JSON for {path}") from exc
            if status == 404 and allow_404:
                return None
            if status == 429:
                raw_retry = response.headers.get("Retry-After", "1")
                try:
                    retry = max(1.0, float(raw_retry))
                except ValueError:
                    retry = 1.0
                self._rate_limit_hits += 1
                self._last_retry_after = max(self._last_retry_after, retry)
                LOGGER.warning(
                    "429 rate limit (%s) on %s; retry-after %.1fs; attempt %d/%d",
                    response.headers.get("X-Rate-Limit-Type", "unknown"), path, retry,
                    attempt, self.max_retries,
                )
                # Pause every request sharing this host. A little extra margin
                # prevents other in-flight workers from immediately hitting 429 again.
                await limiter.pause(retry + 0.35)
                if attempt == self.max_retries:
                    raise RiotRateLimitError(
                        f"Rate limit persisted for {path}", retry_after=retry
                    )
                await asyncio.sleep(retry + 0.1)
                continue
            if status == 401:
                body = response.text[:300].replace("\n", " ")
                LOGGER.error("401 Unauthorized on %s: %s", path, body)
                raise RiotAuthenticationError(
                    "Riot rejected the saved API key (401 Unknown apikey). "
                    "Development keys expire after about 24 hours; the key may also be "
                    "mistyped or copied incompletely. Generate a fresh key in the Riot "
                    "Developer Portal, paste it into Settings, and save again."
                )
            if status == 403:
                LOGGER.error("403 Forbidden on %s. The development key may be expired or revoked.", path)
                raise RiotForbiddenError(
                    "Riot returned 403 Forbidden. Generate a fresh development key and "
                    "save it in Settings before collecting more matches."
                )
            if status in {500, 502, 503, 504} and attempt < self.max_retries:
                delay = min(30.0, 2 ** (attempt - 1))
                LOGGER.warning("Riot returned %d for %s; retrying in %.1fs", status, path, delay)
                await asyncio.sleep(delay)
                continue
            body = response.text[:400].replace("\n", " ")
            raise RiotApiError(f"Riot API {status} for {path}: {body}")
        raise RiotApiError(f"Exhausted retries for {path}")


    def consume_rate_limit_signal(self) -> tuple[int, float]:
        """Return and reset the 429 activity observed since the last call."""
        hits = self._rate_limit_hits
        retry_after = self._last_retry_after
        self._rate_limit_hits = 0
        self._last_retry_after = 0.0
        return hits, retry_after


class DraftDatabase:
    """Thread-safe SQLite raw store. Schema migrations preserve v1 databases."""

    PARTICIPANT_COLUMNS: dict[str, str] = {
        "game_duration_seconds": "INTEGER NOT NULL DEFAULT 0",
        "kills": "INTEGER NOT NULL DEFAULT 0",
        "deaths": "INTEGER NOT NULL DEFAULT 0",
        "assists": "INTEGER NOT NULL DEFAULT 0",
        "gold_earned": "INTEGER NOT NULL DEFAULT 0",
        "physical_damage_to_champions": "INTEGER NOT NULL DEFAULT 0",
        "magic_damage_to_champions": "INTEGER NOT NULL DEFAULT 0",
        "true_damage_to_champions": "INTEGER NOT NULL DEFAULT 0",
        "damage_to_objectives": "INTEGER NOT NULL DEFAULT 0",
        "damage_taken": "INTEGER NOT NULL DEFAULT 0",
        "damage_mitigated": "INTEGER NOT NULL DEFAULT 0",
        "time_cc_dealt": "INTEGER NOT NULL DEFAULT 0",
        "vision_score": "REAL NOT NULL DEFAULT 0",
        "total_minions_killed": "INTEGER NOT NULL DEFAULT 0",
        "neutral_minions_killed": "INTEGER NOT NULL DEFAULT 0",
        "turret_takedowns": "INTEGER NOT NULL DEFAULT 0",
        "inhibitor_takedowns": "INTEGER NOT NULL DEFAULT 0",
        "time_ccing_others": "REAL NOT NULL DEFAULT 0",
        "team_damage_percentage": "REAL NOT NULL DEFAULT 0",
        "kill_participation": "REAL NOT NULL DEFAULT 0",
        "lane_minions_first_10": "REAL NOT NULL DEFAULT 0",
        "takedowns_first_15": "REAL NOT NULL DEFAULT 0",
        "solo_kills": "REAL NOT NULL DEFAULT 0",
        "skillshots_dodged": "REAL NOT NULL DEFAULT 0",
        "objectives_stolen": "REAL NOT NULL DEFAULT 0",
        "challenges_json": "TEXT NOT NULL DEFAULT '{}'",
        "summoner1_id": "INTEGER NOT NULL DEFAULT 0",
        "summoner2_id": "INTEGER NOT NULL DEFAULT 0",
        "primary_style_id": "INTEGER NOT NULL DEFAULT 0",
        "sub_style_id": "INTEGER NOT NULL DEFAULT 0",
        "rune_page_json": "TEXT NOT NULL DEFAULT '[]'",
        "stat_perks_json": "TEXT NOT NULL DEFAULT '{}'",
        "items_json": "TEXT NOT NULL DEFAULT '[]'",
    }

    def __init__(self, path: Path = DEFAULT_DB_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._initialize()

    def _initialize(self) -> None:
        with self._lock, self.connection:
            self.connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                PRAGMA foreign_keys=ON;
                CREATE TABLE IF NOT EXISTS matches (
                    match_id TEXT PRIMARY KEY,
                    game_creation INTEGER NOT NULL,
                    game_duration INTEGER NOT NULL,
                    game_version TEXT NOT NULL,
                    queue_id INTEGER NOT NULL,
                    target_tier TEXT NOT NULL,
                    winning_team_id INTEGER,
                    fetched_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS participants (
                    match_id TEXT NOT NULL,
                    participant_id INTEGER NOT NULL,
                    puuid TEXT,
                    team_id INTEGER NOT NULL,
                    champion_id INTEGER NOT NULL,
                    champion_name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    win INTEGER NOT NULL CHECK(win IN (0,1)),
                    PRIMARY KEY(match_id, participant_id),
                    FOREIGN KEY(match_id) REFERENCES matches(match_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS bans (
                    match_id TEXT NOT NULL,
                    team_id INTEGER NOT NULL,
                    champion_id INTEGER NOT NULL,
                    pick_turn INTEGER NOT NULL,
                    PRIMARY KEY(match_id, team_id, pick_turn),
                    FOREIGN KEY(match_id) REFERENCES matches(match_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS scrape_players (
                    puuid TEXT PRIMARY KEY,
                    tier TEXT NOT NULL,
                    division TEXT,
                    league_points INTEGER,
                    sampled_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_participants_role_champion
                    ON participants(role, champion_id);
                CREATE INDEX IF NOT EXISTS idx_participants_match_team
                    ON participants(match_id, team_id);
                """
            )
            existing = {row[1] for row in self.connection.execute("PRAGMA table_info(participants)")}
            for name, definition in self.PARTICIPANT_COLUMNS.items():
                if name not in existing:
                    self.connection.execute(f"ALTER TABLE participants ADD COLUMN {name} {definition}")

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def existing_match_ids(self, match_ids: list[str]) -> set[str]:
        found: set[str] = set()
        with self._lock:
            for start in range(0, len(match_ids), 800):
                chunk = match_ids[start:start + 800]
                if not chunk:
                    continue
                marks = ",".join("?" for _ in chunk)
                found.update(str(row[0]) for row in self.connection.execute(
                    f"SELECT match_id FROM matches WHERE match_id IN ({marks})", chunk
                ))
        return found

    def upsert_player(self, entry: Mapping[str, Any], puuid: str, tier: str) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                """INSERT INTO scrape_players(puuid,tier,division,league_points,sampled_at)
                   VALUES(?,?,?,?,?) ON CONFLICT(puuid) DO UPDATE SET
                   tier=excluded.tier, division=excluded.division,
                   league_points=excluded.league_points, sampled_at=excluded.sampled_at""",
                (puuid, tier, str(entry.get("rank", "")), int(entry.get("leaguePoints", 0)),
                 datetime.now(UTC).isoformat()),
            )

    def recent_player_puuids(
        self, limit: int, tiers: tuple[str, ...] | None = None
    ) -> list[str]:
        """Return recently sampled players, optionally limited to target tiers."""
        with self._lock:
            if tiers:
                marks = ",".join("?" for _ in tiers)
                rows = self.connection.execute(
                    f"SELECT puuid FROM scrape_players WHERE tier IN ({marks}) "
                    "ORDER BY sampled_at DESC LIMIT ?",
                    (*tiers, max(1, int(limit))),
                ).fetchall()
            else:
                rows = self.connection.execute(
                    "SELECT puuid FROM scrape_players ORDER BY sampled_at DESC LIMIT ?",
                    (max(1, int(limit)),),
                ).fetchall()
        return [str(row[0]) for row in rows if str(row[0]).strip()]

    def latest_player_sample(self) -> str:
        with self._lock:
            row = self.connection.execute(
                "SELECT MAX(sampled_at) FROM scrape_players"
            ).fetchone()
        return str(row[0] or "") if row else ""

    @staticmethod
    def _runes(participant: Mapping[str, Any]) -> tuple[int, int, str, str]:
        perks = participant.get("perks", {}) or {}
        selections: list[int] = []
        primary = 0
        sub = 0
        for style in perks.get("styles", []) or []:
            description = str(style.get("description", ""))
            style_id = int(style.get("style", 0) or 0)
            if description == "primaryStyle":
                primary = style_id
            elif description == "subStyle":
                sub = style_id
            selections.extend(int(x.get("perk", 0) or 0) for x in style.get("selections", []) or [])
        return primary, sub, json.dumps([x for x in selections if x > 0]), json.dumps(perks.get("statPerks", {}) or {})

    def store_match(self, payload: Mapping[str, Any], target_tier: str, catalog: ChampionCatalog) -> bool:
        metadata = payload.get("metadata", {}) or {}
        info = payload.get("info", {}) or {}
        match_id = str(metadata.get("matchId", ""))
        if not match_id:
            raise ValueError("Missing metadata.matchId")
        teams = info.get("teams", []) or []
        winning_team_id = next((int(t.get("teamId", 0)) for t in teams if bool(t.get("win"))), None)
        duration = int(info.get("gameDuration", 0))

        with self._lock, self.connection:
            cursor = self.connection.execute(
                """INSERT OR IGNORE INTO matches(match_id,game_creation,game_duration,game_version,
                   queue_id,target_tier,winning_team_id,fetched_at) VALUES(?,?,?,?,?,?,?,?)""",
                (match_id, int(info.get("gameCreation", 0)), duration,
                 str(info.get("gameVersion", "")), int(info.get("queueId", 0)),
                 target_tier, winning_team_id, datetime.now(UTC).isoformat()),
            )
            if cursor.rowcount == 0:
                return False

            rows: list[tuple[Any, ...]] = []
            for p in info.get("participants", []) or []:
                role = ROLE_MAP.get(str(p.get("teamPosition") or p.get("individualPosition") or "").upper(), "")
                champion_id = int(p.get("championId", 0) or 0)
                if role not in VALID_ROLES or champion_id <= 0:
                    continue
                primary, sub, rune_json, stat_json = self._runes(p)
                items = [int(p.get(f"item{i}", 0) or 0) for i in range(7)]
                challenges = p.get("challenges", {}) or {}
                rows.append((
                    match_id, int(p.get("participantId", 0)), str(p.get("puuid", "")),
                    int(p.get("teamId", 0)), champion_id,
                    str(p.get("championName", "")) or catalog.name_for_id(champion_id),
                    role, int(bool(p.get("win"))), duration,
                    int(p.get("kills", 0)), int(p.get("deaths", 0)), int(p.get("assists", 0)),
                    int(p.get("goldEarned", 0)), int(p.get("physicalDamageDealtToChampions", 0)),
                    int(p.get("magicDamageDealtToChampions", 0)), int(p.get("trueDamageDealtToChampions", 0)),
                    int(p.get("damageDealtToObjectives", 0)), int(p.get("totalDamageTaken", 0)),
                    int(p.get("damageSelfMitigated", 0)), int(p.get("totalTimeCCDealt", 0)),
                    float(p.get("visionScore", 0) or 0),
                    int(p.get("totalMinionsKilled", 0) or 0),
                    int(p.get("neutralMinionsKilled", 0) or 0),
                    int(p.get("turretTakedowns", 0) or 0),
                    int(p.get("inhibitorTakedowns", 0) or 0),
                    float(p.get("timeCCingOthers", 0) or 0),
                    float(challenges.get("teamDamagePercentage", 0) or 0),
                    float(challenges.get("killParticipation", 0) or 0),
                    float(challenges.get("laneMinionsFirst10Minutes", 0) or 0),
                    float(challenges.get("takedownsFirstXMinutes", 0) or 0),
                    float(challenges.get("soloKills", 0) or 0),
                    float(challenges.get("skillshotsDodged", 0) or 0),
                    float(challenges.get("objectivesStolen", 0) or 0),
                    json.dumps(challenges, separators=(",", ":")),
                    int(p.get("summoner1Id", 0)), int(p.get("summoner2Id", 0)),
                    primary, sub, rune_json, stat_json, json.dumps(items),
                ))
            self.connection.executemany(
                """INSERT OR REPLACE INTO participants(
                   match_id,participant_id,puuid,team_id,champion_id,champion_name,role,win,
                   game_duration_seconds,kills,deaths,assists,gold_earned,
                   physical_damage_to_champions,magic_damage_to_champions,true_damage_to_champions,
                   damage_to_objectives,damage_taken,damage_mitigated,time_cc_dealt,vision_score,
                   total_minions_killed,neutral_minions_killed,turret_takedowns,inhibitor_takedowns,
                   time_ccing_others,team_damage_percentage,kill_participation,lane_minions_first_10,
                   takedowns_first_15,solo_kills,skillshots_dodged,objectives_stolen,challenges_json,
                   summoner1_id,summoner2_id,primary_style_id,sub_style_id,rune_page_json,
                   stat_perks_json,items_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            bans: list[tuple[Any, ...]] = []
            for team in teams:
                for ban in team.get("bans", []) or []:
                    champ = int(ban.get("championId", 0) or 0)
                    if champ > 0:
                        bans.append((match_id, int(team.get("teamId", 0)), champ, int(ban.get("pickTurn", 0))))
            self.connection.executemany(
                "INSERT OR REPLACE INTO bans(match_id,team_id,champion_id,pick_turn) VALUES(?,?,?,?)", bans
            )
        return True

    def counts(self) -> tuple[int, int, int]:
        with self._lock:
            return tuple(int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                         for table in ("matches", "participants", "bans"))  # type: ignore[return-value]


class RiotMatchScraper:
    def __init__(self, settings: ScrapeSettings, client: AsyncRiotClient,
                 database: DraftDatabase, catalog: ChampionCatalog) -> None:
        self.settings = settings
        self.client = client
        self.database = database
        self.catalog = catalog
        self.semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    async def _get(self, route: str, path: str, *, params: Mapping[str, Any] | None = None,
                   allow_404: bool = False) -> Any:
        async with self.semaphore:
            return await self.client.get_json(route, path, params=params, allow_404=allow_404)

    async def _fetch_one_tier(self, tier: str, desired: int) -> list[dict[str, Any]]:
        s = self.settings
        entries: list[dict[str, Any]] = []
        if tier in HIGH_TIERS:
            endpoint = {
                "MASTER": "masterleagues",
                "GRANDMASTER": "grandmasterleagues",
                "CHALLENGER": "challengerleagues",
            }[tier]
            payload = await self._get(s.platform, f"/lol/league/v4/{endpoint}/by-queue/{s.queue}")
            entries = list(payload.get("entries", [])) if isinstance(payload, dict) else []
        else:
            per_division = max(1, math.ceil(desired / max(1, len(s.divisions))))
            for division in s.divisions:
                division_entries: list[dict[str, Any]] = []
                for page in range(1, 21):
                    payload = await self._get(
                        s.platform,
                        f"/lol/league/v4/entries/{s.queue}/{tier}/{division}",
                        params={"page": page},
                    )
                    if not payload:
                        break
                    division_entries.extend(x for x in payload if isinstance(x, dict))
                    if len(division_entries) >= per_division:
                        break
                random.shuffle(division_entries)
                entries.extend(division_entries[:per_division])
        random.shuffle(entries)
        for entry in entries:
            entry["_source_tier"] = tier
        return entries

    async def fetch_ladder_entries(self) -> list[dict[str, Any]]:
        s = self.settings
        tiers = expand_target_tiers(s.target_elo)
        quota = max(1, math.ceil(s.players_per_run / len(tiers)))

        # Fetching is sequential by tier so the platform-route method limits remain
        # easy to reason about. Match-V5 download remains asynchronous afterwards.
        by_tier: dict[str, list[dict[str, Any]]] = {}
        for tier in tiers:
            by_tier[tier] = await self._fetch_one_tier(tier, quota)

        selected: list[dict[str, Any]] = []
        leftovers: list[dict[str, Any]] = []
        for tier in tiers:
            tier_entries = by_tier[tier]
            selected.extend(tier_entries[:quota])
            leftovers.extend(tier_entries[quota:])
        random.shuffle(leftovers)
        selected.extend(leftovers[:max(0, s.players_per_run - len(selected))])
        random.shuffle(selected)
        selected = selected[:s.players_per_run]
        breakdown: dict[str, int] = {}
        for entry in selected:
            source = str(entry.get("_source_tier", s.target_elo))
            breakdown[source] = breakdown.get(source, 0) + 1
        LOGGER.info(
            "Selected %d players for %s from tiers %s.",
            len(selected), s.target_elo,
            ", ".join(f"{tier}={breakdown.get(tier, 0)}" for tier in tiers),
        )
        return selected

    async def resolve_puuid(self, entry: Mapping[str, Any]) -> str | None:
        direct = str(entry.get("puuid", "")).strip()
        if direct:
            return direct
        summoner_id = str(entry.get("summonerId", "")).strip()
        if not summoner_id:
            return None
        payload = await self._get(self.settings.platform,
                                  f"/lol/summoner/v4/summoners/{summoner_id}", allow_404=True)
        return str(payload.get("puuid", "")).strip() if payload else None

    async def collect_puuids(self, entries: list[dict[str, Any]]) -> list[str]:
        async def one(entry: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
            try:
                return entry, await self.resolve_puuid(entry)
            except (RiotAuthenticationError, RiotForbiddenError, RiotRateLimitError):
                raise
            except RiotApiError:
                LOGGER.exception("Could not resolve one ladder player.")
                return entry, None
        values = await asyncio.gather(*(one(entry) for entry in entries))
        output: list[str] = []
        seen: set[str] = set()
        for entry, puuid in values:
            if puuid and puuid not in seen:
                seen.add(puuid)
                output.append(puuid)
                source_tier = str(entry.get("_source_tier", self.settings.target_elo))
                await asyncio.to_thread(self.database.upsert_player, entry, puuid, source_tier)
        return output

    async def collect_match_ids(self, puuids: list[str]) -> list[str]:
        # Optional recency cutoff (epoch ms). Riot already returns newest-first;
        # this only keeps very old games out of every listing scan.
        start_time_ms = (
            int((time.time() - self.settings.matches_window_days * 86400) * 1000)
            if self.settings.matches_window_days > 0
            else None
        )

        async def one(puuid: str) -> list[str]:
            try:
                params: dict[str, Any] = {
                    "queue": self.settings.queue_id, "start": 0,
                    "count": self.settings.matches_per_player,
                }
                if start_time_ms is not None:
                    params["startTime"] = start_time_ms
                payload = await self._get(
                    self.settings.regional_route,
                    f"/lol/match/v5/matches/by-puuid/{puuid}/ids",
                    params=params,
                )
                return [str(x) for x in payload] if isinstance(payload, list) else []
            except (RiotAuthenticationError, RiotForbiddenError, RiotRateLimitError):
                raise
            except RiotApiError:
                LOGGER.exception("Could not fetch match IDs for one player.")
                return []
        lists = await asyncio.gather(*(one(puuid) for puuid in puuids))
        ids = list(dict.fromkeys(match for group in lists for match in group))
        random.shuffle(ids)
        ids = ids[:self.settings.max_matches_per_run]
        existing = await asyncio.to_thread(self.database.existing_match_ids, ids)
        pending = [match for match in ids if match not in existing]
        LOGGER.info("Found %d unique IDs; %d need downloading.", len(ids), len(pending))
        return pending

    def usable(self, payload: Mapping[str, Any]) -> tuple[bool, str]:
        info = payload.get("info", {}) or {}
        if int(info.get("queueId", 0)) != self.settings.queue_id:
            return False, "wrong queue"
        if int(info.get("gameDuration", 0)) < self.settings.min_game_duration_seconds:
            return False, "short game/remake"
        roles: dict[int, set[str]] = {}
        for p in info.get("participants", []) or []:
            role = ROLE_MAP.get(str(p.get("teamPosition") or p.get("individualPosition") or "").upper(), "")
            team = int(p.get("teamId", 0))
            if team and role in VALID_ROLES:
                roles.setdefault(team, set()).add(role)
        if len(roles) != 2 or any(not VALID_ROLES.issubset(x) for x in roles.values()):
            return False, "incomplete role assignment"
        return True, "ok"

    async def fetch_and_store(self, match_ids: list[str]) -> tuple[int, int]:
        stored = 0
        skipped = 0
        lock = asyncio.Lock()

        async def one(match_id: str) -> None:
            nonlocal stored, skipped
            try:
                payload = await self._get(self.settings.regional_route,
                                          f"/lol/match/v5/matches/{match_id}", allow_404=True)
                if not payload:
                    async with lock: skipped += 1
                    return
                ok, reason = self.usable(payload)
                if not ok:
                    LOGGER.debug("Skipping %s: %s", match_id, reason)
                    async with lock: skipped += 1
                    return
                inserted = await asyncio.to_thread(
                    self.database.store_match, payload, self.settings.target_elo, self.catalog
                )
                async with lock:
                    stored += int(inserted)
                    skipped += int(not inserted)
            except (RiotAuthenticationError, RiotForbiddenError, RiotRateLimitError):
                raise
            except Exception:
                LOGGER.exception("Failed processing match %s", match_id)
                async with lock: skipped += 1
        await asyncio.gather(*(one(match_id) for match_id in match_ids))
        return stored, skipped

    async def run(self) -> tuple[int, int]:
        entries = await self.fetch_ladder_entries()
        puuids = await self.collect_puuids(entries)
        ids = await self.collect_match_ids(puuids)
        result = await self.fetch_and_store(ids)
        LOGGER.info("Collection finished: stored=%d skipped=%d totals=%s", *result, self.database.counts())
        return result


@dataclass(slots=True)
class BackgroundWatcherSettings:
    """Adaptive controls for the long-lived match watcher."""

    enabled: bool
    minimum_poll_seconds: int
    maximum_poll_seconds: int
    player_refresh_seconds: int
    players_per_poll: int
    maximum_backoff_seconds: int
    rebuild_analytics_each_batch: bool

    @classmethod
    def from_profile(cls, profile: Mapping[str, Any]) -> "BackgroundWatcherSettings":
        raw = profile.get("background_collector", {})
        minimum = max(15, int(raw.get("minimum_poll_seconds", 45)))
        maximum = max(minimum, int(raw.get("maximum_poll_seconds", 300)))
        return cls(
            enabled=bool(raw.get("enabled", True)),
            minimum_poll_seconds=minimum,
            maximum_poll_seconds=maximum,
            player_refresh_seconds=max(300, int(raw.get("player_refresh_minutes", 30)) * 60),
            players_per_poll=max(5, min(100, int(raw.get("players_per_poll", 20)))),
            maximum_backoff_seconds=max(60, int(raw.get("maximum_backoff_minutes", 20)) * 60),
            rebuild_analytics_each_batch=bool(
                raw.get("rebuild_analytics_each_batch", True)
            ),
        )


def _watcher_emit(
    callback: Callable[[dict[str, Any]], None] | None,
    state: str,
    message: str,
    **extra: Any,
) -> None:
    payload = {"state": state, "message": message, **extra}
    LOGGER.info("Background watcher: %s", message)
    if callback is not None:
        try:
            callback(payload)
        except Exception:
            LOGGER.debug("Background watcher status callback failed.", exc_info=True)


async def _interruptible_wait(
    seconds: float,
    stop_event: threading.Event,
    wake_event: threading.Event | None,
) -> bool:
    """Wait without making shutdown or a user-requested rescan feel delayed."""
    deadline = time.monotonic() + max(0.0, seconds)
    while not stop_event.is_set():
        if wake_event is not None and wake_event.is_set():
            wake_event.clear()
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(0.5, remaining))
    return True


def _scraper_signature(settings: ScrapeSettings, api_key: str) -> tuple[Any, ...]:
    return (
        api_key,
        settings.target_elo,
        settings.platform,
        settings.regional_route,
        settings.queue,
        settings.queue_id,
        tuple(settings.divisions),
        settings.players_per_run,
        settings.matches_per_player,
        settings.max_matches_per_run,
        settings.max_concurrent_requests,
        settings.matches_window_days,
    )


async def run_background_match_watcher(
    stop_event: threading.Event,
    *,
    pause_event: threading.Event | None = None,
    wake_event: threading.Event | None = None,
    database_path: Path = DEFAULT_DB_PATH,
    job_lock: threading.Lock | None = None,
    status_callback: Callable[[dict[str, Any]], None] | None = None,
    catalog: ChampionCatalog | None = None,
    analytics_rebuild_callback: Callable[[], Any] | None = None,
) -> None:
    """Continuously discover and store unseen matches while the app is open.

    This is a long-lived watcher rather than a fixed recurring batch timer. It
    keeps a cached ladder-player roster, polls those players' newest Match-V5
    IDs, filters IDs already present in SQLite, and adaptively slows down when
    no new matches are available. Riot 429 responses are respected immediately
    at request level and also increase the watcher-level delay. Any temporary
    failure is retried automatically; replacing an expired key wakes the same
    service without requiring a new Start action.
    """

    catalog = catalog or ChampionCatalog.load(
        refresh=not (PROJECT_ROOT / "data" / "champion_map.json").exists()
    )
    database = DraftDatabase(database_path)
    client: AsyncRiotClient | None = None
    scraper: RiotMatchScraper | None = None
    signature: tuple[Any, ...] | None = None
    roster_signature: tuple[Any, ...] | None = None
    puuids: list[str] = []
    roster_refreshed_at = 0.0
    empty_cycles = 0
    failure_streak = 0
    poll_cursor = 0
    last_state = ""

    try:
        while not stop_event.is_set():
            profile = load_profile(PROFILE_PATH)
            watcher = BackgroundWatcherSettings.from_profile(profile)
            if not watcher.enabled:
                if last_state != "paused":
                    _watcher_emit(
                        status_callback,
                        "paused",
                        "paused in Settings",
                        busy=False,
                    )
                    last_state = "paused"
                await _interruptible_wait(1.0, stop_event, wake_event)
                continue

            if pause_event is not None and pause_event.is_set():
                if last_state != "draft_paused":
                    _watcher_emit(
                        status_callback,
                        "draft_paused",
                        "paused during Champion Select to protect live responsiveness",
                        busy=False,
                    )
                    last_state = "draft_paused"
                await _interruptible_wait(1.0, stop_event, wake_event)
                continue

            acquired = job_lock.acquire(blocking=False) if job_lock is not None else True
            if not acquired:
                if last_state != "waiting_job":
                    _watcher_emit(
                        status_callback,
                        "waiting_job",
                        "waiting for the current manual data job to finish",
                        busy=False,
                    )
                    last_state = "waiting_job"
                await _interruptible_wait(2.0, stop_event, wake_event)
                continue

            delay = float(watcher.minimum_poll_seconds)
            try:
                api_key = _saved_api_key()
                settings = ScrapeSettings.from_profile(profile)
                current_signature = _scraper_signature(settings, api_key)
                if current_signature != signature or client is None or scraper is None:
                    client = AsyncRiotClient(
                        api_key,
                        timeout=settings.request_timeout_seconds,
                        max_retries=settings.max_retries,
                    )
                    status_payload = await client.get_json(
                        settings.platform, "/lol/status/v4/platform-data"
                    )
                    platform_name = (
                        str(status_payload.get("name", settings.platform))
                        if isinstance(status_payload, Mapping)
                        else settings.platform
                    )
                    scraper = RiotMatchScraper(settings, client, database, catalog)
                    signature = current_signature
                    current_roster_signature = (
                        settings.target_elo,
                        settings.platform,
                        settings.queue,
                        tuple(settings.divisions),
                        settings.players_per_run,
                    )
                    # Cached PUUIDs are useful immediately after a normal restart, but
                    # a changed tier/region/queue requires a fresh ladder roster.
                    if current_roster_signature != roster_signature:
                        puuids = database.recent_player_puuids(
                            settings.players_per_run,
                            expand_target_tiers(settings.target_elo),
                        )
                        poll_cursor = 0
                        roster_refreshed_at = time.monotonic() if puuids else 0.0
                    roster_signature = current_roster_signature
                    _watcher_emit(
                        status_callback,
                        "authenticated",
                        f"authenticated for {settings.platform} ({platform_name})",
                        busy=False,
                    )

                assert scraper is not None and client is not None
                now = time.monotonic()
                roster_due = (
                    not puuids
                    or now - roster_refreshed_at >= watcher.player_refresh_seconds
                )
                if roster_due:
                    _watcher_emit(
                        status_callback,
                        "refreshing_players",
                        "refreshing the tracked ranked-player roster",
                        busy=True,
                    )
                    entries = await scraper.fetch_ladder_entries()
                    refreshed = await scraper.collect_puuids(entries)
                    if refreshed:
                        puuids = refreshed
                        poll_cursor = 0
                        roster_refreshed_at = time.monotonic()
                    elif not puuids:
                        raise RiotApiError("No ranked players could be resolved for the watcher.")

                # Rotate through a bounded slice instead of querying the full
                # roster every cycle. This keeps the service comfortably below the
                # personal-key rolling window while every tracked player is revisited.
                poll_count = min(len(puuids), watcher.players_per_poll)
                if poll_count <= 0:
                    raise RiotApiError("The watcher has no tracked players to poll.")
                selected_puuids = [
                    puuids[(poll_cursor + offset) % len(puuids)]
                    for offset in range(poll_count)
                ]
                poll_cursor = (poll_cursor + poll_count) % len(puuids)
                _watcher_emit(
                    status_callback,
                    "scanning",
                    f"checking {poll_count} of {len(puuids)} tracked players for unseen matches",
                    busy=True,
                )
                pending_ids = await scraper.collect_match_ids(selected_puuids)
                stored = skipped = 0
                if pending_ids:
                    stored, skipped = await scraper.fetch_and_store(pending_ids)

                analytics_rebuilt = False
                if stored > 0 and watcher.rebuild_analytics_each_batch:
                    _watcher_emit(
                        status_callback,
                        "rebuilding",
                        f"stored {stored} new matches; rebuilding analytics",
                        busy=True,
                        stored=stored,
                    )
                    if analytics_rebuild_callback is not None:
                        await asyncio.to_thread(analytics_rebuild_callback)
                    else:
                        from analytics_builder import AnalyticsBuilder

                        await asyncio.to_thread(AnalyticsBuilder(database_path).build_all)
                    analytics_rebuilt = True

                rate_hits, retry_after = client.consume_rate_limit_signal()
                failure_streak = 0
                if stored > 0:
                    empty_cycles = 0
                    delay = float(watcher.minimum_poll_seconds)
                    message = (
                        f"stored {stored} new matches and rebuilt analytics"
                        if analytics_rebuilt
                        else f"stored {stored} new matches"
                    )
                else:
                    empty_cycles += 1
                    delay = min(
                        float(watcher.maximum_poll_seconds),
                        float(watcher.minimum_poll_seconds) * (1.6 ** min(empty_cycles, 6)),
                    )
                    message = "no unseen matches found; continuing to watch"

                if rate_hits:
                    # Request-level Retry-After has already been obeyed. This extra
                    # adaptive delay prevents the watcher from immediately returning
                    # to the edge of the same rolling application/method window.
                    delay = max(delay, retry_after + min(90.0, 5.0 * rate_hits))
                    message += f"; observed {rate_hits} rate-limit response(s)"

                _watcher_emit(
                    status_callback,
                    "watching",
                    message,
                    busy=False,
                    stored=stored,
                    skipped=skipped,
                    analytics_rebuilt=analytics_rebuilt,
                    next_check_seconds=int(round(delay)),
                    rate_limit_hits=rate_hits,
                )
                last_state = "watching"
            except (RiotAuthenticationError, RiotForbiddenError) as exc:
                signature = None
                client = None
                scraper = None
                failure_streak += 1
                delay = min(
                    watcher.maximum_backoff_seconds,
                    max(30.0, 30.0 * (2 ** min(failure_streak - 1, 4))),
                )
                _watcher_emit(
                    status_callback,
                    "invalid_key",
                    f"{exc} The watcher will retry automatically after the key is replaced.",
                    busy=False,
                    next_check_seconds=int(delay),
                )
                last_state = "invalid_key"
            except RiotRateLimitError as exc:
                failure_streak += 1
                delay = min(
                    watcher.maximum_backoff_seconds,
                    max(exc.retry_after + 1.0, 15.0 * (2 ** min(failure_streak, 6))),
                )
                _watcher_emit(
                    status_callback,
                    "rate_limited",
                    f"Riot rate limit persisted; backing off for {int(delay)} seconds",
                    busy=False,
                    next_check_seconds=int(delay),
                )
                last_state = "rate_limited"
            except RiotApiError as exc:
                failure_streak += 1
                delay = min(
                    watcher.maximum_backoff_seconds,
                    max(15.0, 10.0 * (2 ** min(failure_streak, 7))),
                )
                _watcher_emit(
                    status_callback,
                    "backoff",
                    f"temporary Riot/network error: {exc}; retrying automatically",
                    busy=False,
                    next_check_seconds=int(delay),
                )
                last_state = "backoff"
            except Exception as exc:
                failure_streak += 1
                LOGGER.exception("Unexpected background watcher failure.")
                delay = min(
                    watcher.maximum_backoff_seconds,
                    max(30.0, 15.0 * (2 ** min(failure_streak, 6))),
                )
                _watcher_emit(
                    status_callback,
                    "backoff",
                    f"unexpected watcher error: {exc}; retrying automatically",
                    busy=False,
                    next_check_seconds=int(delay),
                )
                last_state = "backoff"
            finally:
                if job_lock is not None and acquired:
                    job_lock.release()

            if await _interruptible_wait(delay, stop_event, wake_event):
                break
    finally:
        database.close()
        _watcher_emit(status_callback, "stopped", "stopped with the application", busy=False)



def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)
    file_handler = logging.FileHandler(DEFAULT_LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def _saved_api_key() -> str:
    reconcile_legacy_api_key()
    load_dotenv(ENV_PATH, override=True)
    import os

    api_key = os.getenv("RIOT_API_KEY", "").strip()
    if not api_key:
        raise RiotAuthenticationError(
            "No RIOT_API_KEY is saved. Open Settings and save a fresh key."
        )
    return api_key


async def validate_saved_api_key() -> dict[str, Any]:
    """Verify the saved key with one lightweight platform-status request."""
    profile = load_profile(PROFILE_PATH)
    settings = ScrapeSettings.from_profile(profile)
    client = AsyncRiotClient(
        _saved_api_key(),
        timeout=settings.request_timeout_seconds,
        max_retries=min(2, settings.max_retries),
    )
    payload = await client.get_json(
        settings.platform, "/lol/status/v4/platform-data"
    )
    name = str(payload.get("name", settings.platform)) if isinstance(payload, Mapping) else settings.platform
    return {"platform": settings.platform, "name": name}


async def run_scrape(database_path: Path = DEFAULT_DB_PATH) -> tuple[int, int]:
    profile = load_profile(PROFILE_PATH)
    settings = ScrapeSettings.from_profile(profile)
    api_key = _saved_api_key()
    catalog = ChampionCatalog.load(refresh=not (PROJECT_ROOT / "data" / "champion_map.json").exists())
    database = DraftDatabase(database_path)
    try:
        client = AsyncRiotClient(api_key, timeout=settings.request_timeout_seconds,
                                 max_retries=settings.max_retries)
        # Fail immediately with a clear message before ladder discovery and match
        # collection if the saved key has expired or was copied incorrectly.
        status_payload = await client.get_json(
            settings.platform, "/lol/status/v4/platform-data"
        )
        platform_name = (
            str(status_payload.get("name", settings.platform))
            if isinstance(status_payload, Mapping) else settings.platform
        )
        LOGGER.info("Riot API key accepted for %s (%s).", settings.platform, platform_name)
        return await RiotMatchScraper(settings, client, database, catalog).run()
    finally:
        database.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect ranked League matches into SQLite.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--build-analytics", action="store_true")
    parser.add_argument(
        "--validate-key", action="store_true",
        help="Validate config.env without downloading matches.",
    )
    args = parser.parse_args()
    configure_logging(args.verbose)
    try:
        if args.validate_key:
            result = asyncio.run(validate_saved_api_key())
            LOGGER.info(
                "Riot API key is valid for %s (%s).",
                result["platform"], result["name"],
            )
            return
        asyncio.run(run_scrape(args.database))
        if args.build_analytics:
            from analytics_builder import AnalyticsBuilder
            AnalyticsBuilder(args.database).build_all()
    except (RiotAuthenticationError, RiotForbiddenError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(3) from exc
    except KeyboardInterrupt:
        LOGGER.info("Cancelled.")


if __name__ == "__main__":
    main()
