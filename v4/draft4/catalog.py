"""Champion catalog: Data Dragon metadata + coach-editable seed roles/traits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ROLES = ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")
ROLE_CODES = {"T": 0, "J": 1, "M": 2, "A": 3, "S": 4}
TRAITS = ("engage", "disengage", "peel", "poke", "pick", "dive", "waveclear",
          "scaling", "early", "splitpush", "frontline", "teamfight")


@dataclass(frozen=True, slots=True)
class Champion:
    idx: int  # dense engine index
    id: int  # Riot numeric champion id
    key: str  # Data Dragon id, e.g. "MonkeyKing"
    name: str
    roles: tuple[int, ...]  # seed roles, main first
    traits: frozenset[str]
    damage: str  # AD / AP / MIX
    ranged: bool
    image: str


class Catalog:
    def __init__(self, champions: list[Champion], version: str) -> None:
        self.champions = champions
        self.version = version
        self.by_id = {c.id: c for c in champions}
        self._by_name = {}
        for c in champions:
            for alias in (c.key, c.name, str(c.id)):
                self._by_name[_norm(alias)] = c

    def __len__(self) -> int:
        return len(self.champions)

    def resolve(self, value: int | str) -> Champion | None:
        if isinstance(value, int):
            return self.by_id.get(value)
        return self._by_name.get(_norm(value))

    def idx_of_id(self, champion_id: int) -> int:
        return self.by_id[champion_id].idx


def _norm(value: str) -> str:
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())


def load_catalog(data_dir: Path = DATA_DIR) -> Catalog:
    dd = json.loads((data_dir / "ddragon_champion.json").read_text(encoding="utf-8"))
    version = str(dd.get("version", ""))
    seed: dict[str, tuple[tuple[int, ...], frozenset[str], str]] = {}
    for line in (data_dir / "champion_seed.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, roles, traits, damage = (part.strip() for part in line.split("|"))
        unknown = set(traits.split()) - set(TRAITS)
        if unknown:
            raise ValueError(f"champion_seed.txt: unknown traits {unknown} for {key}")
        seed[key] = (tuple(ROLE_CODES[r] for r in roles.split()), frozenset(traits.split()), damage)
    champions = []
    for key, entry in sorted(dd["data"].items(), key=lambda kv: kv[1]["name"]):
        roles, traits, damage = seed.get(key, ((), frozenset(), "MIX"))
        if not roles:  # new champion missing from seed: guess from Data Dragon class tags
            roles = _guess_roles(entry["tags"])
        champions.append(Champion(
            idx=len(champions), id=int(entry["key"]), key=key, name=entry["name"], roles=roles,
            traits=traits, damage=damage, ranged=entry["stats"]["attackrange"] >= 325,
            image=f"https://ddragon.leagueoflegends.com/cdn/{version}/img/champion/{entry['image']['full']}",
        ))
    return Catalog(champions, version)


def _guess_roles(tags: list[str]) -> tuple[int, ...]:
    table = {"Marksman": 3, "Support": 4, "Mage": 2, "Assassin": 2, "Tank": 0, "Fighter": 0}
    return tuple(dict.fromkeys(table[t] for t in tags if t in table)) or (0,)


def refresh(data_dir: Path = DATA_DIR) -> str:
    """Download the latest Data Dragon champion list. New champions get guessed roles until added to the seed."""
    import requests
    version = requests.get("https://ddragon.leagueoflegends.com/api/versions.json", timeout=20).json()[0]
    r = requests.get(f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json", timeout=30)
    r.raise_for_status()
    (data_dir / "ddragon_champion.json").write_text(r.text, encoding="utf-8")
    cat = load_catalog(data_dir)
    seeded = {line.split("|")[0] for line in (data_dir / "champion_seed.txt").read_text().splitlines() if "|" in line}
    missing = [c.key for c in cat.champions if c.key not in seeded]
    return f"{version}: {len(cat)} champions" + (f"; add to champion_seed.txt: {', '.join(missing)}" if missing else "")


if __name__ == "__main__":
    print(refresh())
