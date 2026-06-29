"""Fetch historical matches from openfootball/football.json on GitHub."""

import hashlib
import requests
import pandas as pd

LEAGUE_MAP = {
    "en.1": "PL",
    "es.1": "PD",
    "de.1": "BL1",
    "it.1": "SA",
    "fr.1": "FL1",
}

SEASON_FOLDERS = [
    "2010-11", "2011-12", "2012-13", "2013-14", "2014-15",
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
]

RAW_BASE = "https://raw.githubusercontent.com/openfootball/football.json/master/{season}/{file}.json"

TEAM_ID_MAP: dict[str, int] = {}


def team_id(name: str) -> int:
    # Hash-based IDs start at 100000 to avoid colliding with football-data.org IDs.
    if name not in TEAM_ID_MAP:
        h = int(hashlib.md5(name.encode()).hexdigest(), 16) % 900000 + 100000
        while h in TEAM_ID_MAP.values():
            h += 1
        TEAM_ID_MAP[name] = h
    return TEAM_ID_MAP[name]


def season_to_year(folder: str) -> int:
    return int(folder.split("-")[0])


def parse_matchday(round_str: str) -> int:
    try:
        return int(round_str.replace("Matchday", "").strip())
    except (ValueError, AttributeError):
        return 0


def fetch_file(season_folder: str, file_key: str, comp_code: str) -> list[dict]:
    url = RAW_BASE.format(season=season_folder, file=file_key)
    try:
        resp = requests.get(url, timeout=15)
    except requests.RequestException as e:
        print(f"    SKIP {comp_code} {season_folder}: {e}")
        return []

    if resp.status_code != 200:
        return []

    try:
        data = resp.json()
    except ValueError:
        return []

    matches = data.get("matches", [])
    year = season_to_year(season_folder)
    rows = []
    base_id = abs(hash(f"{comp_code}{season_folder}")) % 10_000_000

    for i, m in enumerate(matches):
        try:
            ft = m.get("score", {}).get("ft")
            if not ft or len(ft) < 2:
                continue

            hg, ag = int(ft[0]), int(ft[1])
            home = str(m.get("team1", "")).strip()
            away = str(m.get("team2", "")).strip()
            date_str = str(m.get("date", "")).strip()

            if not home or not away or not date_str:
                continue

            date_parsed = pd.to_datetime(date_str, errors="coerce")
            if pd.isna(date_parsed):
                continue

            result = 1 if hg > ag else (0 if hg < ag else 2)

            rows.append({
                "match_id":     base_id + i,
                "competition":  comp_code,
                "season":       year,
                "matchday":     parse_matchday(m.get("round", "")),
                "date":         date_parsed.strftime("%Y-%m-%d"),
                "home_team":    home,
                "away_team":    away,
                "home_team_id": team_id(home),
                "away_team_id": team_id(away),
                "home_goals":   hg,
                "away_goals":   ag,
                "result":       result,
                "source":       "openfoot",
            })
        except (ValueError, TypeError, KeyError):
            continue

    return rows


def fetch_all() -> pd.DataFrame:
    all_rows = []
    for season_folder in SEASON_FOLDERS:
        season_rows = 0
        for file_key, comp_code in LEAGUE_MAP.items():
            rows = fetch_file(season_folder, file_key, comp_code)
            season_rows += len(rows)
            all_rows.extend(rows)
        if season_rows:
            print(f"  {season_folder}: {season_rows} meciuri")

    if not all_rows:
        print("Nicio data descarcata!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["competition", "season", "date", "home_team", "away_team"])
    df = df.sort_values(["competition", "season", "date"]).reset_index(drop=True)

    dist = df["result"].value_counts().sort_index()
    print(f"\nTotal: {len(df)} meciuri")
    print(f"  Away(0): {dist.get(0,0)}  Home(1): {dist.get(1,0)}  Draw(2): {dist.get(2,0)}")
    print(f"  Ligi: {df['competition'].nunique()}  |  Sezoane: {df['season'].nunique()}")

    return df


if __name__ == "__main__":
    print("FETCH OpenFootball")

    df = fetch_all()

    if df.empty:
        print("ERROR: nicio data descarcata.")
        raise SystemExit(1)

    out = "openfoot_matches.csv"
    df.to_csv(out, index=False)
    print(f"\nSalvat: {out} ({len(df)} randuri)")
