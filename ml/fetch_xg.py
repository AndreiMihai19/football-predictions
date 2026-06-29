"""Fetch xG per match from understat.com via understatapi."""

import csv
import json
import os
import time

import understatapi

ML_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(ML_DIR, "xg_matches.csv")

LEAGUE_MAP = {
    "EPL":        "PL",
    "La_Liga":    "PD",
    "Bundesliga": "BL1",
    "Serie_A":    "SA",
    "Ligue_1":    "FL1",
}

SEASONS = ["2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]

SLEEP_BETWEEN_LEAGUES = 2.0
SLEEP_BETWEEN_SEASONS = 0.5


def fetch_all_xg():
    rows = []
    total_leagues = len(LEAGUE_MAP)

    with understatapi.UnderstatClient() as client:
        for i, (understat_league, comp_code) in enumerate(LEAGUE_MAP.items()):
            print(f"\n[{i+1}/{total_leagues}] {understat_league} ({comp_code})")

            for season in SEASONS:
                try:
                    matches = client.league(understat_league).get_match_data(season=season)
                    finished = [m for m in matches if m.get("isResult")]
                    print(f"  {season}: {len(finished)} meciuri finite")

                    for m in finished:
                        xg_h = m["xG"].get("h")
                        xg_a = m["xG"].get("a")
                        if xg_h is None or xg_a is None:
                            continue

                        fc = m.get("forecast", {})
                        rows.append({
                            "date":                m["datetime"][:10],
                            "home_team_understat": m["h"]["title"],
                            "away_team_understat": m["a"]["title"],
                            "home_xg":             float(xg_h),
                            "away_xg":             float(xg_a),
                            "home_goals_us":       int(m["goals"]["h"]),
                            "away_goals_us":       int(m["goals"]["a"]),
                            "forecast_w":          float(fc.get("w", 0)),
                            "forecast_d":          float(fc.get("d", 0)),
                            "forecast_l":          float(fc.get("l", 0)),
                            "league_understat":    understat_league,
                            "comp":                comp_code,
                            "season":              int(season),
                        })

                    time.sleep(SLEEP_BETWEEN_SEASONS)

                except Exception as e:
                    print(f"  WARN: {understat_league} {season} — {e}")

            if i < total_leagues - 1:
                time.sleep(SLEEP_BETWEEN_LEAGUES)

    return rows


def save_csv(rows):
    if not rows:
        print("Niciun rand de salvat.")
        return

    fieldnames = [
        "date", "home_team_understat", "away_team_understat",
        "home_xg", "away_xg", "home_goals_us", "away_goals_us",
        "forecast_w", "forecast_d", "forecast_l",
        "league_understat", "comp", "season",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSalvat: {OUT_CSV}")
    print(f"Total randuri: {len(rows)}")

    from collections import Counter
    per_league = Counter(r["comp"] for r in rows)
    for comp, cnt in sorted(per_league.items()):
        print(f"  {comp}: {cnt} meciuri")


if __name__ == "__main__":
    print("FETCH xG (understat)")

    if os.path.exists(OUT_CSV):
        print(f"\nATENTIE: {OUT_CSV} exista deja.")
        resp = input("Suprascrie? (y/N): ").strip().lower()
        if resp != "y":
            print("Anulat.")
            exit(0)

    rows = fetch_all_xg()
    save_csv(rows)
