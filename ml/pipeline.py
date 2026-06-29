"""Data pipeline: fetch matches, build features, persist tensors."""

import os
import re
import json
import time
import datetime
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
import joblib

load_dotenv()

API_KEY = os.environ.get("FOOTBALL_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "FOOTBALL_API_KEY not set. Export it or add to ml/.env file."
    )

COMPETITIONS = {
    "PL":  "Premier League",
    "SA":  "Serie A",
    "PD":  "La Liga",
    "BL1": "Bundesliga",
    "FL1": "Ligue 1",
}

TODAY   = datetime.date.today().isoformat()
SEASONS = [2023, 2024, 2025]

HEADERS     = {"X-Auth-Token": API_KEY}
BASE_URL    = "https://api.football-data.org/v4"
MAX_RETRIES = 3


def fetch_one(url, params, label):
    for attempt in range(1, MAX_RETRIES + 1):
        resp = requests.get(url, headers=HEADERS, params=params, timeout=20)

        if resp.status_code == 200:
            return resp

        if resp.status_code == 429:
            if attempt == MAX_RETRIES:
                print(f"  429 — rate limit dupa {MAX_RETRIES} incercari. Skip {label}.")
                return None
            msg   = resp.json().get("message", "")
            match = re.search(r"Wait (\d+) seconds", msg)
            wait  = (int(match.group(1)) + 2) if match else 62
            print(f"  429 — astept {wait}s si reincerc... (tentativa {attempt}/3)")
            time.sleep(wait)
        else:
            print(f"  Eroare {resp.status_code} pentru {label}")
            return None

    return None


def fetch_matches():
    print(f"Data curenta: {TODAY}")
    print(f"Sezoane: {SEASONS}")
    all_matches = []

    for i, (code, name) in enumerate(COMPETITIONS.items()):
        if i > 0:
            time.sleep(2)

        for season in SEASONS:
            url    = f"{BASE_URL}/competitions/{code}/matches"
            params = {"season": season, "status": "FINISHED"}
            label  = f"{name} {season}"

            print(f"  Descarc {label}...")
            resp = fetch_one(url, params, label)
            if resp is None:
                continue

            matches = resp.json().get("matches", [])
            print(f"  {len(matches)} meciuri gasite")

            for m in matches:
                if m["score"]["fullTime"]["home"] is None:
                    continue

                home_goals = m["score"]["fullTime"]["home"]
                away_goals = m["score"]["fullTime"]["away"]

                if home_goals > away_goals:
                    result = 1
                elif home_goals < away_goals:
                    result = 0
                else:
                    result = 2

                all_matches.append({
                    "match_id":     m["id"],
                    "competition":  code,
                    "season":       season,
                    "matchday":     m.get("matchday", 0),
                    "date":         m["utcDate"][:10],
                    "home_team":    m["homeTeam"]["name"],
                    "away_team":    m["awayTeam"]["name"],
                    "home_team_id": m["homeTeam"]["id"],
                    "away_team_id": m["awayTeam"]["id"],
                    "home_goals":   home_goals,
                    "away_goals":   away_goals,
                    "result":       result,
                })

    df = pd.DataFrame(all_matches)
    dist = df["result"].value_counts().sort_index()
    print(f"\nTotal meciuri (inclusiv egaluri): {len(df)}")
    print(f"  Away(0): {dist.get(0,0)}  Home(1): {dist.get(1,0)}  Draw(2): {dist.get(2,0)}")

    df["date"]  = pd.to_datetime(df["date"])
    filtered    = df[df["date"] <= pd.Timestamp(TODAY)].copy()
    print(f"  Meciuri dupa filtrare data (<= {TODAY}): {len(filtered)}")
    filtered["date"] = filtered["date"].dt.strftime("%Y-%m-%d")
    return filtered


def load_combined_matches(api_df: pd.DataFrame) -> pd.DataFrame:
    openfoot_path = os.path.join(os.path.dirname(__file__), "openfoot_matches.csv")
    if not os.path.exists(openfoot_path):
        print("  openfoot_matches.csv negasit — folosesc doar datele API.")
        return api_df

    of_df = pd.read_csv(openfoot_path)
    of_df["date"] = pd.to_datetime(of_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    of_df = of_df.dropna(subset=["date"])

    for col in ["match_id", "competition", "season", "matchday", "date",
                "home_team", "away_team", "home_team_id", "away_team_id",
                "home_goals", "away_goals", "result"]:
        if col not in of_df.columns:
            of_df[col] = 0

    of_df = of_df[api_df.columns.intersection(of_df.columns)]

    for col in api_df.columns:
        if col not in of_df.columns:
            of_df[col] = 0

    combined = pd.concat([api_df, of_df[api_df.columns]], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["competition", "season", "date", "home_team", "away_team"]
    )
    combined = combined.sort_values("date").reset_index(drop=True)

    print(f"  Date API:          {len(api_df):>6} meciuri")
    print(f"  Date OpenFootball: {len(of_df):>6} meciuri")
    print(f"  Dupa deduplicare:  {len(combined):>6} meciuri")

    return combined


def compute_team_form(df, team_id_col, goals_for_col, goals_against_col,
                      win_col, draw_col, n=5):
    df      = df.sort_values("date").copy()
    t_pts   = {}
    t_gf    = {}
    t_ga    = {}
    prefix  = team_id_col.split("_")[0]
    pts_col = f"{prefix}_form_pts"
    gf_col  = f"{prefix}_form_gf"
    ga_col  = f"{prefix}_form_ga"
    pts_l, gf_l, ga_l = [], [], []

    for _, row in df.iterrows():
        tid = row[team_id_col]
        pts_l.append(np.mean(t_pts.get(tid, [])[-n:]) if t_pts.get(tid) else 1.0)
        gf_l.append( np.mean(t_gf.get(tid, [])[-n:])  if t_gf.get(tid)  else 1.0)
        ga_l.append( np.mean(t_ga.get(tid, [])[-n:])  if t_ga.get(tid)  else 1.0)

        pts = 3 if row[win_col] else (1 if row[draw_col] else 0)
        t_pts.setdefault(tid, []).append(pts)
        t_gf.setdefault(tid, []).append(row[goals_for_col])
        t_ga.setdefault(tid, []).append(row[goals_against_col])

    df[pts_col] = pts_l
    df[gf_col]  = gf_l
    df[ga_col]  = ga_l
    return df, [pts_col, gf_col, ga_col]


def compute_ranks(df):
    home_ranks = []
    away_ranks = []
    team_pts   = {}

    for _, row in df.iterrows():
        comp = row["competition"]
        seas = row["season"]
        htid = row["home_team_id"]
        atid = row["away_team_id"]
        res  = row["result"]

        cs_pts = {
            tid: pts
            for (c, s, tid), pts in team_pts.items()
            if c == comp and s == seas
        }

        if len(cs_pts) >= 2:
            ranked  = sorted(cs_pts.items(), key=lambda x: x[1], reverse=True)
            ranks   = {tid: i + 1 for i, (tid, _) in enumerate(ranked)}
            n_teams = len(ranked)
            home_ranks.append(ranks.get(htid, n_teams + 1))
            away_ranks.append(ranks.get(atid, n_teams + 1))
        else:
            home_ranks.append(1)
            away_ranks.append(1)

        key_h = (comp, seas, htid)
        key_a = (comp, seas, atid)
        team_pts[key_h] = team_pts.get(key_h, 0) + (3 if res == 1 else (1 if res == 2 else 0))
        team_pts[key_a] = team_pts.get(key_a, 0) + (3 if res == 0 else (1 if res == 2 else 0))

    return home_ranks, away_ranks


def compute_venue_goals(df):
    gf_home = {}
    gf_away = {}
    hgh_l, hga_l, agh_l, aga_l = [], [], [], []

    for _, row in df.iterrows():
        htid = row["home_team_id"]
        atid = row["away_team_id"]

        hgh_l.append(np.mean(gf_home[htid]) if gf_home.get(htid) else 1.0)
        hga_l.append(np.mean(gf_away[htid]) if gf_away.get(htid) else 1.0)
        agh_l.append(np.mean(gf_home[atid]) if gf_home.get(atid) else 1.0)
        aga_l.append(np.mean(gf_away[atid]) if gf_away.get(atid) else 1.0)

        gf_home.setdefault(htid, []).append(row["home_goals"])
        gf_away.setdefault(atid, []).append(row["away_goals"])

    return hgh_l, hga_l, agh_l, aga_l


def compute_h2h(df):
    h2h_db      = {}
    home_w_list = []
    away_w_list = []
    draw_list   = []

    for _, row in df.iterrows():
        htid = row["home_team_id"]
        atid = row["away_team_id"]
        key  = (min(htid, atid), max(htid, atid))
        last = h2h_db.get(key, [])[-3:]

        if not last:
            home_w_list.append(1.0)
            away_w_list.append(1.0)
            draw_list.append(0.0)
        else:
            hw = aw = dw = 0
            for _, h_home_id, h_result in last:
                if h_result == 2:
                    dw += 1
                elif h_home_id == htid:
                    if h_result == 1: hw += 1
                    else:             aw += 1
                else:
                    if h_result == 1: aw += 1
                    else:             hw += 1
            home_w_list.append(float(hw))
            away_w_list.append(float(aw))
            draw_list.append(float(dw))

        h2h_db.setdefault(key, []).append((row["date"], htid, row["result"]))

    return home_w_list, away_w_list, draw_list


def compute_elo(df, k=20, default_elo=1500.0):
    elo = {}
    home_elo_l, away_elo_l, elo_diff_l = [], [], []

    for _, row in df.iterrows():
        h = row["home_team_id"]
        a = row["away_team_id"]
        ra = elo.get(h, default_elo)
        rb = elo.get(a, default_elo)

        home_elo_l.append(ra)
        away_elo_l.append(rb)
        elo_diff_l.append(ra - rb)

        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        eb = 1.0 - ea
        result = row["result"]
        sa = 1.0 if result == 1 else (0.5 if result == 2 else 0.0)
        sb = 1.0 - sa

        elo[h] = ra + k * (sa - ea)
        elo[a] = rb + k * (sb - eb)

    return home_elo_l, away_elo_l, elo_diff_l, elo


def compute_venue_form(df, n=5):
    venue_pts = {}
    hph_l, hpa_l, aph_l, apa_l = [], [], [], []

    for _, row in df.iterrows():
        htid = row["home_team_id"]
        atid = row["away_team_id"]
        res  = row["result"]

        hph_l.append(np.mean(venue_pts.get((htid, 'H'), [])[-n:]) if venue_pts.get((htid, 'H')) else 1.0)
        hpa_l.append(np.mean(venue_pts.get((htid, 'A'), [])[-n:]) if venue_pts.get((htid, 'A')) else 1.0)
        aph_l.append(np.mean(venue_pts.get((atid, 'H'), [])[-n:]) if venue_pts.get((atid, 'H')) else 1.0)
        apa_l.append(np.mean(venue_pts.get((atid, 'A'), [])[-n:]) if venue_pts.get((atid, 'A')) else 1.0)

        h_pts = 3 if res == 1 else (1 if res == 2 else 0)
        a_pts = 3 if res == 0 else (1 if res == 2 else 0)
        venue_pts.setdefault((htid, 'H'), []).append(h_pts)
        venue_pts.setdefault((atid, 'A'), []).append(a_pts)

    return hph_l, hpa_l, aph_l, apa_l


def compute_rest_days(df, default_days=7):
    last_match_date = {}
    home_rest_l, away_rest_l = [], []

    for _, row in df.iterrows():
        htid = row["home_team_id"]
        atid = row["away_team_id"]
        match_date = pd.Timestamp(row["date"])

        if htid in last_match_date:
            home_rest_l.append((match_date - last_match_date[htid]).days)
        else:
            home_rest_l.append(default_days)

        if atid in last_match_date:
            away_rest_l.append((match_date - last_match_date[atid]).days)
        else:
            away_rest_l.append(default_days)

        last_match_date[htid] = match_date
        last_match_date[atid] = match_date

    return home_rest_l, away_rest_l


def compute_goal_trends(df, recent=5, older=5):
    team_goals = {}
    home_trend_l, away_trend_l = [], []

    def trend(tid):
        hist = team_goals.get(tid, [])
        if len(hist) < recent + older:
            return 0.0
        recent_avg = np.mean(hist[-recent:])
        older_avg  = np.mean(hist[-(recent + older):-recent])
        return float(recent_avg - older_avg)

    for _, row in df.iterrows():
        htid = row["home_team_id"]
        atid = row["away_team_id"]

        home_trend_l.append(trend(htid))
        away_trend_l.append(trend(atid))

        team_goals.setdefault(htid, []).append(row["home_goals"])
        team_goals.setdefault(atid, []).append(row["away_goals"])

    return home_trend_l, away_trend_l


def compute_exp_form(df, decay=0.7, n=7):
    team_pts_hist = {}
    home_exp_l, away_exp_l = [], []

    def ewm_pts(history):
        if not history:
            return 1.0
        recent = history[-n:]
        weights = np.array([decay ** (len(recent) - 1 - i) for i in range(len(recent))])
        return float(np.dot(weights, recent) / weights.sum())

    for _, row in df.iterrows():
        htid = row["home_team_id"]
        atid = row["away_team_id"]

        home_exp_l.append(ewm_pts(team_pts_hist.get(htid, [])))
        away_exp_l.append(ewm_pts(team_pts_hist.get(atid, [])))

        h_pts = 3 if row["result"] == 1 else (1 if row["result"] == 2 else 0)
        a_pts = 3 if row["result"] == 0 else (1 if row["result"] == 2 else 0)
        team_pts_hist.setdefault(htid, []).append(h_pts)
        team_pts_hist.setdefault(atid, []).append(a_pts)

    return home_exp_l, away_exp_l


def compute_win_rate(df, n=5):
    team_results = {}
    home_wr_l, away_wr_l = [], []
    home_dr_l, away_dr_l = [], []

    def rate(history, value):
        if not history:
            return 0.33
        recent = history[-n:]
        return float(sum(1 for r in recent if r == value) / len(recent))

    for _, row in df.iterrows():
        htid = row["home_team_id"]
        atid = row["away_team_id"]
        res  = row["result"]

        h_hist = team_results.get(htid, [])
        a_hist = team_results.get(atid, [])

        home_wr_l.append(rate(h_hist, "W"))
        away_wr_l.append(rate(a_hist, "W"))
        home_dr_l.append(rate(h_hist, "D"))
        away_dr_l.append(rate(a_hist, "D"))

        h_res = "W" if res == 1 else ("D" if res == 2 else "L")
        a_res = "W" if res == 0 else ("D" if res == 2 else "L")
        team_results.setdefault(htid, []).append(h_res)
        team_results.setdefault(atid, []).append(a_res)

    return home_wr_l, away_wr_l, home_dr_l, away_dr_l


def compute_defense_variance(df, n=7):
    team_ga = {}
    home_var_l, away_var_l = [], []

    def ga_var(history):
        if len(history) < 3:
            return 1.0
        recent = history[-n:]
        return float(np.var(recent))

    for _, row in df.iterrows():
        htid = row["home_team_id"]
        atid = row["away_team_id"]

        home_var_l.append(ga_var(team_ga.get(htid, [])))
        away_var_l.append(ga_var(team_ga.get(atid, [])))

        team_ga.setdefault(htid, []).append(row["away_goals"])
        team_ga.setdefault(atid, []).append(row["home_goals"])

    return home_var_l, away_var_l


def compute_rank_trajectory(df, window=5):
    team_rank_hist = {}
    home_traj_l, away_traj_l = [], []

    for _, row in df.iterrows():
        htid = row["home_team_id"]
        atid = row["away_team_id"]
        hr   = row["home_rank"]
        ar   = row["away_rank"]

        h_hist = team_rank_hist.get(htid, [])
        a_hist = team_rank_hist.get(atid, [])

        if len(h_hist) >= window:
            home_traj_l.append(float(h_hist[-window] - hr))
        else:
            home_traj_l.append(0.0)

        if len(a_hist) >= window:
            away_traj_l.append(float(a_hist[-window] - ar))
        else:
            away_traj_l.append(0.0)

        team_rank_hist.setdefault(htid, []).append(hr)
        team_rank_hist.setdefault(atid, []).append(ar)

    return home_traj_l, away_traj_l


def compute_league_relative_elo(df):
    league_elo_sum   = {}
    league_elo_count = {}
    team_last_elo    = {}

    home_rel_l, away_rel_l = [], []

    for _, row in df.iterrows():
        comp = row["competition"]
        htid = row["home_team_id"]
        atid = row["away_team_id"]
        h_elo = row["home_elo"]
        a_elo = row["away_elo"]

        total = league_elo_sum.get(comp, 0.0)
        count = league_elo_count.get(comp, 0)
        league_mean = (total / count) if count >= 5 else 1500.0

        home_rel_l.append(h_elo - league_mean)
        away_rel_l.append(a_elo - league_mean)

        for tid, elo in [(htid, h_elo), (atid, a_elo)]:
            old = team_last_elo.get(tid)
            if old is not None:
                league_elo_sum[comp] = league_elo_sum.get(comp, 0.0) - old + elo
            else:
                league_elo_sum[comp]   = league_elo_sum.get(comp, 0.0) + elo
                league_elo_count[comp] = league_elo_count.get(comp, 0) + 1
            team_last_elo[tid] = elo

    return home_rel_l, away_rel_l


def compute_draw_signals(df):
    goals_sim   = np.abs(df["home_form_gf"] - df["away_form_gf"])
    pts_close   = np.abs(df["home_form_pts"] - df["away_form_pts"])
    avg_concede = (df["home_form_ga"] + df["away_form_ga"]) / 2.0

    return goals_sim.values, pts_close.values, avg_concede.values


def build_features(df):
    print("\nConstruiesc features...")
    df = df.sort_values(["competition", "season", "date"]).copy().reset_index(drop=True)

    df["home_win"] = (df["result"] == 1).astype(int)
    df["away_win"] = (df["result"] == 0).astype(int)
    df["is_draw"]  = (df["result"] == 2).astype(int)

    df, home_cols = compute_team_form(df, "home_team_id", "home_goals", "away_goals", "home_win", "is_draw")
    df, away_cols = compute_team_form(df, "away_team_id", "away_goals", "home_goals", "away_win", "is_draw")

    df["form_pts_diff"] = df["home_form_pts"] - df["away_form_pts"]
    df["form_gf_diff"]  = df["home_form_gf"]  - df["away_form_gf"]
    df["form_ga_diff"]  = df["home_form_ga"]  - df["away_form_ga"]

    home_ranks, away_ranks = compute_ranks(df)
    df["home_rank"] = home_ranks
    df["away_rank"] = away_ranks
    df["rank_diff"] = df["away_rank"] - df["home_rank"]

    hgh, hga, agh, aga = compute_venue_goals(df)
    df["home_goals_scored_home"] = hgh
    df["home_goals_scored_away"] = hga
    df["away_goals_scored_home"] = agh
    df["away_goals_scored_away"] = aga

    hw, aw, dw = compute_h2h(df)
    df["h2h_home_wins"] = hw
    df["h2h_away_wins"] = aw
    df["h2h_draws"]     = dw

    goals_sim, pts_close, avg_concede = compute_draw_signals(df)
    df["goals_scored_similarity"] = goals_sim
    df["form_pts_closeness"]      = pts_close
    df["avg_goals_conceded"]      = avg_concede

    home_elo, away_elo, elo_diff, final_elos = compute_elo(df)
    df["home_elo"] = home_elo
    df["away_elo"] = away_elo
    df["elo_diff"] = elo_diff

    try:
        with open("elo_ratings.json", "w") as f:
            json.dump({str(k): float(v) for k, v in final_elos.items()}, f, indent=2)
        print(f"  ELO ratings saved: {len(final_elos)} teams -> elo_ratings.json")
    except Exception as e:
        print(f"  WARN: failed to save elo_ratings.json: {e}")

    hph, hpa, aph, apa = compute_venue_form(df)
    df["home_form_pts_home"] = hph
    df["home_form_pts_away"] = hpa
    df["away_form_pts_home"] = aph
    df["away_form_pts_away"] = apa

    home_rest, away_rest = compute_rest_days(df)
    df["days_since_last_match_home"] = home_rest
    df["days_since_last_match_away"] = away_rest

    home_trend, away_trend = compute_goal_trends(df)
    df["home_goals_trend"] = home_trend
    df["away_goals_trend"] = away_trend

    home_exp, away_exp = compute_exp_form(df)
    df["home_form_pts_exp"] = home_exp
    df["away_form_pts_exp"] = away_exp

    home_wr, away_wr, home_dr, away_dr = compute_win_rate(df)
    df["home_win_rate"]  = home_wr
    df["away_win_rate"]  = away_wr
    df["home_draw_rate"] = home_dr
    df["away_draw_rate"] = away_dr
    df["win_rate_diff"]  = df["home_win_rate"] - df["away_win_rate"]

    home_ga_var, away_ga_var = compute_defense_variance(df)
    df["home_defense_variance"] = home_ga_var
    df["away_defense_variance"] = away_ga_var

    home_traj, away_traj = compute_rank_trajectory(df)
    df["home_rank_trajectory"] = home_traj
    df["away_rank_trajectory"] = away_traj
    df["rank_traj_diff"] = df["home_rank_trajectory"] - df["away_rank_trajectory"]

    home_rel_elo, away_rel_elo = compute_league_relative_elo(df)
    df["home_elo_vs_league"] = home_rel_elo
    df["away_elo_vs_league"] = away_rel_elo
    df["elo_vs_league_diff"] = df["home_elo_vs_league"] - df["away_elo_vs_league"]

    df["home_advantage"] = 1

    comp_dummies = pd.get_dummies(df["competition"], prefix="comp")
    df = pd.concat([df, comp_dummies], axis=1)
    comp_cols = list(comp_dummies.columns)

    feature_cols = (
        home_cols + away_cols +
        ["form_pts_diff", "form_gf_diff", "form_ga_diff"] +
        ["home_rank", "away_rank", "rank_diff"] +
        ["home_goals_scored_home", "home_goals_scored_away",
         "away_goals_scored_home", "away_goals_scored_away"] +
        ["h2h_home_wins", "h2h_away_wins", "h2h_draws"] +
        ["goals_scored_similarity", "form_pts_closeness", "avg_goals_conceded"] +
        ["home_elo", "away_elo", "elo_diff"] +
        ["home_form_pts_home", "home_form_pts_away",
         "away_form_pts_home", "away_form_pts_away"] +
        ["days_since_last_match_home", "days_since_last_match_away"] +
        ["home_goals_trend", "away_goals_trend"] +
        ["home_form_pts_exp", "away_form_pts_exp"] +
        ["home_win_rate", "away_win_rate", "win_rate_diff"] +
        ["home_draw_rate", "away_draw_rate"] +
        ["home_defense_variance", "away_defense_variance"] +
        ["home_rank_trajectory", "away_rank_trajectory", "rank_traj_diff"] +
        ["home_elo_vs_league", "away_elo_vs_league", "elo_vs_league_diff"] +
        ["home_advantage"] +
        comp_cols
    )

    print(f"\n  Total features: {len(feature_cols)}")
    for f in feature_cols:
        print(f"  {f}")

    return df, feature_cols


def save_pipeline(df, feature_cols, test_size=0.2, random_state=42):
    print("\nSalvez datele...")

    df_clean = df.dropna(subset=feature_cols + ["result"]).copy()
    print(f"  Meciuri dupa curatare NaN: {len(df_clean)}")

    df_clean = df_clean.sort_values("date").reset_index(drop=True)

    df_clean.to_csv("data_processed.csv", index=False)
    print("  data_processed.csv salvat")

    X = df_clean[feature_cols].values.astype(np.float32)
    y = df_clean["result"].values.astype(np.int32)

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    dist_tr = np.bincount(y_train)
    dist_te = np.bincount(y_test)

    print(f"\n  Temporal split (oldest {100*(1-test_size):.0f}% train / newest {100*test_size:.0f}% test):")
    print(f"  X_train: {X_train.shape}  |  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}   |  y_test:  {y_test.shape}")
    print(f"  Train -- Away(0): {dist_tr[0]}  Home(1): {dist_tr[1]}  Draw(2): {dist_tr[2]}")
    print(f"  Test  -- Away(0): {dist_te[0]}  Home(1): {dist_te[1]}  Draw(2): {dist_te[2]}")
    print(f"  Train dates: {df_clean['date'].iloc[0]} -> {df_clean['date'].iloc[split_idx-1]}")
    print(f"  Test dates:  {df_clean['date'].iloc[split_idx]} -> {df_clean['date'].iloc[-1]}")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    np.save("X_train.npy", X_train)
    np.save("X_test.npy",  X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy",  y_test)
    print("  X_train.npy, X_test.npy, y_train.npy, y_test.npy salvate")

    joblib.dump(scaler, "scaler.pkl")
    print("  scaler.pkl salvat")

    metadata = {
        "feature_names": feature_cols,
        "n_features":    len(feature_cols),
        "n_train":       int(X_train.shape[0]),
        "n_test":        int(X_test.shape[0]),
        "test_size":     test_size,
        "random_state":  random_state,
        "target":        "0=away_win, 1=home_win, 2=draw",
        "competitions":  list(COMPETITIONS.keys()),
        "seasons":       SEASONS,
        "form_window":   5,
        "h2h_window":    3,
    }
    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("  metadata.json salvat")

    metadata_v3 = {**metadata, "version": "v3"}
    with open("metadata_v3.json", "w") as f:
        json.dump(metadata_v3, f, indent=2)
    print("  metadata_v3.json salvat")


if __name__ == "__main__":
    print("PIPELINE DATE")

    print("\nPASUL 1 — Descarc meciuri de la API...")
    df_raw = fetch_matches()

    if df_raw.empty:
        print("\nERROR: Nu s-au descarcat date. Verifica API key-ul!")
        exit(1)

    print("\nPASUL 1B — Combin cu date OpenFootball...")
    df_raw = load_combined_matches(df_raw)

    print("\nPASUL 2 — Feature Engineering...")
    df_features, feature_cols = build_features(df_raw)

    print("\nPASUL 3 — Salvez CSV + Tensori...")
    save_pipeline(df_features, feature_cols)

    print(f"\nPipeline complet. {len(feature_cols)} features.")
