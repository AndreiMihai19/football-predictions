"""Join data_processed.csv with xg_matches.csv (understat) and add xG features."""

import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

ML_DIR = os.path.dirname(os.path.abspath(__file__))

MANUAL_MAP = {
    "AFC Bournemouth": "Bournemouth",
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "Birmingham City": "Birmingham",
    "Blackburn Rovers": "Blackburn",
    "Blackpool FC": "Blackpool",
    "Bolton Wanderers": "Bolton",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Burnley FC": "Burnley",
    "Cardiff City": "Cardiff",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Hull City": "Hull",
    "Ipswich Town FC": "Ipswich",
    "Leeds United FC": "Leeds",
    "Leicester City": "Leicester",
    "Leicester City FC": "Leicester",
    "Liverpool FC": "Liverpool",
    "Luton Town FC": "Luton",
    "Manchester City FC": "Manchester City",
    "Manchester United FC": "Manchester United",
    "Middlesbrough FC": "Middlesbrough",
    "Newcastle United FC": "Newcastle United",
    "Norwich City": "Norwich",
    "Norwich City FC": "Norwich",
    "Nottingham Forest FC": "Nottingham Forest",
    "Queens Park Rangers": "QPR",
    "Reading FC": "Reading",
    "Sheffield United FC": "Sheffield United",
    "Southampton FC": "Southampton",
    "Stoke City": "Stoke",
    "Sunderland AFC": "Sunderland",
    "Swansea City": "Swansea",
    "Tottenham Hotspur": "Tottenham",
    "Tottenham Hotspur FC": "Tottenham",
    "Watford FC": "Watford",
    "West Bromwich Albion FC": "West Brom",
    "West Ham United": "West Ham",
    "West Ham United FC": "West Ham",
    "Wigan Athletic": "Wigan",
    "Wolverhampton Wanderers FC": "Wolverhampton Wanderers",
    "Sheffield Wednesday FC": "Sheffield Wednesday",
    "Athletic Club": "Athletic Club",
    "Club Atlético de Madrid": "Atletico Madrid",
    "CA Osasuna": "Osasuna",
    "CD Alavés": "Alaves",
    "CD Leganés": "Leganes",
    "Cádiz CF": "Cadiz",
    "Córdoba CF": "Cordoba",
    "Deportivo Alavés": "Alaves",
    "UD Almería": "Almeria",
    "Elche CF": "Elche",
    "Espanyol Barcelona": "Espanyol",
    "RCD Espanyol de Barcelona": "Espanyol",
    "FC Barcelona": "Barcelona",
    "Getafe CF": "Getafe",
    "Girona FC": "Girona",
    "Levante UD": "Levante",
    "Málaga CF": "Malaga",
    "RC Celta": "Celta Vigo",
    "RC Celta de Vigo": "Celta Vigo",
    "Rayo Vallecano de Madrid": "Rayo Vallecano",
    "Real Betis Balompié": "Real Betis",
    "Real Oviedo": None,
    "Real Sociedad de Fútbol": "Real Sociedad",
    "Real Zaragoza": None,
    "SD Eibar": "Eibar",
    "SD Huesca": "Huesca",
    "Sevilla FC": "Sevilla",
    "Sporting Gijón": None,
    "UD Las Palmas": "Las Palmas",
    "Valencia CF": "Valencia",
    "Villarreal CF": "Villarreal",
    "1. FC Heidenheim 1846": "FC Heidenheim",
    "1. FC Kaiserslautern": None,
    "1. FC Köln": "Koeln",
    "1. FC Nürnberg": "Nuernberg",
    "1. FC Union Berlin": "Union Berlin",
    "1. FSV Mainz 05": "Mainz 05",
    "1899 Hoffenheim": "Hoffenheim",
    "Bayer 04 Leverkusen": "Bayer Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "Borussia Mönchengladbach": "Borussia M.Gladbach",
    "Bor. Mönchengladbach": "Borussia M.Gladbach",
    "Eintracht Braunschweig": None,
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "FC Augsburg": "Augsburg",
    "FC Bayern München": "Bayern Munich",
    "FC Ingolstadt 04": None,
    "FC St. Pauli 1910": "St. Pauli",
    "Hamburger SV": "Hamburg",
    "Hannover 96": "Hannover",
    "Hertha BSC": "Hertha Berlin",
    "RB Leipzig": "RasenBallsport Leipzig",
    "SC Freiburg": "Freiburg",
    "SC Paderborn 07": "Paderborn",
    "SV Darmstadt 98": "Darmstadt",
    "SV Werder Bremen": "Werder Bremen",
    "SpVgg Greuther Fürth": "Greuther Fuerth",
    "SpVgg Greuther Fürth 1903": "Greuther Fuerth",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "VfB Stuttgart": "Stuttgart",
    "VfL Bochum 1848": "Bochum",
    "VfL Wolfsburg": "Wolfsburg",
    "AC Cesena": None,
    "AC Milan": "AC Milan",
    "AC Monza": "Monza",
    "AC Pisa 1909": None,
    "ACF Fiorentina": "Fiorentina",
    "AS Livorno": None,
    "AS Roma": "Roma",
    "Atalanta BC": "Atalanta",
    "Benevento Calcio": "Benevento",
    "Bologna FC 1909": "Bologna",
    "Brescia Calcio": "Brescia",
    "Cagliari Calcio": "Cagliari",
    "Calcio Catania": None,
    "Carpi FC": None,
    "Chievo Verona": "Chievo",
    "Como 1907": "Como",
    "Delfino Pescara": None,
    "FC Internazionale Milano": "Internazionale",
    "Frosinone Calcio": "Frosinone",
    "Genoa CFC": "Genoa",
    "Hellas Verona": "Verona",
    "Hellas Verona FC": "Verona",
    "Juventus FC": "Juventus",
    "Lazio Roma": "Lazio",
    "Parma FC": "Parma Calcio 1913",
    "SPAL 2013 Ferrara": "SPAL 2013",
    "SS Lazio": "Lazio",
    "SSC Napoli": "Napoli",
    "Sassuolo Calcio": "Sassuolo",
    "Spezia Calcio": "Spezia",
    "Torino FC": "Torino",
    "UC Sampdoria": "Sampdoria",
    "US Lecce": "Lecce",
    "US Palermo": None,
    "US Salernitana 1919": "Salernitana",
    "US Sassuolo Calcio": "Sassuolo",
    "Udinese Calcio": "Udinese",
    "Venezia FC": "Venezia",
    "AC Ajaccio": "Ajaccio",
    "AJ Auxerre": "Auxerre",
    "AS Monaco FC": "Monaco",
    "AS Nancy Lorraine": None,
    "AS Saint-Étienne": "Saint-Etienne",
    "Angers SCO": "Angers",
    "Dijon FCO": "Dijon",
    "ESTAC Troyes": "Troyes",
    "FC Lorient": "Lorient",
    "FC Metz": "Metz",
    "FC Nantes": "Nantes",
    "Gazélec FC Ajaccio": None,
    "Girondins Bordeaux": "Bordeaux",
    "Lille OSC": "Lille",
    "Montpellier HSC": "Montpellier",
    "Nîmes Olympique": "Nimes",
    "OGC Nice": "Nice",
    "Olympique Lyonnais": "Lyon",
    "Olympique Marseille": "Marseille",
    "Olympique de Marseille": "Marseille",
    "Paris FC": None,
    "Paris Saint-Germain": "Paris Saint Germain",
    "Paris Saint-Germain FC": "Paris Saint Germain",
    "RC Lens": "Lens",
    "RC Strasbourg Alsace": "Strasbourg",
    "Racing Club de Lens": "Lens",
    "SM Caen": "Caen",
    "SC Bastia": None,
    "Stade Brestois 29": "Brest",
    "Stade Rennais": "Rennes",
    "Stade Rennais FC 1901": "Rennes",
    "Stade de Reims": "Reims",
    "Toulouse FC": "Toulouse",
    "Évian Thonon Gaillard": None,
}


def build_team_mapping(df, xg):
    from rapidfuzz import process, fuzz

    mapping = dict(MANUAL_MAP)

    all_df_teams = set(df["home_team"].tolist() + df["away_team"].tolist())
    unmapped = {t for t in all_df_teams if t not in mapping}

    if unmapped:
        xg_teams = set(xg["home_team_understat"].tolist() + xg["away_team_understat"].tolist())
        for team in unmapped:
            result = process.extractOne(team, xg_teams, scorer=fuzz.token_sort_ratio)
            if result and result[1] >= 85:
                mapping[team] = result[0]
            else:
                mapping[team] = None

    return mapping


def compute_xg_features(df, xg, team_map):
    xg_lookup = {}
    for _, row in xg.iterrows():
        key = (row["date"], row["comp"], row["home_team_understat"])
        xg_lookup[key] = {
            "home_xg": row["home_xg"],
            "away_xg": row["away_xg"],
        }

    team_xg_hist    = {}
    team_xga_hist   = {}
    team_goals_hist = {}
    season_xg_hist  = {}

    home_xg_roll5    = []
    away_xg_roll5    = []
    home_xga_roll5   = []
    away_xga_roll5   = []
    home_xg_overperf = []
    away_xg_overperf = []
    home_xg_season   = []
    away_xg_season   = []
    home_xga_season  = []
    away_xga_season  = []

    n = 5

    def rolling_mean(hist):
        if not hist:
            return np.nan
        return float(np.mean(hist[-n:]))

    def season_mean(hist_list):
        if not hist_list:
            return np.nan
        return float(np.mean(hist_list))

    for _, row in df.iterrows():
        htid = row["home_team_id"]
        atid = row["away_team_id"]
        comp = row["competition"]
        date = str(row["date"])[:10]
        season = row["season"]

        us_home = team_map.get(row["home_team"])
        match_xg = xg_lookup.get((date, comp, us_home)) if us_home else None

        h_roll  = rolling_mean(team_xg_hist.get(htid, []))
        a_roll  = rolling_mean(team_xg_hist.get(atid, []))
        h_aroll = rolling_mean(team_xga_hist.get(htid, []))
        a_aroll = rolling_mean(team_xga_hist.get(atid, []))

        home_xg_roll5.append(h_roll)
        away_xg_roll5.append(a_roll)
        home_xga_roll5.append(h_aroll)
        away_xga_roll5.append(a_aroll)

        h_skey = (htid, season)
        a_skey = (atid, season)
        h_season_data = season_xg_hist.get(h_skey, {"xg": [], "xga": []})
        a_season_data = season_xg_hist.get(a_skey, {"xg": [], "xga": []})

        home_xg_season.append(season_mean(h_season_data["xg"]))
        away_xg_season.append(season_mean(a_season_data["xg"]))
        home_xga_season.append(season_mean(h_season_data["xga"]))
        away_xga_season.append(season_mean(a_season_data["xga"]))

        h_goals_hist = team_goals_hist.get(htid, [])
        a_goals_hist = team_goals_hist.get(atid, [])
        h_goals_roll = rolling_mean(h_goals_hist) if h_goals_hist else np.nan
        a_goals_roll = rolling_mean(a_goals_hist) if a_goals_hist else np.nan
        home_xg_overperf.append((h_goals_roll - h_roll) if (not np.isnan(h_roll) and not np.isnan(h_goals_roll)) else np.nan)
        away_xg_overperf.append((a_goals_roll - a_roll) if (not np.isnan(a_roll) and not np.isnan(a_goals_roll)) else np.nan)

        if match_xg:
            team_xg_hist.setdefault(htid, []).append(match_xg["home_xg"])
            team_xg_hist.setdefault(atid, []).append(match_xg["away_xg"])
            team_xga_hist.setdefault(htid, []).append(match_xg["away_xg"])
            team_xga_hist.setdefault(atid, []).append(match_xg["home_xg"])

            season_xg_hist.setdefault(h_skey, {"xg": [], "xga": []})
            season_xg_hist.setdefault(a_skey, {"xg": [], "xga": []})
            season_xg_hist[h_skey]["xg"].append(match_xg["home_xg"])
            season_xg_hist[h_skey]["xga"].append(match_xg["away_xg"])
            season_xg_hist[a_skey]["xg"].append(match_xg["away_xg"])
            season_xg_hist[a_skey]["xga"].append(match_xg["home_xg"])

        team_goals_hist.setdefault(htid, []).append(row["home_goals"])
        team_goals_hist.setdefault(atid, []).append(row["away_goals"])

    df = df.copy()
    df["home_xg_roll5"]     = home_xg_roll5
    df["away_xg_roll5"]     = away_xg_roll5
    df["home_xga_roll5"]    = home_xga_roll5
    df["away_xga_roll5"]    = away_xga_roll5
    df["xg_diff_roll5"]     = df["home_xg_roll5"] - df["away_xg_roll5"]
    df["home_xg_overperf"]  = home_xg_overperf
    df["away_xg_overperf"]  = away_xg_overperf
    df["home_xg_season"]    = home_xg_season
    df["away_xg_season"]    = away_xg_season
    df["home_xga_season"]   = home_xga_season
    df["away_xga_season"]   = away_xga_season
    df["xg_diff_season"]    = df["home_xg_season"] - df["away_xg_season"]

    xg_features = [
        "home_xg_roll5", "away_xg_roll5",
        "home_xga_roll5", "away_xga_roll5",
        "xg_diff_roll5",
        "home_xg_overperf", "away_xg_overperf",
        "home_xg_season", "away_xg_season",
        "home_xga_season", "away_xga_season",
        "xg_diff_season",
    ]

    for col in xg_features:
        comp_means = df.groupby("competition")[col].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        df[col] = df[col].fillna(comp_means).fillna(df[col].mean())

    matched = sum(1 for h in team_xg_hist.values() if h)
    print(f"  Echipe cu xG in istoric: {len(team_xg_hist)}")
    nan_count = df["home_xg_roll5"].isna().sum()
    print(f"  NaN dupa fillna: {nan_count} (ar trebui 0)")

    return df, xg_features


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ML_DIR)
    from pipeline import build_features, save_pipeline

    TRAIN_START = "2017-08-01"

    print("MERGE xG + REGENERARE TENSORI")
    print(f"Filtru date: {TRAIN_START}+")

    RAW_COLS = ["match_id", "competition", "season", "matchday", "date",
                "home_team", "away_team", "home_team_id", "away_team_id",
                "home_goals", "away_goals", "result"]
    SRC_CSV = "data_2015plus_raw.csv"
    all_cols = pd.read_csv(os.path.join(ML_DIR, SRC_CSV), nrows=0).columns.tolist()
    use_cols = [c for c in RAW_COLS if c in all_cols]

    df_raw = pd.read_csv(os.path.join(ML_DIR, SRC_CSV), usecols=use_cols)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw = df_raw.sort_values("date").reset_index(drop=True)

    df_raw = df_raw[df_raw["date"] >= TRAIN_START].copy().reset_index(drop=True)
    print(f"\nDate dupa filtrare {TRAIN_START}+: {len(df_raw)} meciuri")

    df_raw["date"] = df_raw["date"].dt.strftime("%Y-%m-%d")

    dist = df_raw["result"].value_counts().sort_index()
    print(f"Distributie: Away(0)={dist.get(0,0)}  Home(1)={dist.get(1,0)}  Draw(2)={dist.get(2,0)}")

    xg = pd.read_csv(os.path.join(ML_DIR, "xg_matches.csv"))
    xg["date"] = pd.to_datetime(xg["date"]).dt.strftime("%Y-%m-%d")
    xg_filtered = xg[xg["date"] >= TRAIN_START]
    print(f"xG data {TRAIN_START}+: {len(xg_filtered)} meciuri")

    print("\n[1/3] Feature engineering de baza...")
    df_feat, feature_cols = build_features(df_raw)

    print("\n[2/3] Merge xG features...")
    team_map = build_team_mapping(df_raw, xg)
    mapped_count = sum(1 for v in team_map.values() if v is not None)
    print(f"  Team mapping: {mapped_count}/{len(team_map)} echipe mapate in understat")

    df_feat, xg_feature_cols = compute_xg_features(df_feat, xg, team_map)
    feature_cols = feature_cols + xg_feature_cols

    print(f"\n  Total features: {len(feature_cols)}")
    print(f"  xG features adaugate: {xg_feature_cols}")

    print("\n[3/3] Salvez tensori cu xG...")
    save_pipeline(df_feat, feature_cols)

    x = np.load(os.path.join(ML_DIR, "X_train.npy"))
    print(f"\nX_train shape: {x.shape}")
    print("DONE")
