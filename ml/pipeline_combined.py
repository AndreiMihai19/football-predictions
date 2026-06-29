"""Combined pipeline: API + OpenFootball data, full feature engineering, regenerate tensors."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from pipeline import (
    build_features,
    save_pipeline,
)

RAW_COLS = [
    "match_id", "competition", "season", "matchday", "date",
    "home_team", "away_team", "home_team_id", "away_team_id",
    "home_goals", "away_goals", "result",
]


def load_raw_api(csv_path: str = "data_processed.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[RAW_COLS].copy()
    df["source"] = "api"
    return df


def load_openfoot(csv_path: str = "openfoot_matches.csv") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print("  openfoot_matches.csv negasit — rulati fetch_openfoot.py intai!")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df = df[RAW_COLS].copy()
    df["source"] = "openfoot"
    return df


def combine(api_df: pd.DataFrame, of_df: pd.DataFrame, date_from: str = None) -> pd.DataFrame:
    combined = pd.concat([api_df, of_df], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["competition", "season", "date", "home_team", "away_team"]
    )
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    combined = combined.dropna(subset=["date"])
    combined = combined.sort_values("date").reset_index(drop=True)

    print(f"  Date API:          {len(api_df):>6} meciuri")
    print(f"  Date OpenFootball: {len(of_df):>6} meciuri")
    print(f"  Dupa deduplicare:  {len(combined):>6} meciuri")

    if date_from:
        before = len(combined)
        combined = combined[combined["date"] >= date_from].reset_index(drop=True)
        print(f"  Filtru {date_from}+:   {len(combined):>6} meciuri (eliminat {before - len(combined)})")

    dist = combined["result"].value_counts().sort_index()
    print(f"  Away(0): {dist.get(0,0)}  Home(1): {dist.get(1,0)}  Draw(2): {dist.get(2,0)}")

    return combined


if __name__ == "__main__":
    print("PIPELINE COMBINAT (API + OpenFootball)")

    api_df = load_raw_api()
    of_df  = load_openfoot()

    if of_df.empty:
        raise SystemExit("Lipseste openfoot_matches.csv. Ruleaza fetch_openfoot.py mai intai.")

    df_raw = combine(api_df, of_df)

    df_features, feature_cols = build_features(df_raw)

    save_pipeline(df_features, feature_cols)

    print(f"\nPIPELINE COMPLET: {len(df_features)} meciuri, {len(feature_cols)} features")
