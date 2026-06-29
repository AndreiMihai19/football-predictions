"""Batch predict matches from data_processed.csv via FastAPI and persist to Firestore."""

import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import firebase_admin
from firebase_admin import credentials, firestore

BACKEND_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
SERVICE_ACCOUNT = os.path.join(BACKEND_DIR, "firebase-service-account.json")

if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT)
    firebase_admin.initialize_app(cred)

db         = firestore.client()
COLLECTION = "predictions"

ML_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_CSV      = os.path.join(ML_DIR, "data_processed.csv")
METADATA_FILE = os.path.join(ML_DIR, "metadata_v3.json")
FASTAPI_URL   = "http://localhost:8000/predict"

DATE_FROM = "2026-01-01"
DATE_TO   = "2026-06-13"

RESULT_MAP = {1: "home", 0: "away", 2: "draw"}


def load_feature_names():
    with open(METADATA_FILE) as f:
        return json.load(f)["feature_names"]


def build_payload(row, feature_names):
    payload = {}
    for feat in feature_names:
        if feat in row.index and pd.notna(row[feat]):
            payload[feat] = float(row[feat])
        else:
            payload[feat] = 0.0
    return payload


def predict_match(payload):
    resp = requests.post(FASTAPI_URL, json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def actual_result_label(result_int):
    return RESULT_MAP.get(int(result_int), "unknown")


def save_to_firestore(row, prediction, actual_result):
    predicted_label = prediction.get("label", "")

    if actual_result == "draw":
        was_correct = predicted_label == "Draw"
    elif actual_result == "home":
        was_correct = predicted_label == "Home Win"
    else:
        was_correct = predicted_label == "Away Win"

    entry = {
        "timestamp":   row["date"] + "T00:00:00Z",
        "homeTeam": {
            "id":    int(row["home_team_id"]),
            "name":  row["home_team"],
            "crest": None,
            "rank":  int(row.get("home_rank", 0)),
        },
        "awayTeam": {
            "id":    int(row["away_team_id"]),
            "name":  row["away_team"],
            "crest": None,
            "rank":  int(row.get("away_rank", 0)),
        },
        "competition": row["competition"],
        "prediction": {
            "label":                prediction.get("label"),
            "probability_home_win": prediction.get("probability_home_win"),
            "probability_away_win": prediction.get("probability_away_win"),
            "probability_draw":     prediction.get("probability_draw"),
            "confidence":           prediction.get("confidence"),
            "confidencePercent":    round(
                max(
                    prediction.get("probability_home_win", 0),
                    prediction.get("probability_away_win", 0),
                    prediction.get("probability_draw", 0),
                ) * 100
            ),
        },
        "score": {
            "home": int(row["home_goals"]),
            "away": int(row["away_goals"]),
        },
        "actualResult": actual_result,
        "wasCorrect":   was_correct,
        "batchImport":  True,
        "createdAt":    datetime.now(timezone.utc),
    }

    db.collection(COLLECTION).add(entry)
    return was_correct


def main():
    print("BATCH PREDICT")

    feature_names = load_feature_names()
    print(f"Features model: {len(feature_names)}")

    df = pd.read_csv(DATA_CSV)
    df_batch = df[(df["date"] >= DATE_FROM) & (df["date"] <= DATE_TO)].copy()
    print(f"Meciuri in interval {DATE_FROM} -> {DATE_TO}: {len(df_batch)}")

    if df_batch.empty:
        print("Niciun meci gasit. Verifica DATA_CSV sau intervalul de date.")
        return

    correct = 0
    total   = 0
    errors  = 0

    for i, (_, row) in enumerate(df_batch.iterrows()):
        payload       = build_payload(row, feature_names)
        actual_result = actual_result_label(row["result"])

        try:
            prediction  = predict_match(payload)
            was_correct = save_to_firestore(row, prediction, actual_result)

            status = "OK" if was_correct else "X "
            print(
                f"[{i+1:>4}/{len(df_batch)}] {status} "
                f"{row['date']} | {row['home_team'][:20]:<20} vs {row['away_team'][:20]:<20} | "
                f"Pred: {prediction.get('label', '?'):<10} | Real: {actual_result}"
            )
            correct += int(was_correct)
            total   += 1

            if (i + 1) % 90 == 0:
                print("  Pauza 5s pentru rate limit...")
                time.sleep(5)

        except Exception as e:
            print(f"[{i+1:>4}] EROARE {row['date']} {row['home_team']} vs {row['away_team']}: {e}")
            errors += 1

    print(f"\nDONE: {total} predictii salvate in Firestore")
    if total > 0:
        print(f"Corecte: {correct}/{total} = {correct/total*100:.1f}%")
    if errors:
        print(f"Erori: {errors}")


if __name__ == "__main__":
    main()
