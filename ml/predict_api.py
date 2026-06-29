"""FastAPI inference server for 3-class football outcome predictions."""

import json
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.calibration import calibration_curve

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

ML_DIR = os.path.dirname(os.path.abspath(__file__))

LABEL_MAP = {0: "Away Win", 1: "Home Win", 2: "Draw"}

def p(filename: str) -> str:
    return os.path.join(ML_DIR, filename)


DEAD_FEATURES = ["home_advantage"]

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Incarcare model (v3 preferred, fallback v2)...")

    use_v3 = (
        os.path.exists(p("model_v3.pkl"))
        and os.path.exists(p("metadata_v3.json"))
    )

    if use_v3:
        model_file   = "model_v3.pkl"
        meta_file    = "metadata_v3.json"
        results_file = "results_v3.json"
        version      = "v3"
    else:
        model_file   = "model_v2.pkl"
        meta_file    = "metadata_v2.json"
        results_file = "results_v2.json"
        version      = "v2"

    required = [model_file, "scaler.pkl", "metadata.json", meta_file]
    for f in required:
        if not os.path.exists(p(f)):
            raise RuntimeError(
                f"Fisier lipsa: {f}. Ruleaza pipeline.py si train_model_v{version[-1]}.py intai."
            )

    state["model_version"] = version
    state["model"]  = joblib.load(p(model_file))
    state["scaler"] = joblib.load(p("scaler.pkl"))

    with open(p("metadata.json")) as f:
        meta = json.load(f)
    state["all_feature_names"] = meta["feature_names"]

    with open(p(meta_file)) as f:
        meta_active = json.load(f)
    state["feature_names"] = meta_active["feature_names"]
    state["n_classes"]     = meta_active.get("n_classes", 3)
    state["supports_modes"] = bool(meta_active.get("supports_modes", False))
    state["default_mode"]   = meta_active.get("default_mode", "f1") or "f1"
    state["draw_threshold"] = meta_active.get("draw_threshold")

    state["dead_indices"] = [
        i for i, name in enumerate(state["all_feature_names"])
        if name in DEAD_FEATURES
    ]

    state["explainer"] = None
    if HAS_SHAP:
        try:
            model = state["model"]
            base  = model.estimator if hasattr(model, "estimator") else model
            if hasattr(base, "estimators_"):
                base = base.estimators_[0]
            state["explainer"] = shap.TreeExplainer(base)
            print("SHAP explainer: TreeExplainer (XGBoost component)")
        except Exception as e:
            print(f"SHAP explainer failed: {e} - predictions will work without SHAP")

    state["n_train"]  = meta.get("n_train", 0)
    state["accuracy"] = 0.0
    state["results"]  = {}
    if os.path.exists(p(results_file)):
        with open(p(results_file)) as f:
            results = json.load(f)
        state["accuracy"] = results.get("test_metrics", {}).get("accuracy", 0.0)
        state["results"]  = results

    print(f"Model {version} incarcat: YES ({meta_active.get('model_type', 'ensemble')}, {state['n_classes']} clase)")
    print(f"Active features ({len(state['feature_names'])}): {state['feature_names']}")
    print(f"Accuracy: {state['accuracy'] * 100:.2f}%")
    print("Server gata pe http://localhost:8000")

    yield
    state.clear()
    print("Server oprit.")


app = FastAPI(
    title="Football Predictions API",
    description="Pronosticuri fotbal 3-clase: Away Win / Home Win / Draw",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000",
                   "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class MatchFeatures(BaseModel):
    prediction_mode: Optional[str] = None
    home_form_pts:   float
    home_form_gf:    float
    home_form_ga:    float
    away_form_pts:   float
    away_form_gf:    float
    away_form_ga:    float
    form_pts_diff:   float
    form_gf_diff:    float
    form_ga_diff:    float
    home_rank:                float = 10.0
    away_rank:                float = 10.0
    rank_diff:                float = 0.0
    home_goals_scored_home:   float = 1.0
    home_goals_scored_away:   float = 1.0
    away_goals_scored_home:   float = 1.0
    away_goals_scored_away:   float = 1.0
    h2h_home_wins:            float = 1.0
    h2h_away_wins:            float = 1.0
    h2h_draws:                float = 0.0
    goals_scored_similarity:  float = 0.0
    form_pts_closeness:       float = 0.0
    avg_goals_conceded:       float = 1.0
    home_elo:                 float = 1500.0
    away_elo:                 float = 1500.0
    elo_diff:                 float = 0.0
    home_form_pts_home:       float = 1.0
    home_form_pts_away:       float = 1.0
    away_form_pts_home:       float = 1.0
    away_form_pts_away:       float = 1.0
    days_since_last_match_home: float = 7.0
    days_since_last_match_away: float = 7.0
    home_goals_trend:         float = 0.0
    away_goals_trend:         float = 0.0
    home_form_pts_exp:        float = 1.0
    away_form_pts_exp:        float = 1.0
    home_win_rate:            float = 0.33
    away_win_rate:            float = 0.33
    win_rate_diff:            float = 0.0
    home_draw_rate:           float = 0.33
    away_draw_rate:           float = 0.33
    home_defense_variance:    float = 1.0
    away_defense_variance:    float = 1.0
    home_rank_trajectory:     float = 0.0
    away_rank_trajectory:     float = 0.0
    rank_traj_diff:           float = 0.0
    home_elo_vs_league:       float = 0.0
    away_elo_vs_league:       float = 0.0
    elo_vs_league_diff:       float = 0.0
    home_xg_roll5:            float = 0.0
    away_xg_roll5:            float = 0.0
    home_xga_roll5:           float = 0.0
    away_xga_roll5:           float = 0.0
    xg_diff_roll5:            float = 0.0
    home_xg_overperf:         float = 0.0
    away_xg_overperf:         float = 0.0
    home_xg_season:           float = 0.0
    away_xg_season:           float = 0.0
    home_xga_season:          float = 0.0
    away_xga_season:          float = 0.0
    xg_diff_season:           float = 0.0
    # Scaler expects these in the input vector; they are stripped via dead_indices before inference.
    home_advantage:           float = 1.0
    comp_BL1:                 float = 0.0
    comp_FL1:                 float = 0.0
    comp_PD:                  float = 0.0
    comp_PL:                  float = 0.0
    comp_SA:                  float = 0.0

    model_config = {"json_schema_extra": {"example": {
        "home_form_pts": 2.1, "home_form_gf": 1.8, "home_form_ga": 0.9,
        "away_form_pts": 1.4, "away_form_gf": 1.2, "away_form_ga": 1.5,
        "form_pts_diff": 0.7, "form_gf_diff": 0.6, "form_ga_diff": -0.6,
        "home_rank": 5.0, "away_rank": 8.0, "rank_diff": 3.0,
        "home_goals_scored_home": 1.8, "home_goals_scored_away": 1.2,
        "away_goals_scored_home": 1.5, "away_goals_scored_away": 1.0,
        "h2h_home_wins": 1.0, "h2h_away_wins": 1.0, "h2h_draws": 0.0,
        "goals_scored_similarity": 0.6, "form_pts_closeness": 0.7,
        "avg_goals_conceded": 1.2,
        "home_advantage": 1, "comp_BL1": 0, "comp_PD": 0, "comp_PL": 1,
        "comp_SA": 0, "comp_FL1": 0,
    }}}


class ShapContribution(BaseModel):
    feature: str
    value:   float


class PredictionResponse(BaseModel):
    prediction:           int
    label:                str
    probability_home_win: float
    probability_away_win: float
    probability_draw:     float
    confidence:           str
    mode_used:            Optional[str] = None
    shap_contributions:   Optional[List[ShapContribution]] = None


@app.get("/health")
def health():
    version = state.get("model_version", "v2")
    return {
        "status":   "ok",
        "model":    f"Ensemble {version} (3-class: Away/Home/Draw)",
        "version":  version,
        "accuracy": round(state["accuracy"] * 100, 2),
        "features": len(state["feature_names"]),
        "n_classes": state.get("n_classes", 3),
        "supports_modes": state.get("supports_modes", False),
        "default_mode":   state.get("default_mode", "f1"),
        "draw_threshold": state.get("draw_threshold"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: MatchFeatures):
    try:
        all_feature_names = state["all_feature_names"]
        feature_dict      = features.model_dump()
        req_mode = feature_dict.pop("prediction_mode", None)

        X_raw = np.array(
            [feature_dict[name] for name in all_feature_names],
            dtype=np.float32,
        ).reshape(1, -1)

        X_scaled = state["scaler"].transform(X_raw)
        X_v2     = np.delete(X_scaled, state["dead_indices"], axis=1)

        model = state["model"]
        supports_modes = state.get("supports_modes", False)
        mode = (req_mode or state.get("default_mode", "f1")).lower()
        if mode not in ("f1", "accuracy"):
            raise HTTPException(status_code=422, detail=f"prediction_mode invalid: {mode}. Use 'f1' or 'accuracy'.")

        if supports_modes:
            prediction = int(model.predict(X_v2, mode=mode)[0])
            proba      = model.predict_proba(X_v2, mode=mode)[0]
            mode_used  = mode
        else:
            prediction = int(model.predict(X_v2)[0])
            proba      = model.predict_proba(X_v2)[0]
            mode_used  = "default"

        prob_away = round(float(proba[0]), 4)
        prob_home = round(float(proba[1]), 4)
        prob_draw = round(float(proba[2]), 4) if len(proba) > 2 else 0.0

        max_prob = max(prob_home, prob_away, prob_draw)
        if max_prob > 0.65:
            confidence = "High"
        elif max_prob > 0.50:
            confidence = "Medium"
        else:
            confidence = "Low"

        shap_contributions = None
        if state["explainer"] is not None:
            try:
                sv = state["explainer"].shap_values(X_v2)
                if isinstance(sv, list) and len(sv) > prediction:
                    sv = sv[prediction][0]
                elif sv.ndim == 3:
                    sv = sv[prediction][0]
                else:
                    sv = sv[0]
                shap_contributions = [
                    ShapContribution(feature=name, value=round(float(val), 4))
                    for name, val in zip(state["feature_names"], sv)
                ]
            except Exception as e:
                print(f"[SHAP] Error: {e}")

        return PredictionResponse(
            prediction=prediction,
            label=LABEL_MAP.get(prediction, "Unknown"),
            probability_home_win=prob_home,
            probability_away_win=prob_away,
            probability_draw=prob_draw,
            confidence=confidence,
            mode_used=mode_used,
            shap_contributions=shap_contributions,
        )

    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Feature lipsa: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def stats():
    if not state["results"]:
        raise HTTPException(status_code=404, detail="results_v2.json nu a fost gasit.")
    return {**state["results"], "n_train": state.get("n_train", 0)}


@app.get("/calibration")
def get_calibration():
    x_path = p("X_test.npy")
    y_path = p("y_test.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise HTTPException(status_code=404, detail="Date de test negasite.")

    X_test = np.load(x_path)
    y_test = np.load(y_path)

    X_scaled = state["scaler"].transform(X_test)
    X_v2     = np.delete(X_scaled, state["dead_indices"], axis=1)
    proba    = state["model"].predict_proba(X_v2)

    y_binary = (y_test == 1).astype(int)
    frac_pos, mean_pred = calibration_curve(
        y_binary, proba[:, 1], n_bins=10, strategy="uniform"
    )

    return {
        "fraction_of_positives": [round(float(v), 4) for v in frac_pos],
        "mean_predicted_value":  [round(float(v), 4) for v in mean_pred],
        "n_bins":    len(frac_pos),
        "n_samples": int(len(y_test)),
        "class":     "Home Win (1)",
    }


if __name__ == "__main__":
    uvicorn.run("predict_api:app", host="0.0.0.0", port=8000, reload=True)
