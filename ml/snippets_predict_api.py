# Snippet 1 — Modelele de date (predict_api.py)
class PredictionResponse(BaseModel):
    prediction:           int
    label:                str
    probability_home_win: float
    probability_away_win: float
    probability_draw:     float
    confidence:           str
    mode_used:            Optional[str] = None
    shap_contributions:   Optional[List[ShapContribution]] = None


# Snippet 2 — Endpoint /predict (predict_api.py)
@app.post("/predict", response_model=PredictionResponse)
def predict(features: MatchFeatures):
    feature_dict = features.model_dump()
    req_mode = feature_dict.pop("prediction_mode", None)

    X_raw = np.array(
        [feature_dict[name] for name in state["all_feature_names"]],
        dtype=np.float32,
    ).reshape(1, -1)

    X_scaled = state["scaler"].transform(X_raw)
    X_v2     = np.delete(X_scaled, state["dead_indices"], axis=1)

    model = state["model"]
    mode  = (req_mode or state.get("default_mode", "f1")).lower()

    prediction = int(model.predict(X_v2, mode=mode)[0])
    proba      = model.predict_proba(X_v2, mode=mode)[0]

    prob_away = round(float(proba[0]), 4)
    prob_home = round(float(proba[1]), 4)
    prob_draw = round(float(proba[2]), 4)

    max_prob = max(prob_home, prob_away, prob_draw)
    if max_prob > 0.65:
        confidence = "High"
    elif max_prob > 0.50:
        confidence = "Medium"
    else:
        confidence = "Low"

    return PredictionResponse(
        prediction=prediction,
        label=LABEL_MAP.get(prediction, "Unknown"),
        probability_home_win=prob_home,
        probability_away_win=prob_away,
        probability_draw=prob_draw,
        confidence=confidence,
        mode_used=mode,
    )


# Snippet 3 — Calculul SHAP contributions (predict_api.py)
if state["explainer"] is not None:
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
