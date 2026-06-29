const axios = require("axios");
const path = require("path");
const { initializeApp, getApps, cert } = require("firebase-admin/app");
const { getFirestore, FieldValue } = require("firebase-admin/firestore");

if (!getApps().length) {
  const serviceAccount = require(path.join(__dirname, "..", "firebase-service-account.json"));
  initializeApp({ credential: cert(serviceAccount) });
}

const db = getFirestore();
const COLLECTION = "predictions";
const MAX_HISTORY = 100;

function mlClient() {
  return axios.create({
    baseURL: process.env.ML_SERVICE_URL || "http://localhost:8000",
    timeout: 10_000,
  });
}

async function predict(matchData) {
  try {
    const { data } = await mlClient().post("/predict", matchData);
    return data;
  } catch (err) {
    const detail = err.response?.data?.detail ?? err.message;
    throw new Error(`ML service error: ${detail}`);
  }
}

async function getModelStats() {
  try {
    const { data } = await mlClient().get("/stats");
    return data;
  } catch (err) {
    const detail = err.response?.data?.detail ?? err.message;
    throw new Error(`ML service error: ${detail}`);
  }
}

async function healthCheck() {
  try {
    const { data } = await mlClient().get("/health");
    return data;
  } catch {
    return { status: "unavailable" };
  }
}

async function savePrediction(prediction) {
  const entry = {
    timestamp: prediction.timestamp,
    homeTeam: {
      id: prediction.homeTeam?.id ?? null,
      name: prediction.homeTeam?.name ?? null,
      crest: prediction.homeTeam?.crest ?? null,
      rank: prediction.homeTeam?.rank ?? null,
    },
    awayTeam: {
      id: prediction.awayTeam?.id ?? null,
      name: prediction.awayTeam?.name ?? null,
      crest: prediction.awayTeam?.crest ?? null,
      rank: prediction.awayTeam?.rank ?? null,
    },
    competition: prediction.competition ?? null,
    prediction: {
      label: prediction.prediction?.label ?? null,
      probability_home_win: prediction.prediction?.probability_home_win ?? null,
      probability_away_win: prediction.prediction?.probability_away_win ?? null,
      probability_draw: prediction.prediction?.probability_draw ?? null,
      confidence: prediction.prediction?.confidence ?? null,
      confidencePercent: prediction.prediction?.confidencePercent ?? null,
    },
    actualResult: null,
    wasCorrect: null,
    createdAt: FieldValue.serverTimestamp(),
  };

  const docRef = await db.collection(COLLECTION).add(entry);
  return { id: docRef.id, ...entry };
}

async function getPredictionHistory(limit = 20) {
  const snap = await db
    .collection(COLLECTION)
    .orderBy("createdAt", "desc")
    .limit(Math.min(limit, MAX_HISTORY))
    .get();

  return snap.docs.map((doc) => ({ id: doc.id, ...doc.data() }));
}

async function updatePredictionResult(id, actualResult) {
  const docRef = db.collection(COLLECTION).doc(String(id));
  const doc = await docRef.get();
  if (!doc.exists) return null;

  const pred = doc.data();
  let wasCorrect = null;
  if (actualResult === "draw") {
    wasCorrect = pred.prediction?.label === "Draw";
  } else if (actualResult === "home") {
    wasCorrect = pred.prediction?.label === "Home Win";
  } else if (actualResult === "away") {
    wasCorrect = pred.prediction?.label === "Away Win";
  }

  await docRef.update({ actualResult, wasCorrect });
  return { id, ...pred, actualResult, wasCorrect };
}

async function getPredictionAccuracy() {
  const snap = await db
    .collection(COLLECTION)
    .where("wasCorrect", "!=", null)
    .get();

  if (snap.empty) {
    return { accuracy: null, total: 0, correct: 0, message: "No verified predictions yet" };
  }

  const verified = snap.docs.map((d) => d.data());
  const correct  = verified.filter((p) => p.wasCorrect === true).length;
  return {
    accuracy: Math.round((correct / verified.length) * 100),
    total: verified.length,
    correct,
    incorrect: verified.length - correct,
  };
}

module.exports = {
  predict,
  getModelStats,
  healthCheck,
  savePrediction,
  getPredictionHistory,
  updatePredictionResult,
  getPredictionAccuracy,
};
