const fs = require("fs");
const path = require("path");

const ML_ELO_PATH      = path.join(__dirname, "..", "..", "ml", "elo_ratings.json");
const STATE_PATH       = path.join(__dirname, "..", "elo_state.json");
const DEFAULT_ELO      = 1500.0;
const K_FACTOR         = 20;

let elos = {};
let loaded = false;

function load() {
  if (loaded) return;

  const sourcePath = fs.existsSync(STATE_PATH) ? STATE_PATH : ML_ELO_PATH;

  if (fs.existsSync(sourcePath)) {
    try {
      const raw = fs.readFileSync(sourcePath, "utf-8");
      elos = JSON.parse(raw);
      console.log(`[elo] Loaded ${Object.keys(elos).length} team ratings from ${path.basename(sourcePath)}`);
    } catch (err) {
      console.warn(`[elo] Failed to load ${sourcePath}:`, err.message);
      elos = {};
    }
  } else {
    console.warn(`[elo] No bootstrap file found at ${ML_ELO_PATH}. All teams will start at ${DEFAULT_ELO}.`);
    elos = {};
  }

  loaded = true;
}

function save() {
  try {
    fs.writeFileSync(STATE_PATH, JSON.stringify(elos, null, 2));
  } catch (err) {
    console.warn("[elo] Failed to persist state:", err.message);
  }
}

function getElo(teamId) {
  load();
  return elos[String(teamId)] ?? DEFAULT_ELO;
}

function updateAfterMatch(homeTeamId, awayTeamId, result) {
  load();

  const ra = getElo(homeTeamId);
  const rb = getElo(awayTeamId);

  const ea = 1.0 / (1.0 + Math.pow(10, (rb - ra) / 400.0));
  const eb = 1.0 - ea;

  const sa = result === "home" ? 1.0 : result === "draw" ? 0.5 : 0.0;
  const sb = 1.0 - sa;

  elos[String(homeTeamId)] = ra + K_FACTOR * (sa - ea);
  elos[String(awayTeamId)] = rb + K_FACTOR * (sb - eb);

  save();
}

function getMatchupFeatures(homeTeamId, awayTeamId) {
  const home_elo = getElo(homeTeamId);
  const away_elo = getElo(awayTeamId);
  return {
    home_elo,
    away_elo,
    elo_diff: home_elo - away_elo,
  };
}

module.exports = { getElo, updateAfterMatch, getMatchupFeatures };
