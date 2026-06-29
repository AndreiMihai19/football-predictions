const express = require("express");
const Joi = require("joi");
const { getTeamStats, getStandings, getH2H, getFinishedMatches } = require("../services/footballApi");
const { predict, getModelStats, getPredictionHistory, savePrediction, updatePredictionResult, getPredictionAccuracy } = require("../services/mlService");
const { getMatchupFeatures: getEloFeatures, updateAfterMatch: updateElo } = require("../services/eloService");

const router = express.Router();

const predictSchema = Joi.object({
  homeTeamId: Joi.number().integer().positive().required(),
  awayTeamId: Joi.number().integer().positive().required(),
  competition: Joi.string().uppercase().valid("PL", "SA", "PD", "BL1", "FL1").required(),
  homeTeamName:  Joi.string().optional().allow("", null),
  awayTeamName:  Joi.string().optional().allow("", null),
  homeTeamCrest: Joi.string().optional().allow("", null),
  awayTeamCrest: Joi.string().optional().allow("", null),
});

router.post("/predict", async (req, res) => {
  const { error, value } = predictSchema.validate(req.body);
  if (error) {
    return res.status(400).json({
      error: error.details[0].message,
      validCompetitions: ["PL", "SA", "PD", "BL1", "FL1"],
    });
  }

  const { homeTeamId, awayTeamId, competition, homeTeamName, awayTeamName, homeTeamCrest, awayTeamCrest } = value;

  try {
    const [homeStats, awayStats, standings, h2hData] = await Promise.all([
      getTeamStats(Number(homeTeamId)),
      getTeamStats(Number(awayTeamId)),
      getStandings(competition),
      getH2H(Number(homeTeamId), Number(awayTeamId)),
    ]);

    const homeRank = standings.table.find(t => t.teamId === Number(homeTeamId))?.position || 10;
    const awayRank = standings.table.find(t => t.teamId === Number(awayTeamId))?.position || 10;
    const rankDiff = awayRank - homeRank;

    const homeGoalsScoredHome = homeStats.form_gf * 1.1;
    const homeGoalsScoredAway = homeStats.form_gf * 0.9;
    const awayGoalsScoredHome = awayStats.form_gf * 1.1;
    const awayGoalsScoredAway = awayStats.form_gf * 0.9;

    const matchFeatures = {
      home_form_pts:  homeStats.form_pts,
      home_form_gf:   homeStats.form_gf,
      home_form_ga:   homeStats.form_ga,
      away_form_pts:  awayStats.form_pts,
      away_form_gf:   awayStats.form_gf,
      away_form_ga:   awayStats.form_ga,
      form_pts_diff:  parseFloat((homeStats.form_pts - awayStats.form_pts).toFixed(3)),
      form_gf_diff:   parseFloat((homeStats.form_gf - awayStats.form_gf).toFixed(3)),
      form_ga_diff:   parseFloat((homeStats.form_ga - awayStats.form_ga).toFixed(3)),

      home_rank:      homeRank,
      away_rank:      awayRank,
      rank_diff:      rankDiff,

      home_goals_scored_home: parseFloat(homeGoalsScoredHome.toFixed(3)),
      home_goals_scored_away: parseFloat(homeGoalsScoredAway.toFixed(3)),
      away_goals_scored_home: parseFloat(awayGoalsScoredHome.toFixed(3)),
      away_goals_scored_away: parseFloat(awayGoalsScoredAway.toFixed(3)),

      h2h_home_wins:  h2hData.h2h_home_wins,
      h2h_away_wins:  h2hData.h2h_away_wins,
      h2h_draws:      h2hData.h2h_draws,

      goals_scored_similarity: parseFloat(
        (1 - Math.abs(homeStats.form_gf - awayStats.form_gf) / Math.max(homeStats.form_gf + awayStats.form_gf, 0.1)).toFixed(3)
      ),
      form_pts_closeness: parseFloat(
        (1 - Math.abs(homeStats.form_pts - awayStats.form_pts) / 3).toFixed(3)
      ),
      avg_goals_conceded: parseFloat(
        ((homeStats.form_ga + awayStats.form_ga) / 2).toFixed(3)
      ),

      ...getEloFeatures(Number(homeTeamId), Number(awayTeamId)),

      home_form_pts_home: homeStats.form_pts_home,
      home_form_pts_away: homeStats.form_pts_away,
      away_form_pts_home: awayStats.form_pts_home,
      away_form_pts_away: awayStats.form_pts_away,

      days_since_last_match_home: homeStats.days_since_last,
      days_since_last_match_away: awayStats.days_since_last,

      home_goals_trend: homeStats.goals_trend,
      away_goals_trend: awayStats.goals_trend,

      home_form_pts_exp: homeStats.form_pts_exp,
      away_form_pts_exp: awayStats.form_pts_exp,

      comp_PL:  competition === "PL"  ? 1.0 : 0.0,
      comp_BL1: competition === "BL1" ? 1.0 : 0.0,
      comp_FL1: competition === "FL1" ? 1.0 : 0.0,
      comp_PD:  competition === "PD"  ? 1.0 : 0.0,
      comp_SA:  competition === "SA"  ? 1.0 : 0.0,
    };

    const prediction = await predict(matchFeatures);

    const response = {
      prediction: {
        ...prediction,
        confidencePercent: Math.round(
          Math.max(
            prediction.probability_home_win,
            prediction.probability_away_win,
            prediction.probability_draw ?? 0
          ) * 100
        ),
      },
      features: matchFeatures,
      homeTeam: {
        id: homeTeamId,
        name: homeStats.teamName ?? homeTeamName ?? null,
        crest: homeStats.teamCrest ?? homeTeamCrest ?? null,
        rank: homeRank,
        stats: homeStats,
      },
      awayTeam: {
        id: awayTeamId,
        name: awayStats.teamName ?? awayTeamName ?? null,
        crest: awayStats.teamCrest ?? awayTeamCrest ?? null,
        rank: awayRank,
        stats: awayStats,
      },
      competition,
      h2h: h2hData,
      timestamp: new Date().toISOString(),
    };

    try {
      await savePrediction(response);
    } catch (e) {
      console.warn("[predictions] Failed to save to history:", e.message);
    }

    res.json(response);
  } catch (err) {
    console.error("[predictions] Error:", err.message);
    const status = err.message.includes("ML service") ? 502 : 500;
    res.status(status).json({ error: err.message });
  }
});

router.get("/stats", async (req, res) => {
  try {
    const stats = await getModelStats();
    res.json(stats);
  } catch (err) {
    res.status(502).json({ error: err.message });
  }
});

router.get("/history", async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 20;
    const history = await getPredictionHistory(limit);
    res.json({ predictions: history, count: history.length });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.post("/auto-verify", async (req, res) => {
  try {
    const limit  = parseInt(req.query.limit) || 100;
    const all    = await getPredictionHistory(limit);
    const pending = all.filter(
      (p) => p.wasCorrect === null || p.wasCorrect === undefined
    );

    if (pending.length === 0) {
      return res.json({ verified: 0, message: "Nicio predictie in asteptare." });
    }

    const competitions = [...new Set(pending.map((p) => p.competition).filter(Boolean))];
    const finishedByComp = {};
    await Promise.all(
      competitions.map(async (comp) => {
        try {
          finishedByComp[comp] = await getFinishedMatches(comp);
        } catch {
          finishedByComp[comp] = [];
        }
      })
    );

    let verified = 0;
    for (const pred of pending) {
      const matches = finishedByComp[pred.competition] || [];

      const match = matches.find(
        (m) =>
          m.homeTeam.name === pred.homeTeam?.name &&
          m.awayTeam.name === pred.awayTeam?.name
      );

      if (!match || match.homeScore == null || match.awayScore == null) continue;

      let actualResult;
      if (match.homeScore > match.awayScore)      actualResult = "home";
      else if (match.awayScore > match.homeScore) actualResult = "away";
      else                                        actualResult = "draw";

      await updatePredictionResult(pred.id, actualResult);

      const homeId = pred.homeTeam?.id ?? match.homeTeam?.id;
      const awayId = pred.awayTeam?.id ?? match.awayTeam?.id;
      if (homeId && awayId) {
        try {
          updateElo(Number(homeId), Number(awayId), actualResult);
        } catch (e) {
          console.warn("[elo] update failed:", e.message);
        }
      }

      verified++;
    }

    res.json({
      verified,
      pending: pending.length,
      message: `${verified} din ${pending.length} predictii verificate automat.`,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.get("/accuracy", async (req, res) => {
  try {
    const stats = await getPredictionAccuracy();
    res.json(stats);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.patch("/:id/result", async (req, res) => {
  const { error, value } = Joi.object({
    actualResult: Joi.string().valid("home", "away", "draw").required(),
  }).validate(req.body);

  if (error) {
    return res.status(400).json({ error: error.details[0].message });
  }

  try {
    const id = parseInt(req.params.id);
    if (isNaN(id)) return res.status(400).json({ error: "ID invalid." });

    const updated = await updatePredictionResult(id, value.actualResult);
    if (!updated) return res.status(404).json({ error: "Predicție negăsită." });

    res.json(updated);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
