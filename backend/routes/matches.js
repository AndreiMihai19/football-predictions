const express = require("express");
const { getUpcomingMatches, getStandings } = require("../services/footballApi");

const router = express.Router();

router.get("/upcoming/:competition", async (req, res) => {
  const { competition } = req.params;

  try {
    const matches = await getUpcomingMatches(competition.toUpperCase());
    res.json({ competition, count: matches.length, matches });
  } catch (err) {
    const httpStatus = err.response?.status;
    console.error(`[matches] /upcoming/${competition} -> HTTP ${httpStatus ?? "N/A"}: ${err.message}`);
    if (err.response?.data) console.error("[matches] API response:", JSON.stringify(err.response.data));
    const status = err.message.includes("invalida") ? 400 : 502;
    res.status(status).json({ error: err.message });
  }
});

router.get("/standings/:competition", async (req, res) => {
  const { competition } = req.params;

  try {
    const standings = await getStandings(competition.toUpperCase());
    res.json(standings);
  } catch (err) {
    const status = err.message.includes("invalida") ? 400 : 502;
    res.status(status).json({ error: err.message });
  }
});

module.exports = router;
