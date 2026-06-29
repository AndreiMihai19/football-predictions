const express = require("express");
const fs      = require("fs").promises;
const path    = require("path");
const axios   = require("axios");

const router = express.Router();

const LEARNING_CURVE_PATH = path.join(__dirname, "../../ml/learning_curve.json");

function mlClient() {
  return axios.create({
    baseURL: process.env.ML_SERVICE_URL || "http://localhost:8000",
    timeout: 10_000,
  });
}

router.get("/learning-curve", async (_req, res) => {
  try {
    const data = await fs.readFile(LEARNING_CURVE_PATH, "utf8");
    res.json(JSON.parse(data));
  } catch (err) {
    if (err.code === "ENOENT") {
      return res.status(404).json({ error: "learning_curve.json not found." });
    }
    res.status(500).json({ error: err.message });
  }
});

router.get("/calibration", async (_req, res) => {
  try {
    const { data } = await mlClient().get("/calibration");
    res.json(data);
  } catch (err) {
    if (err.response?.status === 404) {
      return res.status(404).json({ error: err.response.data.detail });
    }
    res.status(502).json({ error: `ML service error: ${err.message}` });
  }
});

module.exports = router;
