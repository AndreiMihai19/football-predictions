require("dotenv").config();

const express = require("express");
const cors    = require("cors");
const rateLimit = require("express-rate-limit");

const matchesRouter     = require("./routes/matches");
const predictionsRouter = require("./routes/predictions");
const modelRouter       = require("./routes/model");
const { healthCheck }   = require("./services/mlService");

const app  = express();
const PORT = process.env.PORT || 3001;

const apiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: process.env.NODE_ENV === "production" ? 100 : 1000,
  message: {
    error: "Too many requests, please try again later.",
    retryAfter: "15 minutes",
  },
  standardHeaders: true,
  legacyHeaders: false,
});

const predictLimiter = rateLimit({
  windowMs: 1 * 60 * 1000,
  max: process.env.NODE_ENV === "production" ? 10 : 100,
  message: {
    error: "Too many prediction requests. Please wait a moment.",
    retryAfter: "1 minute",
  },
});

app.use(cors({
  origin: [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
  ],
  methods: ["GET", "POST"],
}));
app.use(express.json());

app.use("/api/", apiLimiter);
app.use("/api/predictions/predict", predictLimiter);

app.use("/api/matches",     matchesRouter);
app.use("/api/predictions", predictionsRouter);
app.use("/api/model",       modelRouter);

app.get("/api/health", async (_req, res) => {
  const ml = await healthCheck();
  res.json({
    status:     "ok",
    port:       PORT,
    mlService:  ml,
    version:    "2.0.0",
    improvements: [
      "Rate limiting",
      "Request validation",
      "Prediction history",
      "Enhanced features",
    ],
  });
});

app.use((req, res) => {
  res.status(404).json({ error: `Route not found: ${req.method} ${req.path}` });
});

app.use((err, _req, res, _next) => {
  console.error("[ERROR]", err.message);
  res.status(500).json({ error: "Internal server error." });
});

app.listen(PORT, () => {
  console.log(`Backend running on http://localhost:${PORT}`);
});