import { useState, useEffect } from "react";
import axios from "axios";
import { useTheme } from "../context/ThemeContext";
import { API_URL } from "../config";

const ACCENT = "#00ff87";

const PAD = { left: 52, right: 20, top: 20, bottom: 44 };
const SVG_W = 580;
const SVG_H = 280;
const CHART_W = SVG_W - PAD.left - PAD.right;
const CHART_H = SVG_H - PAD.top - PAD.bottom;
const Y_MIN = 0.35;
const Y_MAX = 0.75;

function toX(val, xMin, xMax) {
  return PAD.left + ((val - xMin) / (xMax - xMin)) * CHART_W;
}

function toY(val) {
  return PAD.top + CHART_H - ((val - Y_MIN) / (Y_MAX - Y_MIN)) * CHART_H;
}

function buildPolyline(sizes, scores) {
  return sizes
    .map((x, i) => `${toX(x, sizes[0], sizes[sizes.length - 1]).toFixed(1)},${toY(scores[i]).toFixed(1)}`)
    .join(" ");
}

function LearningCurveChart({ data, c }) {
  if (!data) return null;
  const { train_sizes, train_scores_mean, val_scores_mean } = data;
  const xMin = train_sizes[0];
  const xMax = train_sizes[train_sizes.length - 1];
  const yTicks = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75];
  const xTicks = train_sizes.filter((_, i) => i % 2 === 0);

  return (
    <svg
      viewBox={`0 0 ${SVG_W} ${SVG_H}`}
      style={{ width: "100%", maxWidth: SVG_W, display: "block", margin: "0 auto" }}
    >
      {yTicks.map((y) => (
        <line
          key={y}
          x1={PAD.left}
          y1={toY(y)}
          x2={SVG_W - PAD.right}
          y2={toY(y)}
          stroke={c.border}
          strokeWidth={1}
          strokeDasharray="4 4"
        />
      ))}

      {yTicks.map((y) => (
        <text
          key={y}
          x={PAD.left - 8}
          y={toY(y) + 4}
          textAnchor="end"
          fill={c.textMuted}
          fontSize={11}
        >
          {(y * 100).toFixed(0)}%
        </text>
      ))}

      {xTicks.map((x) => (
        <text
          key={x}
          x={toX(x, xMin, xMax)}
          y={SVG_H - 8}
          textAnchor="middle"
          fill={c.textMuted}
          fontSize={11}
        >
          {x}
        </text>
      ))}

      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + CHART_H} stroke={c.border} strokeWidth={1} />
      <line x1={PAD.left} y1={PAD.top + CHART_H} x2={SVG_W - PAD.right} y2={PAD.top + CHART_H} stroke={c.border} strokeWidth={1} />

      <polyline
        points={buildPolyline(train_sizes, train_scores_mean)}
        fill="none"
        stroke="#4a9eff"
        strokeWidth={2.5}
        strokeLinejoin="round"
      />

      <polyline
        points={buildPolyline(train_sizes, val_scores_mean)}
        fill="none"
        stroke={ACCENT}
        strokeWidth={2.5}
        strokeLinejoin="round"
      />

      {train_sizes.map((x, i) => (
        <circle key={`t${i}`} cx={toX(x, xMin, xMax)} cy={toY(train_scores_mean[i])} r={3.5} fill="#4a9eff" />
      ))}
      {train_sizes.map((x, i) => (
        <circle key={`v${i}`} cx={toX(x, xMin, xMax)} cy={toY(val_scores_mean[i])} r={3.5} fill={ACCENT} />
      ))}

      <rect x={PAD.left + 10} y={PAD.top + 8} width={12} height={3} fill="#4a9eff" rx={1} />
      <text x={PAD.left + 26} y={PAD.top + 12} fill={c.textMuted} fontSize={12}>Training Accuracy</text>
      <rect x={PAD.left + 160} y={PAD.top + 8} width={12} height={3} fill={ACCENT} rx={1} />
      <text x={PAD.left + 176} y={PAD.top + 12} fill={c.textMuted} fontSize={12}>Validation Accuracy</text>
    </svg>
  );
}

const CAL = { left: 50, right: 20, top: 20, bottom: 40, w: 400, h: 300 };
const CAL_CW = CAL.w - CAL.left - CAL.right;
const CAL_CH = CAL.h - CAL.top - CAL.bottom;

function calX(v) { return CAL.left + v * CAL_CW; }
function calY(v) { return CAL.top  + CAL_CH - v * CAL_CH; }

function CalibrationChart({ data, c }) {
  const { mean_predicted_value: pred, fraction_of_positives: real } = data;
  const ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0];

  const modelPoints = pred.map((x, i) => `${calX(x).toFixed(1)},${calY(real[i]).toFixed(1)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${CAL.w} ${CAL.h}`} style={{ width: "100%", maxWidth: CAL.w, display: "block", margin: "0 auto" }}>
      {ticks.map((t) => (
        <line key={`gx${t}`} x1={calX(t)} y1={CAL.top} x2={calX(t)} y2={CAL.top + CAL_CH}
          stroke={c.border} strokeWidth={1} strokeDasharray="4 4" />
      ))}
      {ticks.map((t) => (
        <line key={`gy${t}`} x1={CAL.left} y1={calY(t)} x2={CAL.left + CAL_CW} y2={calY(t)}
          stroke={c.border} strokeWidth={1} strokeDasharray="4 4" />
      ))}

      <line x1={CAL.left} y1={CAL.top} x2={CAL.left} y2={CAL.top + CAL_CH} stroke={c.border} strokeWidth={1} />
      <line x1={CAL.left} y1={CAL.top + CAL_CH} x2={CAL.left + CAL_CW} y2={CAL.top + CAL_CH} stroke={c.border} strokeWidth={1} />

      {ticks.map((t) => (
        <text key={`lx${t}`} x={calX(t)} y={CAL.h - 8} textAnchor="middle" fill={c.textMuted} fontSize={11}>
          {(t * 100).toFixed(0)}%
        </text>
      ))}
      {ticks.slice(1).map((t) => (
        <text key={`ly${t}`} x={CAL.left - 8} y={calY(t) + 4} textAnchor="end" fill={c.textMuted} fontSize={11}>
          {(t * 100).toFixed(0)}%
        </text>
      ))}

      <line x1={calX(0)} y1={calY(0)} x2={calX(1)} y2={calY(1)}
        stroke="#ffffff30" strokeWidth={1.5} strokeDasharray="6 4" />

      <polyline points={modelPoints} fill="none" stroke={ACCENT} strokeWidth={2.5} strokeLinejoin="round" />

      {pred.map((x, i) => (
        <circle key={i} cx={calX(x)} cy={calY(real[i])} r={5} fill={ACCENT} stroke={c.bgCard} strokeWidth={2} />
      ))}

      <line x1={CAL.left + 10} y1={CAL.top + 12} x2={CAL.left + 30} y2={CAL.top + 12}
        stroke="#ffffff30" strokeWidth={1.5} strokeDasharray="6 4" />
      <text x={CAL.left + 35} y={CAL.top + 16} fill={c.textMuted} fontSize={11}>Perfect calibration</text>
      <circle cx={CAL.left + 20} cy={CAL.top + 28} r={4} fill={ACCENT} />
      <text x={CAL.left + 35} y={CAL.top + 32} fill={c.textMuted} fontSize={11}>Actual model</text>
    </svg>
  );
}

export default function ModelStats() {
  const { theme } = useTheme();
  const c = theme.colors;
  const [stats, setStats] = useState(null);
  const [learningCurve, setLearningCurve] = useState(null);
  const [calibration, setCalibration] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchStats = axios.get(`${API_URL}/predictions/stats`)
      .then((r) => setStats(r.data))
      .catch((err) => setError(`Stats: ${err.response?.data?.error ?? err.message}`));

    const fetchLC = axios.get(`${API_URL}/model/learning-curve`)
      .then((r) => setLearningCurve(r.data))
      .catch(() => {});

    const fetchCal = axios.get(`${API_URL}/model/calibration`)
      .then((r) => setCalibration(r.data))
      .catch(() => {});

    Promise.all([fetchStats, fetchLC, fetchCal]).finally(() => setLoading(false));
  }, []);

  const page = {
    minHeight: "100vh",
    background: c.bg,
    color: c.text,
    padding: "32px 24px",
  };

  const inner = {
    maxWidth: 1100,
    margin: "0 auto",
  };

  const sectionTitle = {
    fontSize: "1.1rem",
    fontWeight: 700,
    color: c.text,
    marginBottom: 20,
    paddingBottom: 10,
    borderBottom: `1px solid ${c.border}`,
    display: "flex",
    alignItems: "center",
    gap: 10,
  };

  const card = {
    background: c.bgCard,
    border: `1px solid ${c.border}`,
    borderRadius: 12,
    padding: "20px 24px",
    marginBottom: 32,
  };

  if (loading) {
    return (
      <div style={{ ...page, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <div style={{ color: c.textMuted, fontSize: "1rem" }}>Loading model statistics...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ ...page, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <div style={{ color: "#ff6b6b", fontSize: "0.95rem" }}>
          Error loading data: {error}
        </div>
      </div>
    );
  }

  const metrics = stats?.test_metrics ?? {};
  const featureImportance = stats?.feature_importance ?? [];
  const confMatrix = metrics.confusion_matrix ?? [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const testSamples = metrics.test_samples ?? 0;
  const trainSamples = stats?.n_train ?? 0;

  const top10 = [...featureImportance]
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 10);
  const maxImportance = top10[0]?.importance ?? 1;

  const metricCards = [
    { label: "Accuracy", value: metrics.accuracy != null ? `${(metrics.accuracy * 100).toFixed(2)}%` : "—", color: ACCENT },
    { label: "F1 Score", value: metrics.f1_macro != null ? metrics.f1_macro.toFixed(4) : "—", color: "#4a9eff" },
    { label: "Precision", value: metrics.precision_macro != null ? metrics.precision_macro.toFixed(4) : "—", color: "#a78bfa" },
    { label: "Recall", value: metrics.recall_macro != null ? metrics.recall_macro.toFixed(4) : "—", color: "#fb923c" },
    { label: "Training data", value: trainSamples ? `${trainSamples.toLocaleString()} matches` : "—", color: c.textMuted },
    { label: "Test data",     value: testSamples  ? `${testSamples.toLocaleString()} matches`  : "—", color: c.textMuted },
  ];

  return (
    <div style={page}>
      <div style={inner}>

        <div style={{ marginBottom: 32 }}>
          <h1 style={{ fontSize: "1.6rem", fontWeight: 800, margin: 0, color: c.text }}>
            Model <span style={{ color: ACCENT }}>&amp;</span> Statistics
          </h1>
        </div>

        <div style={card}>
          <div style={sectionTitle}>
            <span style={{ color: ACCENT }}>▸</span> General Metrics
          </div>
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
            gap: 16,
          }}>
            {metricCards.map(({ label, value, color }) => (
              <div key={label} style={{
                background: c.bg,
                border: `1px solid ${c.border}`,
                borderRadius: 10,
                padding: "16px 18px",
                textAlign: "center",
              }}>
                <div style={{ fontSize: "1.5rem", fontWeight: 800, color, marginBottom: 6, letterSpacing: "-0.5px" }}>
                  {value}
                </div>
                <div style={{ fontSize: "0.78rem", color: c.textMuted, fontWeight: 500 }}>
                  {label}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div style={card}>
          <div style={sectionTitle}>
            <span style={{ color: ACCENT }}>▸</span> Feature Importance — Top 10
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {top10.map(({ feature, importance }) => {
              const pct = ((importance / maxImportance) * 100).toFixed(1);
              const displayPct = ((importance / featureImportance.reduce((s, f) => s + f.importance, 0)) * 100).toFixed(1);
              return (
                <div key={feature} style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <div style={{
                    width: 200,
                    fontSize: "0.82rem",
                    color: c.text,
                    fontFamily: "monospace",
                    flexShrink: 0,
                    textAlign: "right",
                  }}>
                    {feature}
                  </div>
                  <div style={{
                    flex: 1,
                    background: c.bg,
                    borderRadius: 4,
                    height: 18,
                    overflow: "hidden",
                  }}>
                    <div style={{
                      width: `${pct}%`,
                      height: "100%",
                      background: `linear-gradient(90deg, ${ACCENT}cc, ${ACCENT})`,
                      borderRadius: 4,
                      transition: "width 0.5s ease",
                    }} />
                  </div>
                  <div style={{
                    width: 48,
                    fontSize: "0.8rem",
                    color: ACCENT,
                    fontWeight: 600,
                    textAlign: "right",
                    flexShrink: 0,
                  }}>
                    {displayPct}%
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: 24, alignItems: "stretch" }}>

          <div style={{ ...card, marginBottom: 0 }}>
            <div style={sectionTitle}>
              <span style={{ color: ACCENT }}>▸</span> Confusion Matrix
            </div>
            <table style={{ borderCollapse: "collapse", fontSize: "0.85rem" }}>
              <thead>
                <tr>
                  <th style={{ padding: "8px 14px", color: c.textMuted, fontWeight: 500, textAlign: "center" }}></th>
                  <th style={{ padding: "8px 14px", color: c.textMuted, fontWeight: 500, textAlign: "center" }}>Pred Away</th>
                  <th style={{ padding: "8px 14px", color: c.textMuted, fontWeight: 500, textAlign: "center" }}>Pred Home</th>
                  <th style={{ padding: "8px 14px", color: c.textMuted, fontWeight: 500, textAlign: "center" }}>Pred Draw</th>
                </tr>
              </thead>
              <tbody>
                {[["Real Away (0)", 0], ["Real Home (1)", 1], ["Real Draw (2)", 2]].map(([label, row]) => (
                  <tr key={row}>
                    <td style={{ padding: "8px 14px", color: c.textMuted, fontWeight: 500, whiteSpace: "nowrap" }}>
                      {label}
                    </td>
                    <MatrixCell value={confMatrix[row]?.[0] ?? 0} isCorrect={row === 0} c={c} />
                    <MatrixCell value={confMatrix[row]?.[1] ?? 0} isCorrect={row === 1} c={c} />
                    <MatrixCell value={confMatrix[row]?.[2] ?? 0} isCorrect={row === 2} c={c} />
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{ marginTop: 16, display: "flex", gap: 20, fontSize: "0.76rem" }}>
              <span style={{ color: "#4ade80" }}>■ Correct (diagonal)</span>
              <span style={{ color: "#f87171" }}>■ Incorrect</span>
            </div>
          </div>

          <div style={{ ...card, marginBottom: 0 }}>
            <div style={sectionTitle}>
              <span style={{ color: ACCENT }}>▸</span> Learning Curve
            </div>
            <LearningCurveChart data={learningCurve} c={c} />
            <p style={{ marginTop: 12, fontSize: "0.78rem", color: c.textMuted, textAlign: "center" }}>
              Converging curves indicate the absence of severe overfitting.
              X: number of training examples.
            </p>
          </div>
        </div>

        {calibration && (
          <div style={{ ...card, marginTop: 24 }}>
            <div style={sectionTitle}>
              <span style={{ color: ACCENT }}>▸</span> Calibration Curve
              <span style={{ fontSize: "0.78rem", color: c.textMuted, fontWeight: 400, marginLeft: 8 }}>
                ({calibration.n_samples} test samples)
              </span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 32, alignItems: "start" }}>
              <div>
                <CalibrationChart data={calibration} c={c} />
                <p style={{ marginTop: 10, fontSize: "0.78rem", color: c.textMuted, textAlign: "center" }}>
                  Diagonal = perfect calibration. Points above = under-confident model; below diagonal = over-confident.
                </p>
              </div>
              <div style={{ minWidth: 200 }}>
                <div style={{ fontSize: "0.78rem", color: c.textMuted, marginBottom: 12 }}>
                  Predicted probability vs actual frequency:
                </div>
                <table style={{ borderCollapse: "collapse", fontSize: "0.78rem", width: "100%" }}>
                  <thead>
                    <tr>
                      <th style={{ padding: "5px 10px", color: c.textMuted, textAlign: "right", fontWeight: 500 }}>Pred.</th>
                      <th style={{ padding: "5px 10px", color: c.textMuted, textAlign: "right", fontWeight: 500 }}>Actual</th>
                      <th style={{ padding: "5px 10px", color: c.textMuted, textAlign: "right", fontWeight: 500 }}>Δ</th>
                    </tr>
                  </thead>
                  <tbody>
                    {calibration.mean_predicted_value.map((pred, i) => {
                      const real  = calibration.fraction_of_positives[i];
                      const delta = real - pred;
                      return (
                        <tr key={i} style={{ borderTop: `1px solid ${c.border}` }}>
                          <td style={{ padding: "4px 10px", textAlign: "right" }}>{(pred * 100).toFixed(0)}%</td>
                          <td style={{ padding: "4px 10px", textAlign: "right" }}>{(real * 100).toFixed(0)}%</td>
                          <td style={{
                            padding: "4px 10px",
                            textAlign: "right",
                            color: Math.abs(delta) < 0.05 ? ACCENT : Math.abs(delta) < 0.1 ? "#ffa502" : "#ff4757",
                            fontWeight: 600,
                          }}>
                            {delta >= 0 ? "+" : ""}{(delta * 100).toFixed(0)}%
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}

function MatrixCell({ value, isCorrect, c }) {
  const bg = isCorrect ? "#4ade8018" : "#f8717118";
  const border = isCorrect ? "#4ade8055" : "#f8717155";
  const textColor = isCorrect ? "#4ade80" : "#f87171";
  return (
    <td style={{ padding: 6, textAlign: "center" }}>
      <div style={{
        background: bg,
        border: `1px solid ${border}`,
        borderRadius: 8,
        padding: "12px 16px",
        minWidth: 70,
      }}>
        <div style={{ fontSize: "1.3rem", fontWeight: 800, color: textColor }}>{value}</div>
      </div>
    </td>
  );
}
