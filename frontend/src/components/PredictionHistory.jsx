import { useState, useEffect } from "react";
import axios from "axios";
import { useTheme } from "../context/ThemeContext";
import { API_URL } from "../config";

const CONFIDENCE_COLOR_KEY = {
  High:   "accent",
  Medium: "warning",
  Low:    "danger",
};

export default function PredictionHistory() {
  const { theme } = useTheme();
  const c = theme.colors;

  const [history, setHistory]   = useState([]);
  const [accuracy, setAccuracy] = useState(null);
  const [loading, setLoading]   = useState(true);
  const [marking, setMarking]   = useState(null);

  useEffect(() => { fetchAll(); }, []);

  async function fetchAll() {
    setLoading(true);
    try {
      const [histRes, accRes] = await Promise.allSettled([
        axios.get(`${API_URL}/predictions/history?limit=10`),
        axios.get(`${API_URL}/predictions/accuracy`),
      ]);
      if (histRes.status === "fulfilled") setHistory(histRes.value.data.predictions || []);
      if (accRes.status  === "fulfilled") setAccuracy(accRes.value.data);
    } finally {
      setLoading(false);
    }
  }

  async function markResult(id, actualResult) {
    setMarking(id);
    try {
      await axios.patch(`${API_URL}/predictions/${id}/result`, { actualResult });
      await fetchAll();
    } catch (err) {
      console.error("Mark result failed:", err.message);
    } finally {
      setMarking(null);
    }
  }

  const formatTime = (ts) => {
    if (!ts) return "";
    const diff = Date.now() - new Date(ts);
    const m = Math.floor(diff / 60000);
    const h = Math.floor(m / 60);
    const d = Math.floor(h / 24);
    if (m < 1)  return "just now";
    if (m < 60) return `${m}m`;
    if (h < 24) return `${h}h`;
    if (d < 7)  return `${d}d`;
    return new Date(ts).toLocaleDateString("en-GB", { day: "numeric", month: "short" });
  };

  const accuracyLabel = accuracy?.accuracy != null
    ? `${accuracy.accuracy}% (${accuracy.correct}/${accuracy.total})`
    : null;

  return (
    <div style={{ background: c.bgCard, border: `1px solid ${c.border}`, borderRadius: 14, overflow: "hidden" }}>
      <div style={{ padding: "14px 18px", borderBottom: `1px solid ${c.border}`, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
        <span style={{ fontWeight: 700, fontSize: "0.9rem", color: c.text }}>Recent Predictions</span>
        {accuracyLabel && (
          <span style={{ fontSize: "0.72rem", padding: "3px 10px", borderRadius: 10, fontWeight: 700, background: `${c.accent}1f`, color: c.accent }}>
            {accuracyLabel}
          </span>
        )}
      </div>

      <div style={{ maxHeight: "calc(100vh - 220px)", overflowY: "auto" }}>
        {loading ? (
          <div style={{ padding: "36px 20px", textAlign: "center", color: c.textMuted, fontSize: "0.88rem" }}>Loading...</div>
        ) : history.length === 0 ? (
          <div style={{ padding: "36px 20px", textAlign: "center", color: c.textMuted, fontSize: "0.88rem" }}>
            No predictions yet.<br />Select a match to get started!
          </div>
        ) : (
          history.map((item) => {
            const predIsHome    = item.prediction?.label === "Home Win";
            const predColor     = predIsHome ? c.accent : c.danger;
            const confColorKey  = CONFIDENCE_COLOR_KEY[item.prediction?.confidence];
            const confColor     = confColorKey ? c[confColorKey] : c.textMuted;
            const isMarking     = marking === item.id;
            const wasCorrect    = item.wasCorrect;
            const alreadyMarked = wasCorrect !== null && wasCorrect !== undefined;

            return (
              <div
                key={item.id}
                style={{ padding: "10px 14px", borderBottom: `1px solid ${c.border}`, display: "flex", flexDirection: "column", gap: 6 }}
                onMouseEnter={(e) => (e.currentTarget.style.background = c.bgCard2)}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "0.8rem", color: c.text }}>
                      {item.homeTeam?.crest && <img src={item.homeTeam.crest} alt="" style={{ width: 16, height: 16, objectFit: "contain" }} />}
                      <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{item.homeTeam?.name || "Home"}</span>
                    </div>
                    <div style={{ fontSize: "0.65rem", color: c.textMuted, margin: "1px 0 1px 22px" }}>vs</div>
                    <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "0.8rem", color: c.text }}>
                      {item.awayTeam?.crest && <img src={item.awayTeam.crest} alt="" style={{ width: 16, height: 16, objectFit: "contain" }} />}
                      <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{item.awayTeam?.name || "Away"}</span>
                    </div>
                  </div>

                  <div style={{ textAlign: "right", flexShrink: 0 }}>
                    <div style={{ fontSize: "0.78rem", fontWeight: 600, color: predColor }}>{item.prediction?.label || "N/A"}</div>
                    <div style={{ marginTop: 2 }}>
                      <span style={{ padding: "1px 6px", borderRadius: 6, background: `${confColor}33`, color: confColor, fontWeight: 600, fontSize: "0.65rem" }}>
                        {item.prediction?.confidencePercent || 0}%
                      </span>
                    </div>
                    <div style={{ fontSize: "0.65rem", color: c.textMuted, marginTop: 2 }}>{formatTime(item.timestamp)}</div>
                  </div>
                </div>

                <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  {alreadyMarked ? (
                    <>
                      <span style={{ fontSize: "0.65rem", color: c.textMuted, flexShrink: 0 }}>Result:</span>
                      <span style={{ fontSize: "0.7rem", padding: "2px 9px", borderRadius: 8, fontWeight: 700, background: wasCorrect ? `${c.accent}1f` : `${c.danger}1f`, color: wasCorrect ? c.accent : c.danger }}>
                        {wasCorrect ? "✓ Correct" : "✗ Wrong"}
                      </span>
                      <span style={{ fontSize: "0.65rem", color: c.textMuted, marginLeft: "auto" }}>
                        {item.actualResult === "home" ? "Home Win" : item.actualResult === "away" ? "Away Win" : "Draw"}
                      </span>
                    </>
                  ) : (
                    <>
                      <span style={{ fontSize: "0.65rem", color: c.textMuted, flexShrink: 0 }}>Actual result:</span>
                      {[
                        ["home", "Home", c.accent],
                        ["draw", "Draw", c.warning],
                        ["away", "Away", c.danger],
                      ].map(([val, label, color]) => (
                        <button
                          key={val}
                          style={{ fontSize: "0.68rem", padding: "3px 9px", borderRadius: 6, border: `1px solid ${color}66`, cursor: isMarking ? "default" : "pointer", fontWeight: 600, transition: "opacity 0.15s", color, background: `${color}14`, opacity: isMarking ? 0.5 : 1 }}
                          disabled={isMarking}
                          onClick={() => markResult(item.id, val)}
                        >
                          {label}
                        </button>
                      ))}
                    </>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
