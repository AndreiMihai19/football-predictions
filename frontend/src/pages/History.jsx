import { useState, useEffect } from "react";
import axios from "axios";
import { useTheme } from "../context/ThemeContext";
import { API_URL } from "../config";

const ACCENT = "#00ff87";

const CONF_COLOR = {
  High:   "#00ff87",
  Medium: "#ffa502",
  Low:    "#ff4757",
};

function formatDate(ts) {
  if (!ts) return "—";
  return new Date(ts).toLocaleString("ro-RO", {
    day: "2-digit", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

export default function History() {
  const { theme } = useTheme();
  const c = theme.colors;

  const [history, setHistory]       = useState([]);
  const [accuracy, setAccuracy]     = useState(null);
  const [loading, setLoading]       = useState(true);
  const [verifying, setVerifying]   = useState(false);
  const [verifyMsg, setVerifyMsg]   = useState(null);
  const [marking, setMarking]       = useState(null);
  const [filter, setFilter]         = useState("all");
  const [compFilter, setCompFilter] = useState("all");
  const [predFilter, setPredFilter] = useState("all");
  const [search, setSearch]         = useState("");
  const [page, setPage]             = useState(1);
  const PAGE_SIZE = 25;

  useEffect(() => { init(); }, []);

  async function init() {
    setLoading(true);
    try {
      setVerifying(true);
      const { data } = await axios.post(`${API_URL}/predictions/auto-verify`);
      if (data.verified > 0) setVerifyMsg(`✓ ${data.verified} predicții verificate automat`);
    } catch {
    } finally {
      setVerifying(false);
    }
    await fetchAll();
  }

  async function fetchAll() {
    setLoading(true);
    try {
      const [histRes, accRes] = await Promise.allSettled([
        axios.get(`${API_URL}/predictions/history?limit=100`),
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
      console.error(err.message);
    } finally {
      setMarking(null);
    }
  }

  const filtered = history.filter((item) => {
    if (filter === "pending") { if (item.wasCorrect !== null && item.wasCorrect !== undefined) return false; }
    if (filter === "correct") { if (item.wasCorrect !== true) return false; }
    if (filter === "wrong")   { if (item.wasCorrect !== false) return false; }
    if (compFilter !== "all" && item.competition !== compFilter) return false;
    if (predFilter !== "all" && item.prediction?.label !== predFilter) return false;
    if (search.trim()) {
      const q = search.toLowerCase();
      const home = (item.homeTeam?.name || "").toLowerCase();
      const away = (item.awayTeam?.name || "").toLowerCase();
      if (!home.includes(q) && !away.includes(q)) return false;
    }
    return true;
  });

  const totalPages  = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const currentPage = Math.min(page, totalPages);
  const paginated   = filtered.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);

  function resetPage() { setPage(1); }

  const pageStyle = {
    minHeight: "100vh",
    background: c.bg,
    color: c.text,
    padding: "32px 24px",
  };
  const inner = { maxWidth: 1100, margin: "0 auto" };

  const card = {
    background: c.bgCard,
    border: `1px solid ${c.border}`,
    borderRadius: 12,
    overflow: "hidden",
  };

  const th = {
    padding: "10px 14px",
    textAlign: "left",
    fontSize: "0.75rem",
    fontWeight: 600,
    color: c.textMuted,
    textTransform: "uppercase",
    letterSpacing: "0.05em",
    borderBottom: `1px solid ${c.border}`,
    whiteSpace: "nowrap",
  };

  const td = {
    padding: "10px 14px",
    fontSize: "0.83rem",
    borderBottom: `1px solid ${c.border}`,
    verticalAlign: "middle",
  };

  const filterBtn = (key, activeKey) => ({
    padding: "6px 14px",
    borderRadius: 8,
    fontSize: "0.82rem",
    fontWeight: 500,
    cursor: "pointer",
    border: `1px solid ${activeKey === key ? ACCENT : c.border}`,
    background: activeKey === key ? `${ACCENT}15` : "transparent",
    color: activeKey === key ? ACCENT : c.textMuted,
    transition: "all 0.15s",
  });

  const totalPred   = history.length;
  const verified    = history.filter(i => i.wasCorrect !== null && i.wasCorrect !== undefined).length;
  const pending     = totalPred - verified;
  const accPct      = accuracy?.accuracy ?? null;

  const statCards = [
    { label: "Total predictions", value: totalPred,                    color: c.text },
    { label: "Verified",          value: verified,                     color: "#4a9eff" },
    { label: "Pending",           value: pending,                      color: "#ffa502" },
    { label: "Live accuracy",     value: accPct != null ? `${accPct}%` : "—", color: ACCENT },
  ];

  return (
    <div style={pageStyle}>
      <div style={inner}>

        <div style={{ marginBottom: 28, display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
          <div>
            <h1 style={{ fontSize: "1.6rem", fontWeight: 800, margin: 0, color: c.text }}>
              Predictions <span style={{ color: ACCENT }}>History</span>
            </h1>
            {verifyMsg && (
              <div style={{ marginTop: 8, fontSize: "0.82rem", color: ACCENT }}>
                {verifyMsg}
              </div>
            )}
          </div>
          <button
            onClick={init}
            disabled={verifying || loading}
            style={{
              padding: "8px 18px",
              borderRadius: 8,
              border: `1px solid ${ACCENT}55`,
              background: `${ACCENT}10`,
              color: ACCENT,
              fontWeight: 600,
              fontSize: "0.85rem",
              cursor: verifying || loading ? "default" : "pointer",
              opacity: verifying || loading ? 0.6 : 1,
              whiteSpace: "nowrap",
              flexShrink: 0,
            }}
          >
            {verifying ? "Verifying..." : "↻ Verify now"}
          </button>
        </div>

        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
          gap: 14,
          marginBottom: 24,
        }}>
          {statCards.map(({ label, value, color }) => (
            <div key={label} style={{
              background: c.bgCard,
              border: `1px solid ${c.border}`,
              borderRadius: 10,
              padding: "16px 20px",
              textAlign: "center",
            }}>
              <div style={{ fontSize: "1.6rem", fontWeight: 800, color, marginBottom: 4 }}>
                {value}
              </div>
              <div style={{ fontSize: "0.75rem", color: c.textMuted }}>{label}</div>
            </div>
          ))}
        </div>

        <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginBottom: 16, alignItems: "center" }}>
          <input
            type="text"
            placeholder="Search team..."
            value={search}
            onChange={e => { setSearch(e.target.value); resetPage(); }}
            style={{
              padding: "7px 12px",
              borderRadius: 8,
              border: `1px solid ${c.border}`,
              background: c.bgCard,
              color: c.text,
              fontSize: "0.83rem",
              outline: "none",
              width: 180,
            }}
          />

          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {[
              ["all",     `All (${history.length})`],
              ["correct", `Correct (${accuracy?.correct ?? 0})`],
              ["wrong",   `Wrong (${accuracy?.incorrect ?? 0})`],
              ["pending", `Pending (${history.length - (accuracy?.total ?? 0)})`],
            ].map(([key, label]) => (
              <button key={key} style={filterBtn(key, filter)} onClick={() => { setFilter(key); resetPage(); }}>
                {label}
              </button>
            ))}
          </div>

          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {["all", "PL", "SA", "PD", "BL1", "FL1"].map(comp => (
              <button key={comp} style={filterBtn(comp, compFilter)} onClick={() => { setCompFilter(comp); resetPage(); }}>
                {comp === "all" ? "All leagues" : comp}
              </button>
            ))}
          </div>

          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {[["all", "All pred."], ["Home Win", "Home Win"], ["Away Win", "Away Win"], ["Draw", "Draw"]].map(([key, label]) => (
              <button key={key} style={filterBtn(key, predFilter)} onClick={() => { setPredFilter(key); resetPage(); }}>
                {label}
              </button>
            ))}
          </div>
        </div>

        <div style={{ fontSize: "0.8rem", color: c.textMuted, marginBottom: 10 }}>
          {filtered.length} results — page {currentPage}/{totalPages}
        </div>

        <div style={card}>
          {loading ? (
            <div style={{ padding: "48px", textAlign: "center", color: c.textMuted }}>
              Loading...
            </div>
          ) : filtered.length === 0 ? (
            <div style={{ padding: "48px", textAlign: "center", color: c.textMuted }}>
              No predictions found for the selected filters.
            </div>
          ) : (
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr style={{ background: c.bg }}>
                    <th style={th}>#</th>
                    <th style={th}>Home Team</th>
                    <th style={th}>Away Team</th>
                    <th style={th}>Competition</th>
                    <th style={th}>Prediction</th>
                    <th style={th}>Conf.</th>
                    <th style={th}>Score</th>
                    <th style={th}>Date</th>
                    <th style={{ ...th, textAlign: "center" }}>Actual result</th>
                  </tr>
                </thead>
                <tbody>
                  {paginated.map((item, idx) => {
                    const predLabel    = item.prediction?.label;
                    const predColor    = predLabel === "Home Win" ? ACCENT : predLabel === "Away Win" ? "#ff4757" : "#ffa502";
                    const confColor    = CONF_COLOR[item.prediction?.confidence] ?? c.textMuted;
                    const alreadyMarked = item.wasCorrect !== null && item.wasCorrect !== undefined;
                    const isMarking    = marking === item.id;
                    const rowBg        = idx % 2 === 0 ? "transparent" : `${c.bg}88`;

                    return (
                      <tr
                        key={item.id}
                        style={{ background: rowBg }}
                        onMouseEnter={(e) => (e.currentTarget.style.background = `${ACCENT}08`)}
                        onMouseLeave={(e) => (e.currentTarget.style.background = rowBg)}
                      >
                        <td style={{ ...td, color: c.textMuted, fontSize: "0.75rem" }}>
                          {(currentPage - 1) * PAGE_SIZE + idx + 1}
                        </td>

                        <td style={td}>
                          <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                            {item.homeTeam?.crest && (
                              <img src={item.homeTeam.crest} alt="" style={{ width: 18, height: 18, objectFit: "contain" }} />
                            )}
                            <span style={{ fontWeight: 500 }}>{item.homeTeam?.name || "—"}</span>
                            {item.homeTeam?.rank && (
                              <span style={{ fontSize: "0.7rem", color: c.textMuted }}>#{item.homeTeam.rank}</span>
                            )}
                          </div>
                        </td>

                        <td style={td}>
                          <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                            {item.awayTeam?.crest && (
                              <img src={item.awayTeam.crest} alt="" style={{ width: 18, height: 18, objectFit: "contain" }} />
                            )}
                            <span style={{ fontWeight: 500 }}>{item.awayTeam?.name || "—"}</span>
                            {item.awayTeam?.rank && (
                              <span style={{ fontSize: "0.7rem", color: c.textMuted }}>#{item.awayTeam.rank}</span>
                            )}
                          </div>
                        </td>

                        <td style={{ ...td, color: c.textMuted, fontSize: "0.78rem" }}>
                          {item.competition || "—"}
                        </td>

                        <td style={td}>
                          <span style={{ color: predColor, fontWeight: 700 }}>
                            {predLabel || "—"}
                          </span>
                        </td>

                        <td style={td}>
                          <span style={{
                            padding: "2px 8px",
                            borderRadius: 6,
                            fontSize: "0.72rem",
                            fontWeight: 700,
                            background: `${confColor}18`,
                            color: confColor,
                          }}>
                            {item.prediction?.confidencePercent || 0}%
                          </span>
                        </td>

                        <td style={{ ...td, color: c.textMuted, fontSize: "0.78rem", whiteSpace: "nowrap" }}>
                          {item.score ? `${item.score.home} - ${item.score.away}` : "—"}
                        </td>

                        <td style={{ ...td, color: c.textMuted, fontSize: "0.75rem", whiteSpace: "nowrap" }}>
                          {formatDate(item.timestamp)}
                        </td>

                        <td style={{ ...td, textAlign: "center" }}>
                          {alreadyMarked ? (
                            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
                              <span style={{
                                padding: "3px 10px",
                                borderRadius: 8,
                                fontSize: "0.72rem",
                                fontWeight: 700,
                                background: item.wasCorrect ? "rgba(0,255,135,0.12)" : "rgba(255,71,87,0.12)",
                                color: item.wasCorrect ? ACCENT : "#ff4757",
                              }}>
                                {item.wasCorrect ? "✓ Correct" : "✗ Wrong"}
                              </span>
                              <span style={{ fontSize: "0.65rem", color: c.textMuted }}>
                                {item.actualResult === "home" ? "Home Win" : item.actualResult === "away" ? "Away Win" : "Draw"}
                              </span>
                            </div>
                          ) : (
                            <div style={{ display: "flex", gap: 6, justifyContent: "center" }}>
                              <button
                                style={{ padding: "4px 10px", borderRadius: 6, fontSize: "0.72rem", fontWeight: 600, cursor: isMarking ? "default" : "pointer", border: "1px solid rgba(0,255,135,0.35)", background: "rgba(0,255,135,0.08)", color: ACCENT, opacity: isMarking ? 0.5 : 1 }}
                                disabled={isMarking}
                                onClick={() => markResult(item.id, "home")}
                              >Home</button>
                              <button
                                style={{ padding: "4px 10px", borderRadius: 6, fontSize: "0.72rem", fontWeight: 600, cursor: isMarking ? "default" : "pointer", border: "1px solid rgba(255,165,0,0.35)", background: "rgba(255,165,0,0.08)", color: "#ffa502", opacity: isMarking ? 0.5 : 1 }}
                                disabled={isMarking}
                                onClick={() => markResult(item.id, "draw")}
                              >Draw</button>
                              <button
                                style={{ padding: "4px 10px", borderRadius: 6, fontSize: "0.72rem", fontWeight: 600, cursor: isMarking ? "default" : "pointer", border: "1px solid rgba(255,71,87,0.35)", background: "rgba(255,71,87,0.08)", color: "#ff4757", opacity: isMarking ? 0.5 : 1 }}
                                disabled={isMarking}
                                onClick={() => markResult(item.id, "away")}
                              >Away</button>
                            </div>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {totalPages > 1 && (
          <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: 8, marginTop: 16 }}>
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              style={{ padding: "6px 14px", borderRadius: 8, border: `1px solid ${c.border}`, background: c.bgCard, color: c.text, cursor: currentPage === 1 ? "default" : "pointer", opacity: currentPage === 1 ? 0.4 : 1 }}
            >← Prev</button>

            {Array.from({ length: Math.min(7, totalPages) }, (_, i) => {
              let p;
              if (totalPages <= 7) p = i + 1;
              else if (currentPage <= 4) p = i + 1;
              else if (currentPage >= totalPages - 3) p = totalPages - 6 + i;
              else p = currentPage - 3 + i;
              return (
                <button
                  key={p}
                  onClick={() => setPage(p)}
                  style={{ padding: "6px 12px", borderRadius: 8, border: `1px solid ${p === currentPage ? ACCENT : c.border}`, background: p === currentPage ? `${ACCENT}20` : c.bgCard, color: p === currentPage ? ACCENT : c.text, cursor: "pointer", fontWeight: p === currentPage ? 700 : 400 }}
                >{p}</button>
              );
            })}

            <button
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              style={{ padding: "6px 14px", borderRadius: 8, border: `1px solid ${c.border}`, background: c.bgCard, color: c.text, cursor: currentPage === totalPages ? "default" : "pointer", opacity: currentPage === totalPages ? 0.4 : 1 }}
            >Next →</button>
          </div>
        )}

      </div>
    </div>
  );
}
