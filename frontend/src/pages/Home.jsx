import { useState, useEffect, useRef } from "react";
import axios from "axios";
import MatchCard from "../components/MatchCard";
import PredictionCard from "../components/PredictionCard";
import PredictionHistory from "../components/PredictionHistory";
import { API_URL } from "../config";
import { useTheme } from "../context/ThemeContext";

const COMPETITIONS = [
  { code: "PL", name: "Premier League" },
  { code: "SA", name: "Serie A" },
  { code: "PD", name: "La Liga" },
  { code: "BL1", name: "Bundesliga" },
  { code: "FL1", name: "Ligue 1" },
];

export default function Home() {
  const { theme } = useTheme();
  const c = theme.colors;

  const [competition, setCompetition] = useState("PL");
  const [matches, setMatches] = useState([]);
  const [matchesLoading, setMatchesLoading] = useState(false);
  const [matchesError, setMatchesError] = useState(null);

  const [selectedMatch, setSelectedMatch] = useState(null);
  const [predResult, setPredResult] = useState(null);
  const [predLoading, setPredLoading] = useState(false);
  const [predError, setPredError] = useState(null);
  const [accuracy, setAccuracy] = useState(null);

  const [historyKey, setHistoryKey] = useState(0);

  const fetchingFor = useRef(null);

  const s = {
    wrapper: {
      maxWidth: 1400,
      margin: "0 auto",
      padding: "28px 24px",
    },
    grid: {
      display: "grid",
      gridTemplateColumns: "1fr 1.2fr 0.8fr",
      gap: 20,
      alignItems: "start",
    },
    panel: {
      background: c.bgCard,
      border: `1px solid ${c.border}`,
      borderRadius: 14,
      overflow: "hidden",
    },
    panelHeader: {
      padding: "16px 20px 12px",
      borderBottom: `1px solid ${c.border}`,
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      gap: 12,
    },
    panelTitle: {
      fontWeight: 700,
      fontSize: "0.95rem",
      color: c.text,
    },
    select: {
      background: c.bgCard2,
      color: c.text,
      border: `1px solid ${c.border}`,
      borderRadius: 7,
      padding: "6px 12px",
      fontSize: "0.82rem",
      outline: "none",
      cursor: "pointer",
    },
    matchList: {
      padding: "12px",
      display: "flex",
      flexDirection: "column",
      gap: 8,
      maxHeight: "calc(100vh - 180px)",
      overflowY: "auto",
    },
    middlePanel: {
      position: "sticky",
      top: 76,
      background: c.bgCard,
      border: `1px solid ${c.border}`,
      borderRadius: 14,
      overflow: "hidden",
    },
    placeholder: {
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      padding: "60px 32px",
      textAlign: "center",
      gap: 12,
    },
    placeholderIcon: {
      fontSize: "2.5rem",
      opacity: 0.3,
    },
    placeholderText: {
      color: c.textMuted,
      fontSize: "0.9rem",
      lineHeight: 1.6,
    },
    predWrap: {
      padding: 20,
    },
    count: {
      fontSize: "0.78rem",
      color: c.textMuted,
    },
    rightColumn: {
      position: "sticky",
      top: 76,
    },
  };

  useEffect(() => {
    axios
      .get(`${API_URL}/predictions/stats`)
      .then(({ data }) => {
        const acc = data?.test_metrics?.accuracy;
        if (acc != null) setAccuracy(Math.round(acc * 10000) / 100);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (fetchingFor.current === competition) return;
    let cancelled = false;
    fetchingFor.current = competition;
    setMatchesLoading(true);
    setMatchesError(null);
    setMatches([]);
    setSelectedMatch(null);
    setPredResult(null);

    axios
      .get(`${API_URL}/matches/upcoming/${competition}`)
      .then(({ data }) => {
        if (!cancelled) setMatches(data.matches || []);
      })
      .catch((err) => {
        if (!cancelled)
          setMatchesError(
            err.response?.data?.error ?? "Failed to load matches."
          );
      })
      .finally(() => {
        if (!cancelled) setMatchesLoading(false);
      });

    return () => {
      cancelled = true;
      fetchingFor.current = null;
    };
  }, [competition]);

  async function handlePredict(match) {
    if (predLoading) return;
    setSelectedMatch(match.id);
    setPredLoading(true);
    setPredError(null);
    setPredResult(null);

    try {
      const { data } = await axios.post(`${API_URL}/predictions/predict`, {
        homeTeamId: match.homeTeam.id,
        awayTeamId: match.awayTeam.id,
        competition: match.competition,
        homeTeamName: match.homeTeam.name,
        awayTeamName: match.awayTeam.name,
        homeTeamCrest: match.homeTeam.crest,
        awayTeamCrest: match.awayTeam.crest,
      });
      setPredResult(data);
      setHistoryKey((k) => k + 1);
    } catch (err) {
      setPredError(
        err.response?.data?.error ?? "Eroare la generarea predictiei."
      );
    } finally {
      setPredLoading(false);
    }
  }

  return (
    <div style={s.wrapper}>
      <div style={s.grid} className="home-grid">
        <div style={s.panel}>
          <div style={s.panelHeader}>
            <span style={s.panelTitle}>Upcoming Matches</span>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              {!matchesLoading && !matchesError && (
                <span style={s.count}>{matches.length} matches</span>
              )}
              <select
                style={s.select}
                value={competition}
                onChange={(e) => setCompetition(e.target.value)}
              >
                {COMPETITIONS.map((comp) => (
                  <option key={comp.code} value={comp.code}>
                    {comp.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div style={s.matchList}>
            {matchesLoading && <SkeletonList c={c} />}
            {matchesError && <div className="error-msg">{matchesError}</div>}
            {!matchesLoading && !matchesError && matches.length === 0 && (
              <div style={{ ...s.placeholder, padding: "40px 20px" }}>
                <span style={s.placeholderText}>
                  No scheduled matches.
                </span>
              </div>
            )}
            {!matchesLoading &&
              !matchesError &&
              matches.map((m) => (
                <MatchCard
                  key={m.id}
                  match={m}
                  selected={selectedMatch === m.id}
                  loading={predLoading}
                  onPredict={handlePredict}
                />
              ))}
          </div>
        </div>

        <div style={s.middlePanel}>
          <div style={s.panelHeader}>
            <span style={s.panelTitle}>Model Prediction</span>
            {accuracy != null && (
              <span style={{ ...s.count, color: c.accent }}>
                Accuracy {accuracy}%
              </span>
            )}
          </div>

          {!predResult && !predLoading && !predError && (
            <div style={s.placeholder}>
              <span style={s.placeholderIcon}>⚽</span>
              <span style={s.placeholderText}>
                Select a match on the left
                <br />
                to generate a prediction
              </span>
            </div>
          )}

          {predLoading && <div className="spinner" />}

          {predError && (
            <div style={{ padding: 20 }}>
              <div className="error-msg">{predError}</div>
            </div>
          )}

          {!predLoading && predResult && (
            <div style={s.predWrap}>
              <PredictionCard
                data={predResult}
                homeTeamName={predResult.homeTeam?.name ?? ""}
                awayTeamName={predResult.awayTeam?.name ?? ""}
                accuracy={accuracy}
              />
              <TeamStats label="Home" stats={predResult.homeTeam?.stats} c={c} />
              <TeamStats label="Away" stats={predResult.awayTeam?.stats} c={c} />
            </div>
          )}
        </div>

        <div style={s.rightColumn}>
          <PredictionHistory key={historyKey} />
        </div>
      </div>
    </div>
  );
}

function SkeletonCard({ c }) {
  return (
    <div
      style={{
        background: c.bgCard2,
        border: `1px solid ${c.border}`,
        borderRadius: 10,
        padding: "12px 14px",
        display: "flex",
        alignItems: "center",
        gap: 10,
      }}
    >
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 8 }}>
        <div className="skeleton" style={{ height: 14, width: "75%" }} />
        <div className="skeleton" style={{ height: 11, width: "40%" }} />
      </div>
      <div
        className="skeleton"
        style={{ width: 32, height: 28, borderRadius: 7 }}
      />
    </div>
  );
}

function SkeletonList({ c }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {[1, 2, 3, 4].map((i) => (
        <SkeletonCard key={i} c={c} />
      ))}
    </div>
  );
}

function TeamStats({ label, stats, c }) {
  if (!stats) return null;
  const formColors = { W: "#00ff87", D: "#ffa502", L: "#ff4757" };
  const box = {
    background: c.bgCard2,
    borderRadius: 10,
    padding: "14px 16px",
    marginTop: 10,
  };
  const title = {
    fontSize: "0.72rem",
    color: c.textMuted,
    textTransform: "uppercase",
    letterSpacing: "0.06em",
    marginBottom: 10,
    fontWeight: 600,
  };
  const row = {
    display: "flex",
    justifyContent: "space-between",
    padding: "4px 0",
    fontSize: "0.82rem",
    borderBottom: `1px solid ${c.border}`,
  };

  return (
    <div style={box}>
      <div style={title}>Form {label} — last 5 matches</div>
      <div style={row}>
        <span style={{ color: c.textMuted }}>Form</span>
        <span>
          {stats.form.split("").map((char, i) => (
            <span
              key={i}
              style={{
                color: formColors[char] ?? c.text,
                fontWeight: 700,
                marginLeft: 3,
              }}
            >
              {char}
            </span>
          ))}
        </span>
      </div>
      <div style={row}>
        <span style={{ color: c.textMuted }}>W / D / L</span>
        <span>
          {stats.wins} / {stats.draws} / {stats.losses}
        </span>
      </div>
      <div style={row}>
        <span style={{ color: c.textMuted }}>Goals scored / conceded</span>
        <span>
          {stats.goalsFor} / {stats.goalsAgainst}
        </span>
      </div>
      <div style={{ ...row, border: "none" }}>
        <span style={{ color: c.textMuted }}>Avg pts / match</span>
        <span style={{ color: c.accent, fontWeight: 700 }}>{stats.form_pts}</span>
      </div>
    </div>
  );
}