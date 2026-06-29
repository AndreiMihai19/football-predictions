import { useState, useEffect } from "react";
import axios from "axios";
import { useTheme } from "../context/ThemeContext";
import StandingsTable from "../components/StandingsTable";
import { API_URL } from "../config";

const COMPETITIONS = [
  { code: "PL",  name: "Premier League" },
  { code: "SA",  name: "Serie A" },
  { code: "PD",  name: "La Liga" },
  { code: "BL1", name: "Bundesliga" },
  { code: "FL1", name: "Ligue 1" },
];

export default function Standings() {
  const { theme } = useTheme();
  const c = theme.colors;

  const [competition, setCompetition] = useState("PL");
  const [standings,   setStandings]   = useState(null);
  const [loading,     setLoading]     = useState(false);
  const [error,       setError]       = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    axios.get(`${API_URL}/matches/standings/${competition}`)
      .then(({ data }) => { if (!cancelled) setStandings(data); })
      .catch((err)    => { if (!cancelled) setError(err.response?.data?.error ?? "Failed to load standings."); })
      .finally(()     => { if (!cancelled) setLoading(false); });

    return () => { cancelled = true; };
  }, [competition]);

  const compName = COMPETITIONS.find((comp) => comp.code === competition)?.name ?? competition;

  return (
    <div className="page">
      <h1 className="page-title"><span>{compName}</span> — Standings</h1>

      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 24 }}>
        <select
          style={{ background: c.bgCard, color: c.text, border: `1px solid ${c.border}`, borderRadius: 8, padding: "9px 14px", fontSize: "0.9rem", outline: "none", cursor: "pointer" }}
          value={competition}
          onChange={(e) => setCompetition(e.target.value)}
        >
          {COMPETITIONS.map((comp) => (
            <option key={comp.code} value={comp.code}>{comp.name}</option>
          ))}
        </select>
        {standings && (
          <span style={{ color: c.textMuted, fontSize: "0.85rem" }}>Season {standings.season}</span>
        )}
      </div>

      {loading && <div className="spinner" />}
      {error   && <div className="error-msg">{error}</div>}

      {!loading && !error && standings && (
        <div style={{ background: c.bgCard, border: `1px solid ${c.border}`, borderRadius: 12, overflow: "hidden" }}>
          <StandingsTable table={standings.table} />
          <div style={{ display: "flex", gap: 20, padding: "12px 16px", borderTop: `1px solid ${c.border}`, fontSize: "0.78rem", color: c.textMuted }}>
            <span><span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: c.accent, marginRight: 5 }} />Champions League</span>
            <span><span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: c.danger, marginRight: 5 }} />Relegation</span>
          </div>
        </div>
      )}
    </div>
  );
}
