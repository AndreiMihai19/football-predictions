import { useTheme } from "../context/ThemeContext";

export default function StandingsTable({ table }) {
  const { theme } = useTheme();
  const c = theme.colors;

  if (!table || table.length === 0) return null;

  const th = {
    textAlign: "left",
    padding: "10px 14px",
    color: c.textMuted,
    fontWeight: 600,
    fontSize: "0.78rem",
    textTransform: "uppercase",
    letterSpacing: "0.06em",
    borderBottom: `1px solid ${c.border}`,
    whiteSpace: "nowrap",
  };

  const thCenter = { ...th, textAlign: "center", padding: "10px 10px" };

  function rowStyle(idx) {
    return {
      background: idx % 2 === 0 ? "transparent" : `${c.bgCard2}88`,
      borderBottom: `1px solid ${c.border}`,
    };
  }

  function tdStyle(center = false) {
    return {
      padding: "11px 14px",
      textAlign: center ? "center" : "left",
      color: c.text,
    };
  }

  function PositionBadge({ pos }) {
    let color = c.textMuted;
    if (pos <= 4)  color = c.accent;
    if (pos >= 18) color = c.danger;
    return (
      <span style={{ fontWeight: 700, color, display: "inline-block", width: 24, textAlign: "center" }}>
        {pos}
      </span>
    );
  }

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
        <thead>
          <tr>
            <th style={th}>#</th>
            <th style={th}>Team</th>
            <th style={thCenter}>MP</th>
            <th style={thCenter}>W</th>
            <th style={thCenter}>D</th>
            <th style={thCenter}>L</th>
            <th style={thCenter}>GF</th>
            <th style={thCenter}>GA</th>
            <th style={thCenter}>GD</th>
            <th style={thCenter}>Pts</th>
          </tr>
        </thead>
        <tbody>
          {table.map((row, idx) => (
            <tr key={row.teamId} style={rowStyle(idx)}>
              <td style={tdStyle()}><PositionBadge pos={row.position} /></td>
              <td style={{ ...tdStyle(), fontWeight: 600 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  {row.crest && (
                    <img
                      src={row.crest}
                      alt={row.team}
                      style={{ width: 24, height: 24, objectFit: "contain", flexShrink: 0 }}
                      onError={(e) => { e.target.style.display = "none"; }}
                    />
                  )}
                  {row.team}
                </div>
              </td>
              <td style={tdStyle(true)}>{row.playedGames}</td>
              <td style={tdStyle(true)}>{row.won}</td>
              <td style={tdStyle(true)}>{row.draw}</td>
              <td style={tdStyle(true)}>{row.lost}</td>
              <td style={tdStyle(true)}>{row.goalsFor}</td>
              <td style={tdStyle(true)}>{row.goalsAgainst}</td>
              <td style={{ ...tdStyle(true), color: row.goalDifference >= 0 ? c.accent : c.danger }}>
                {row.goalDifference > 0 ? `+${row.goalDifference}` : row.goalDifference}
              </td>
              <td style={{ ...tdStyle(true), fontWeight: 800, color: c.accent }}>{row.points}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
