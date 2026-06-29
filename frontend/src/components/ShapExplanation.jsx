import { useTheme } from "../context/ThemeContext";

const FEATURE_LABELS = {
  rank_diff:               "Rank difference",
  form_pts_diff:           "Form points difference",
  home_goals_scored_home:  "Goals scored home (home)",
  home_goals_scored_away:  "Goals scored home (away)",
  away_goals_scored_home:  "Goals scored away (home)",
  away_goals_scored_away:  "Goals scored away (away)",
  form_gf_diff:            "Goals scored difference",
  form_ga_diff:            "Goals conceded difference",
  home_form_pts:           "Form points (home)",
  away_form_pts:           "Form points (away)",
  home_form_gf:            "Goals per match (home)",
  home_form_ga:            "Goals conceded per match (home)",
  away_form_gf:            "Goals per match (away)",
  away_form_ga:            "Goals conceded per match (away)",
  home_rank:               "Home team rank",
  away_rank:               "Away team rank",
  h2h_home_wins:           "H2H wins (home)",
  h2h_away_wins:           "H2H wins (away)",
};

export default function ShapExplanation({ contributions, prediction }) {
  const { theme } = useTheme();
  const c = theme.colors;

  if (!contributions || contributions.length === 0) return null;

  const top = [...contributions]
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 8);

  const maxAbs = Math.max(...top.map((d) => Math.abs(d.value)));

  return (
    <div style={{ marginTop: 4 }}>
      <div style={{ fontSize: "0.72rem", color: c.textMuted, marginBottom: 12, lineHeight: 1.5 }}>
        Positive values (green) push towards{" "}
        <strong style={{ color: "#00ff87" }}>Home Win</strong>,
        negative (red) towards{" "}
        <strong style={{ color: "#ff4757" }}>Away Win</strong>.
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
        {top.map(({ feature, value }) => {
          const isPositive = value >= 0;
          const color = isPositive ? "#00ff87" : "#ff4757";
          const barPct = maxAbs > 0 ? (Math.abs(value) / maxAbs) * 100 : 0;
          const label = FEATURE_LABELS[feature] ?? feature;

          return (
            <div key={feature} style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{
                width: 170,
                fontSize: "0.75rem",
                color: c.textMuted,
                textAlign: "right",
                flexShrink: 0,
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}>
                {label}
              </div>

              <div style={{ flex: 1, display: "flex", alignItems: "center", position: "relative", height: 16 }}>
                <div style={{
                  position: "absolute",
                  left: "50%",
                  top: 0,
                  bottom: 0,
                  width: 1,
                  background: c.border,
                }} />

                {isPositive ? (
                  <div style={{
                    position: "absolute",
                    left: "50%",
                    width: `${barPct / 2}%`,
                    height: 10,
                    top: 3,
                    background: `${color}cc`,
                    borderRadius: "0 3px 3px 0",
                    transition: "width 0.4s ease",
                  }} />
                ) : (
                  <div style={{
                    position: "absolute",
                    right: `${50}%`,
                    width: `${barPct / 2}%`,
                    height: 10,
                    top: 3,
                    background: `${color}cc`,
                    borderRadius: "3px 0 0 3px",
                    transition: "width 0.4s ease",
                  }} />
                )}
              </div>

              <div style={{
                width: 48,
                fontSize: "0.73rem",
                fontWeight: 700,
                color,
                textAlign: "left",
                flexShrink: 0,
                fontFamily: "monospace",
              }}>
                {value >= 0 ? "+" : ""}{value.toFixed(3)}
              </div>
            </div>
          );
        })}
      </div>

      <div style={{
        display: "flex",
        justifyContent: "center",
        gap: 24,
        marginTop: 12,
        fontSize: "0.7rem",
        color: c.textMuted,
      }}>
        <span><span style={{ color: "#ff4757" }}>◀</span> Away Win</span>
        <span>Home Win <span style={{ color: "#00ff87" }}>▶</span></span>
      </div>
    </div>
  );
}
