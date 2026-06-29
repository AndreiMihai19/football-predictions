import { useTheme } from "../context/ThemeContext";

export default function TeamComparison({ homeTeam, awayTeam, h2h }) {
  const { theme } = useTheme();
  const c = theme.colors;

  if (!homeTeam?.stats || !awayTeam?.stats) return null;

  const homeStats = homeTeam.stats;
  const awayStats = awayTeam.stats;

  const metrics = [
    { label: "Form Pts",    home: homeStats.form_pts || 0, away: awayStats.form_pts || 0, max: 3,    format: (v) => v.toFixed(1) },
    { label: "Goals/Match", home: homeStats.form_gf  || 0, away: awayStats.form_gf  || 0, max: Math.max(homeStats.form_gf || 0, awayStats.form_gf || 0, 2), format: (v) => v.toFixed(1) },
    { label: "Goals Agnst", home: homeStats.form_ga  || 0, away: awayStats.form_ga  || 0, max: Math.max(homeStats.form_ga || 0, awayStats.form_ga || 0, 2), format: (v) => v.toFixed(1), reverse: true },
    { label: "Wins",        home: homeStats.wins     || 0, away: awayStats.wins     || 0, max: 5,    format: (v) => v.toString() },
    { label: "Rank",        home: homeTeam.rank      || 10, away: awayTeam.rank     || 10, max: 20,  format: (v) => `#${v}`, reverse: true },
  ];

  const getBarWidth = (value, max) => Math.min(100, (value / max) * 100);

  const getColor = (homeVal, awayVal, reverse = false) => {
    if (reverse) {
      if (homeVal < awayVal) return { home: c.accent, away: c.danger };
      if (homeVal > awayVal) return { home: c.danger, away: c.accent };
    } else {
      if (homeVal > awayVal) return { home: c.accent, away: c.danger };
      if (homeVal < awayVal) return { home: c.danger, away: c.accent };
    }
    return { home: c.warning, away: c.warning };
  };

  return (
    <div style={{ background: c.bgCard2, borderRadius: 12, padding: 16, marginTop: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, paddingBottom: 12, borderBottom: `1px solid ${c.border}` }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {homeTeam.crest && <img src={homeTeam.crest} alt="" style={{ width: 28, height: 28, objectFit: "contain" }} />}
          <span style={{ fontSize: "0.85rem", fontWeight: 600, color: c.text, maxWidth: 100, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {homeTeam.name || "Home"}
          </span>
        </div>
        <span style={{ fontSize: "0.7rem", color: c.textMuted, fontWeight: 600, padding: "4px 10px", background: c.bg, borderRadius: 10 }}>VS</span>
        <div style={{ display: "flex", alignItems: "center", flexDirection: "row-reverse", gap: 8 }}>
          {awayTeam.crest && <img src={awayTeam.crest} alt="" style={{ width: 28, height: 28, objectFit: "contain" }} />}
          <span style={{ fontSize: "0.85rem", fontWeight: 600, color: c.text, maxWidth: 100, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", textAlign: "right" }}>
            {awayTeam.name || "Away"}
          </span>
        </div>
      </div>

      {metrics.map((metric) => {
        const colors = getColor(metric.home, metric.away, metric.reverse);
        const homeWidth = getBarWidth(metric.home, metric.max);
        const awayWidth = getBarWidth(metric.away, metric.max);

        return (
          <div key={metric.label} style={{ display: "flex", alignItems: "center", marginBottom: 10, gap: 8 }}>
            <span style={{ fontSize: "0.75rem", fontWeight: 600, color: colors.home, minWidth: 30, textAlign: "right" }}>
              {metric.format(metric.home)}
            </span>
            <div style={{ flex: 1, height: 8, background: c.bg, borderRadius: 4, overflow: "hidden" }}>
              <div style={{ height: "100%", borderRadius: "4px 0 0 4px", transition: "width 0.5s ease-out", float: "right", width: `${homeWidth}%`, background: colors.home }} />
            </div>
            <span style={{ width: 80, fontSize: "0.72rem", color: c.textMuted, textAlign: "center", flexShrink: 0 }}>{metric.label}</span>
            <div style={{ flex: 1, height: 8, background: c.bg, borderRadius: 4, overflow: "hidden" }}>
              <div style={{ height: "100%", borderRadius: "0 4px 4px 0", transition: "width 0.5s ease-out", width: `${awayWidth}%`, background: colors.away }} />
            </div>
            <span style={{ fontSize: "0.75rem", fontWeight: 600, color: colors.away, minWidth: 30 }}>
              {metric.format(metric.away)}
            </span>
          </div>
        );
      })}

      <div style={{ display: "flex", justifyContent: "center", gap: 20, marginTop: 12, paddingTop: 12, borderTop: `1px solid ${c.border}`, fontSize: "0.72rem", color: c.textMuted }}>
        {[["Better", c.accent], ["Equal", c.warning], ["Worse", c.danger]].map(([label, color]) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 10, height: 10, borderRadius: 3, background: color }} />
            <span>{label}</span>
          </div>
        ))}
      </div>

      {h2h && h2h.h2h_total > 0 && (
        <div style={{ marginTop: 16, paddingTop: 14, borderTop: `1px solid ${c.border}` }}>
          <div style={{ fontSize: "0.72rem", color: c.textMuted, marginBottom: 10, textTransform: "uppercase", letterSpacing: "0.06em" }}>
            Head-to-Head — {h2h.h2h_total} matches
          </div>
          <div style={{ display: "flex", gap: 4, marginBottom: 10, height: 8, borderRadius: 4, overflow: "hidden" }}>
            {h2h.h2h_home_wins > 0 && <div style={{ flex: h2h.h2h_home_wins, background: c.accent, borderRadius: "4px 0 0 4px" }} />}
            {h2h.h2h_draws > 0 && <div style={{ flex: h2h.h2h_draws, background: c.warning }} />}
            {h2h.h2h_away_wins > 0 && <div style={{ flex: h2h.h2h_away_wins, background: c.danger, borderRadius: "0 4px 4px 0" }} />}
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", marginBottom: 12 }}>
            <span style={{ color: c.accent }}>{homeTeam.name?.split(" ")[0]}: {h2h.h2h_home_wins}W</span>
            {h2h.h2h_draws > 0 && <span style={{ color: c.warning }}>{h2h.h2h_draws}D</span>}
            <span style={{ color: c.danger }}>{awayTeam.name?.split(" ")[0]}: {h2h.h2h_away_wins}W</span>
          </div>
          {h2h.recentH2H?.map((m, i) => (
            <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "0.72rem", color: c.textMuted, padding: "4px 0", borderBottom: i < h2h.recentH2H.length - 1 ? `1px solid ${c.bg}` : "none" }}>
              <span style={{ flex: 1, textAlign: "right", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{m.home}</span>
              <span style={{ margin: "0 8px", color: c.text, fontWeight: 700, fontFamily: "monospace", flexShrink: 0 }}>{m.score}</span>
              <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{m.away}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
