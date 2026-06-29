import { useTheme } from "../context/ThemeContext";

export default function ConfidenceMeter({
  confidence,
  confidencePercent,
  probabilityHomeWin,
  probabilityAwayWin,
  prediction
}) {
  const { theme } = useTheme();
  const c = theme.colors;

  const getConfidenceColor = () => {
    if (confidence === "High")   return c.accent;
    if (confidence === "Medium") return c.warning;
    return c.danger;
  };

  const color   = getConfidenceColor();
  const percent = confidencePercent || Math.round(Math.max(probabilityHomeWin, probabilityAwayWin) * 100);
  const homePercent = Math.round((probabilityHomeWin || 0) * 100);
  const awayPercent = Math.round((probabilityAwayWin || 0) * 100);

  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <span style={{ fontSize: "0.82rem", color: c.textMuted, fontWeight: 500 }}>Prediction Confidence</span>
        <span style={{ fontSize: "0.9rem", fontWeight: 700, color }}>{percent}% ({confidence || "Unknown"})</span>
      </div>

      <div style={{ height: 12, background: c.bgCard2, borderRadius: 6, overflow: "hidden" }}>
        <div style={{
          height: "100%",
          borderRadius: 6,
          width: `${percent}%`,
          background: `linear-gradient(90deg, ${color}80, ${color})`,
          transition: "width 0.8s ease-out",
        }} />
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4, fontSize: "0.68rem", color: c.textMuted }}>
        {["50%","60%","70%","80%","90%","100%"].map(v => <span key={v}>{v}</span>)}
      </div>

      <div style={{ display: "flex", gap: 16, marginTop: 12 }}>
        <div style={{
          flex: 1,
          background: c.bgCard2,
          borderRadius: 8,
          padding: "10px 12px",
          border: prediction === 1 || prediction === "Home Win" ? `1px solid ${c.accent}` : "1px solid transparent",
        }}>
          <div style={{ fontSize: "0.72rem", color: c.textMuted, marginBottom: 4 }}>Home Win</div>
          <div style={{ fontSize: "1rem", fontWeight: 700, color: homePercent > awayPercent ? c.accent : c.textMuted }}>
            {homePercent}%
          </div>
        </div>

        <div style={{
          flex: 1,
          background: c.bgCard2,
          borderRadius: 8,
          padding: "10px 12px",
          border: prediction === 0 || prediction === "Away Win" ? `1px solid ${c.danger}` : "1px solid transparent",
        }}>
          <div style={{ fontSize: "0.72rem", color: c.textMuted, marginBottom: 4 }}>Away Win</div>
          <div style={{ fontSize: "1rem", fontWeight: 700, color: awayPercent > homePercent ? c.danger : c.textMuted }}>
            {awayPercent}%
          </div>
        </div>
      </div>

      <style>{`@keyframes shimmer { 0%{transform:translateX(-100%)} 100%{transform:translateX(100%)} }`}</style>
    </div>
  );
}
