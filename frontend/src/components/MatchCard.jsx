import { useTheme } from "../context/ThemeContext";

function formatDate(utcDate) {
  if (!utcDate) return "—";
  return new Date(utcDate).toLocaleDateString("en-GB", {
    weekday: "short", day: "numeric", month: "short",
    hour: "2-digit", minute: "2-digit",
  });
}

function Crest({ src, alt }) {
  if (!src) return null;
  return (
    <img
      src={src} alt={alt}
      style={{ width: 22, height: 22, objectFit: "contain", flexShrink: 0 }}
      onError={(e) => { e.target.style.display = "none"; }}
    />
  );
}

export default function MatchCard({ match, selected, loading, onPredict }) {
  const { theme } = useTheme();
  const c = theme.colors;

  const cardStyle = {
    background: selected ? c.bgCard2 : c.bgCard,
    border: `1px solid ${selected ? c.accent : c.border}`,
    borderRadius: 10,
    padding: "12px 14px",
    display: "flex",
    alignItems: "center",
    gap: 10,
    cursor: "pointer",
    transition: "border-color 0.15s, background 0.15s",
  };

  const btnStyle = {
    background: loading && selected ? c.bgCard2 : selected ? c.accent : `${c.accent}1a`,
    color: loading && selected ? c.textMuted : selected ? c.bg : c.accent,
    border: `1px solid ${c.accent}4d`,
    borderRadius: 7,
    padding: "6px 12px",
    fontSize: "0.78rem",
    fontWeight: 700,
    flexShrink: 0,
    cursor: loading ? "not-allowed" : "pointer",
    transition: "all 0.15s",
  };

  return (
    <div style={cardStyle} onClick={() => !loading && onPredict(match)}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <Crest src={match.homeTeam.crest} alt={match.homeTeam.name} />
          <span style={{ fontSize: "0.82rem", fontWeight: 600, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", color: c.text }}>
            {match.homeTeam.name}
          </span>
          <span style={{ fontSize: "0.7rem", color: c.textMuted, fontWeight: 700, flexShrink: 0 }}>vs</span>
          <span style={{ fontSize: "0.82rem", fontWeight: 600, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", color: c.text }}>
            {match.awayTeam.name}
          </span>
          <Crest src={match.awayTeam.crest} alt={match.awayTeam.name} />
        </div>
        <div style={{ fontSize: "0.72rem", color: c.textMuted, marginTop: 3 }}>
          {formatDate(match.utcDate)}
        </div>
      </div>

      <button
        style={btnStyle}
        onClick={(e) => { e.stopPropagation(); !loading && onPredict(match); }}
      >
        {loading && selected ? "..." : "▶"}
      </button>
    </div>
  );
}
