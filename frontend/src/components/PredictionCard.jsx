import ConfidenceMeter from "./ConfidenceMeter";
import TeamComparison from "./TeamComparison";
import ShapExplanation from "./ShapExplanation";
import { useTheme } from "../context/ThemeContext";

const CONFIDENCE_STYLE = {
  High:   { background: "rgba(0,255,135,0.15)",  color: "#00ff87",  border: "1px solid rgba(0,255,135,0.4)" },
  Medium: { background: "rgba(255,165,2,0.15)",  color: "#ffa502",  border: "1px solid rgba(255,165,2,0.4)" },
  Low:    { background: "rgba(255,71,87,0.15)",   color: "#ff4757",  border: "1px solid rgba(255,71,87,0.4)" },
};

function CrestImg({ src, alt, size }) {
  if (!src) return null;
  return (
    <img
      src={src}
      alt={alt}
      style={{ width: size, height: size, objectFit: "contain", marginBottom: 8 }}
      onError={(e) => { e.target.style.display = "none"; }}
    />
  );
}

export default function PredictionCard({ data, homeTeamName, awayTeamName, accuracy }) {
  const { theme } = useTheme();
  const c = theme.colors;
  
  const s = {
    card: {
      background: c.bgCard,
      border: `1px solid ${c.border}`,
      borderRadius: 12,
      padding: 24,
    },
    header: {
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      marginBottom: 20,
      flexWrap: "wrap",
      gap: 12,
    },
    title: {
      fontSize: "0.8rem",
      color: c.textMuted,
      textTransform: "uppercase",
      letterSpacing: "0.08em",
      marginBottom: 4,
    },
    result: {
      fontSize: "1.5rem",
      fontWeight: 800,
      color: c.accent,
    },
    badge: {
      padding: "5px 14px",
      borderRadius: 20,
      fontSize: "0.8rem",
      fontWeight: 700,
    },
    teamsRow: {
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      marginBottom: 20,
      gap: 12,
    },
    teamBox: {
      flex: 1,
      background: c.bgCard2,
      borderRadius: 10,
      padding: "14px 18px",
      textAlign: "center",
    },
    teamName: {
      fontWeight: 700,
      fontSize: "1rem",
      marginBottom: 4,
      color: c.text,
    },
    teamProb: {
      fontSize: "1.6rem",
      fontWeight: 800,
    },
    teamLabel: {
      fontSize: "0.75rem",
      color: c.textMuted,
      marginTop: 2,
    },
    vsLabel: {
      color: c.textMuted,
      fontWeight: 700,
      fontSize: "0.85rem",
    },
    footer: {
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      fontSize: "0.78rem",
      color: c.textMuted,
      borderTop: `1px solid ${c.border}`,
      paddingTop: 14,
      marginTop: 16,
      flexWrap: "wrap",
      gap: 8,
    },
    divider: {
      height: 1,
      background: c.border,
      margin: "16px 0",
    },
    sectionTitle: {
      fontSize: "0.75rem",
      color: c.textMuted,
      textTransform: "uppercase",
      letterSpacing: "0.08em",
      marginBottom: 12,
      fontWeight: 600,
    },
  };

  const { prediction } = data;
  const homeCrest = data.homeTeam?.crest ?? null;
  const awayCrest = data.awayTeam?.crest ?? null;
  const homeWinPct = Math.round(prediction.probability_home_win * 100);
  const awayWinPct = Math.round(prediction.probability_away_win * 100);
  const conf = prediction.confidence;
  const confPercent = prediction.confidencePercent || Math.max(homeWinPct, awayWinPct);

  return (
    <div style={s.card}>
      <div style={s.header}>
        <div>
          <div style={s.title}>Model Prediction</div>
          <div style={s.result}>{prediction.label}</div>
        </div>
        <div style={{ ...s.badge, ...CONFIDENCE_STYLE[conf] }}>
          {confPercent}% • {conf}
        </div>
      </div>

      <div style={s.teamsRow}>
        <div style={{
          ...s.teamBox,
          border: prediction.prediction === 1 ? `2px solid ${c.accent}` : "2px solid transparent",
        }}>
          <CrestImg src={homeCrest} alt={homeTeamName} size={48} />
          <div style={s.teamName}>{homeTeamName}</div>
          <div style={{ 
            ...s.teamProb, 
            color: prediction.prediction === 1 ? c.accent : c.text 
          }}>
            {homeWinPct}%
          </div>
          <div style={s.teamLabel}>
            Home {data.homeTeam?.rank ? `• #${data.homeTeam.rank}` : ""}
          </div>
        </div>

        <div style={s.vsLabel}>VS</div>

        <div style={{
          ...s.teamBox,
          border: prediction.prediction === 0 ? `2px solid ${c.danger}` : "2px solid transparent",
        }}>
          <CrestImg src={awayCrest} alt={awayTeamName} size={48} />
          <div style={s.teamName}>{awayTeamName}</div>
          <div style={{ 
            ...s.teamProb, 
            color: prediction.prediction === 0 ? c.danger : c.text 
          }}>
            {awayWinPct}%
          </div>
          <div style={s.teamLabel}>
            Away {data.awayTeam?.rank ? `• #${data.awayTeam.rank}` : ""}
          </div>
        </div>
      </div>

      <div style={s.sectionTitle}>Confidence Analysis</div>
      <ConfidenceMeter
        confidence={conf}
        confidencePercent={confPercent}
        probabilityHomeWin={prediction.probability_home_win}
        probabilityAwayWin={prediction.probability_away_win}
        prediction={prediction.prediction}
      />

      <div style={s.divider} />
      <div style={s.sectionTitle}>⚖️ Head-to-Head Comparison</div>
      <TeamComparison
        homeTeam={data.homeTeam}
        awayTeam={data.awayTeam}
        h2h={data.h2h}
      />

      {prediction.shap_contributions && (
        <>
          <div style={s.divider} />
          <div style={s.sectionTitle}>🔍 Why this prediction? (SHAP)</div>
          <ShapExplanation
            contributions={prediction.shap_contributions}
            prediction={prediction.prediction}
          />
        </>
      )}

      <div style={s.footer}>
        <span>Ensemble Model (XGBoost + LightGBM + CatBoost)</span>
        <span>
          Accuracy: <strong style={{ color: c.accent }}>
            {accuracy != null ? `${accuracy}%` : "—"}
          </strong>
        </span>
        <span>Calibrated Probabilities</span>
      </div>
    </div>
  );
}