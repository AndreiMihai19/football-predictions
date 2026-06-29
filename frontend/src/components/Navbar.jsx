import { NavLink } from "react-router-dom";
import ThemeToggle from "./ThemeToggle";
import { useTheme } from "../context/ThemeContext";

export default function Navbar() {
  const { theme } = useTheme();
  const c = theme.colors;

  const styles = {
    nav: {
      background: c.bgCard,
      borderBottom: `1px solid ${c.border}`,
      position: "sticky",
      top: 0,
      zIndex: 100,
    },
    inner: {
      maxWidth: 1280,
      margin: "0 auto",
      padding: "0 24px",
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      height: 60,
    },
    logo: {
      fontSize: "1.2rem",
      fontWeight: 800,
      letterSpacing: "-0.5px",
      color: c.text,
    },
    accent: { color: c.accent },
    links: { display: "flex", gap: 6, alignItems: "center" },
  };

  const linkStyle = ({ isActive }) => ({
    padding: "7px 16px",
    borderRadius: 8,
    fontWeight: 500,
    fontSize: "0.88rem",
    color: isActive ? c.accent : c.textMuted,
    background: isActive ? `${c.accent}15` : "transparent",
    transition: "all 0.2s",
  });

  return (
    <nav style={styles.nav}>
      <div style={styles.inner}>
        <div style={styles.logo}>
          Football<span style={styles.accent}>AI</span>
        </div>
        <div style={styles.links}>
          <NavLink to="/" style={linkStyle}>
            Matches & Predictions
          </NavLink>
          <NavLink to="/standings" style={linkStyle}>
            Standings
          </NavLink>
          <NavLink to="/model" style={linkStyle}>
            Model
          </NavLink>
          <NavLink to="/history" style={linkStyle}>
            History
          </NavLink>
          <div style={{ marginLeft: 8 }}>
            <ThemeToggle />
          </div>
        </div>
      </div>
    </nav>
  );
}