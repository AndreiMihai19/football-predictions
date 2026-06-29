import { useTheme } from "../context/ThemeContext";

const s = {
  button: {
    background: "transparent",
    border: "1px solid var(--border, #2a2a45)",
    borderRadius: 8,
    padding: "6px 10px",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    gap: 6,
    fontSize: "0.82rem",
    color: "var(--text-muted, #8888aa)",
    transition: "all 0.2s",
  },
  icon: {
    fontSize: "1rem",
    transition: "transform 0.3s",
  },
};

export default function ThemeToggle({ showLabel = false }) {
  const { themeName, toggleTheme } = useTheme();

  const isDark = themeName === "dark";

  return (
    <button
      style={s.button}
      onClick={toggleTheme}
      title={isDark ? "Switch to light mode" : "Switch to dark mode"}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = "var(--accent, #00ff87)";
        e.currentTarget.style.color = "var(--text, #e8e8f0)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = "var(--border, #2a2a45)";
        e.currentTarget.style.color = "var(--text-muted, #8888aa)";
      }}
    >
      <span style={s.icon}>
        {isDark ? "Light Mode" : "Dark Mode"}
      </span>
      {showLabel && (
        <span>{isDark ? "Dark" : "Light"}</span>
      )}
    </button>
  );
}
