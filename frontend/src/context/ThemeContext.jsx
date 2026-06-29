import { createContext, useContext, useState, useEffect } from "react";

export const themes = {
  dark: {
    name: "dark",
    colors: {
      bg: "#0f0f1a",
      bgRoot: "#0f0f1a",
      bgCard: "#1a1a2e",
      bgCard2: "#16213e",
      accent: "#00ff87",
      accentDim: "#00cc6a",
      text: "#e8e8f0",
      textMuted: "#8888aa",
      border: "#2a2a45",
      danger: "#ff4757",
      warning: "#ffa502",
    },
  },
  light: {
    name: "light",
    colors: {
      bg: "#f5f5f7",
      bgRoot: "#f5f5f7",
      bgCard: "#ffffff",
      bgCard2: "#f0f0f5",
      accent: "#00aa55",
      accentDim: "#008844",
      text: "#1a1a2e",
      textMuted: "#666680",
      border: "#e0e0e8",
      danger: "#e74c3c",
      warning: "#f39c12",
    },
  },
};

const ThemeContext = createContext({
  theme: themes.dark,
  themeName: "dark",
  toggleTheme: () => {},
  setTheme: () => {},
});

export function ThemeProvider({ children }) {
  const [themeName, setThemeName] = useState(() => {
    const saved = localStorage.getItem("football-theme");
    return saved || "dark";
  });

  const theme = themes[themeName] || themes.dark;

  useEffect(() => {
    localStorage.setItem("football-theme", themeName);

    const root = document.documentElement;
    Object.entries(theme.colors).forEach(([key, value]) => {
      const cssVar = `--${key.replace(/([A-Z])/g, "-$1").toLowerCase()}`;
      root.style.setProperty(cssVar, value);
    });

    document.body.style.background = theme.colors.bgRoot;
    document.body.style.color = theme.colors.text;
  }, [themeName, theme]);

  const toggleTheme = () => {
    setThemeName((prev) => (prev === "dark" ? "light" : "dark"));
  };

  const setTheme = (name) => {
    if (themes[name]) {
      setThemeName(name);
    }
  };

  return (
    <ThemeContext.Provider value={{ theme, themeName, toggleTheme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}

export default ThemeContext;
