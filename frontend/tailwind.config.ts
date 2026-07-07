import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      fontFamily: {
        serif: ['"EB Garamond"', "Georgia", "Times New Roman", "serif"],
        mono: ['"JetBrains Mono"', "Consolas", "monospace"],
        sans: ['"Inter"', "system-ui", "sans-serif"],
      },
      colors: {
        arxiv: {
          red: "#b31b1b",
          dark: "#1a1a1a",
          gray: "#666666",
          lightgray: "#f5f5f5",
          border: "#e0e0e0",
          link: "#1155cc",
          hover: "#1a0dab",
        },
      },
    },
  },
  plugins: [],
};
export default config;
