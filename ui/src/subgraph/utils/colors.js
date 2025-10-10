const lightBlue = "#b6d6ff";
const blue = "#90cbfc";
const deepBlue = "#2f63b4";
const black = "#000000";
const white = "#ffffff";
const lightGrey = "#f1f1fa";
const darkGrey = "#a0a0a8";
const grey = "#d5d6da";
const lightOrange = "#ffd9b3";
const deepDarkGrey = "#4b5563";
const orange = "#ff8000";
const textGrey = "#666";
const textTitle = black;

const globalColors = {
  blue,
  lightBlue,
  deepBlue,
  black,
  white,
  lightGrey,
  darkGrey,
  grey,
  lightOrange,
  orange,
  textGrey,
  textTitle,
  deepDarkGrey,
};

function hexToRgb(hex) {
  const normalized = hex.replace(/^#/, "").trim();
  if (![3, 6].includes(normalized.length)) throw new Error("Invalid hex color");
  const full =
    normalized.length === 3
      ? normalized
          .split("")
          .map((c) => c + c)
          .join("")
      : normalized;
  const r = parseInt(full.slice(0, 2), 16);
  const g = parseInt(full.slice(2, 4), 16);
  const b = parseInt(full.slice(4, 6), 16);
  if ([r, g, b].some((v) => Number.isNaN(v)))
    throw new Error("Invalid hex color");
  return { r, g, b };
}

function rgbToHex(r, g, b) {
  const toHex = (v) => {
    const clamped = Math.max(0, Math.min(255, Math.round(v)));
    return clamped.toString(16).padStart(2, "0");
  };
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function interpolateNumber(a, b, t) {
  return a + (b - a) * t;
}

function generateColorScale(startHex, endHex, numTicks) {
  if (typeof startHex !== "string" || typeof endHex !== "string") {
    throw new Error("start and end must be hex color strings");
  }
  const n = Number(numTicks);
  if (!Number.isFinite(n) || n < 2) {
    throw new Error("numTicks must be a number >= 2");
  }
  const start = hexToRgb(startHex);
  const end = hexToRgb(endHex);
  const colors = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const r = interpolateNumber(start.r, end.r, t);
    const g = interpolateNumber(start.g, end.g, t);
    const b = interpolateNumber(start.b, end.b, t);
    colors.push(rgbToHex(r, g, b));
  }
  return colors;
}

function getColorScale(startHex, endHex, numTicks) {
  return generateColorScale(startHex, endHex, numTicks);
}
