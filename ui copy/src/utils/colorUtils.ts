/**
 * Color utility functions for graph visualization
 * Used for mixing colors when merging multiple graphs
 */

export interface RGB {
  r: number;
  g: number;
  b: number;
}

export interface HSL {
  h: number;
  s: number;
  l: number;
}

/**
 * Convert hex color to RGB
 */
export const hexToRgb = (hex: string): RGB | null => {
  const h = hex.trim().replace("#", "");
  if (h.length !== 6) return null;
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  if ([r, g, b].some((v) => Number.isNaN(v))) return null;
  return { r, g, b };
};

/**
 * Convert RGB to hex color
 */
export const rgbToHex = (rgb: RGB): string => {
  const clamp = (v: number) => Math.max(0, Math.min(255, Math.round(v)));
  const to2 = (v: number) => clamp(v).toString(16).padStart(2, "0");
  return `#${to2(rgb.r)}${to2(rgb.g)}${to2(rgb.b)}`.toUpperCase();
};

/**
 * Convert RGB to HSL
 */
export const rgbToHsl = (rgb: RGB): HSL => {
  const r = rgb.r / 255;
  const g = rgb.g / 255;
  const b = rgb.b / 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const d = max - min;
  const l = (max + min) / 2;
  let h = 0;
  let s = 0;
  if (d !== 0) {
    s = d / (1 - Math.abs(2 * l - 1));
    switch (max) {
      case r:
        h = ((g - b) / d) % 6;
        break;
      case g:
        h = (b - r) / d + 2;
        break;
      case b:
        h = (r - g) / d + 4;
        break;
    }
    h *= 60;
    if (h < 0) h += 360;
  }
  return { h, s, l };
};

/**
 * Convert HSL to RGB
 */
export const hslToRgb = (hsl: HSL): RGB => {
  const h = ((hsl.h % 360) + 360) % 360;
  const s = Math.max(0, Math.min(1, hsl.s));
  const l = Math.max(0, Math.min(1, hsl.l));
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let rp = 0;
  let gp = 0;
  let bp = 0;
  if (0 <= h && h < 60) [rp, gp, bp] = [c, x, 0];
  else if (60 <= h && h < 120) [rp, gp, bp] = [x, c, 0];
  else if (120 <= h && h < 180) [rp, gp, bp] = [0, c, x];
  else if (180 <= h && h < 240) [rp, gp, bp] = [0, x, c];
  else if (240 <= h && h < 300) [rp, gp, bp] = [x, 0, c];
  else [rp, gp, bp] = [c, 0, x];
  return { r: (rp + m) * 255, g: (gp + m) * 255, b: (bp + m) * 255 };
};

/**
 * Mix multiple hex colors using HSL circular averaging
 * Produces vivid colors by averaging hue vectors and enhancing saturation/lightness
 */
export const mixHexColorsVivid = (hexColors: string[]): string | null => {
  const rgbs = hexColors.map(hexToRgb).filter(Boolean) as RGB[];
  if (rgbs.length === 0) return null;
  const hsls = rgbs.map(rgbToHsl);

  // Hue is a circular angle: use vector averaging to avoid 350°/10° cross-zero issues
  let x = 0;
  let y = 0;
  for (const hsl of hsls) {
    const rad = (hsl.h * Math.PI) / 180;
    x += Math.cos(rad);
    y += Math.sin(rad);
  }
  let hue = 0;
  if (x !== 0 || y !== 0) {
    hue = (Math.atan2(y, x) * 180) / Math.PI;
    if (hue < 0) hue += 360;
  }

  const sAvg = hsls.reduce((acc, v) => acc + v.s, 0) / hsls.length;
  const lAvg = hsls.reduce((acc, v) => acc + v.l, 0) / hsls.length;

  // Empirical parameters: increase saturation, control brightness in middle range
  // to avoid gray appearance on white background
  const s = Math.max(0.65, Math.min(0.95, sAvg));
  const l = Math.max(0.42, Math.min(0.62, lAvg));

  return rgbToHex(hslToRgb({ h: hue, s, l }));
};
