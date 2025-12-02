
/**
 * SVG Lines and Curves Utility
 * 
 * This module provides a comprehensive line drawing system for SVG graphics using D3.js.
 * It supports three main drawing modes:
 * 
 * 1. STRAIGHT LINES
 *    - Simple line between two points
 *    - Generated when no middlePoints are provided
 *    - SVG path: "M x1 y1 L x2 y2"
 * 
 * 2. POLYLINES  
 *    - Connected line segments through multiple points
 *    - Generated when middlePoints provided but curveBeta = 0
 *    - SVG path: "M x1 y1 L x2 y2 L x3 y3 ..."
 * 
 * 3. BEZIER CURVES
 *    - Smooth curved lines with rounded corners
 *    - Generated when middlePoints provided and curveBeta > 0
 *    - Uses cubic bezier curves for smooth transitions
 *    - SVG path: "M x1 y1 L entry C c1 c2 exit L ..."
 * 
 * FEATURES:
 * - Optional arrow markers at start/end points
 * - Automatic line shortening to prevent arrow overlap
 * - Customizable styling (color, width, opacity)
 * - Dashed lines via stroke dash patterns
 * - Efficient marker reuse to avoid SVG bloat
 * - Support for complex multi-segment paths
 * 
 * USAGE:
 * ```javascript
 * // Basic straight line
 * line(svg, {x: 10, y: 10}, {x: 100, y: 50});
 * 
 * // Curved path with arrows
 * line(svg, {x: 0, y: 0}, {x: 200, y: 100}, {
 *   middlePoints: [{x: 50, y: 20}, {x: 150, y: 80}],
 *   curveBeta: 15,
 *   arrowStart: true,
 *   arrowEnd: true,
 *   stroke: 'blue'
 * });
 * ```
 * 
 * @requires d3.js - Must be loaded before using this module
 * @author Aiology-Figures
 * @version 1.0.0
 */

/**
 * Creates SVG lines and curves with optional arrows and styling
 * 
 * The line function supports three drawing modes:
 * 1. Straight lines - simple line between two points
 * 2. Polylines - connected line segments through multiple points
 * 3. Bezier curves - smooth curved lines with rounded corners
 * 
 * @param {d3.Selection} container - D3 selection of SVG container element
 * @param {Object} start - Starting point {x: number, y: number}
 * @param {Object} end - Ending point {x: number, y: number}
 * @param {Object} options - Configuration options
 * @param {string} options.type - Line type ('line' - default)
 * @param {number} options.beta - Legacy parameter (unused)
 * @param {number} options.curveBeta - Curve radius for rounded corners (0 = sharp corners, >0 = smooth curves)
 * @param {Array} options.middlePoints - Array of intermediate points [{x, y}, ...] for polylines/curves
 * @param {string} options.stroke - Stroke color (default: '#000')
 * @param {number} options.strokeWidth - Line thickness (default: 2)
 * @param {number} options.opacity - Line opacity 0-1 (default: 1)
 * @param {string|Array<number>} options.dasharray - Stroke dash pattern (e.g., '6,4' or [6,4])
 * @param {number} options.dashoffset - Stroke dash offset (default: 0)
 * @param {string} options.fill - Fill color (default: 'none')
 * @param {boolean} options.arrowStart - Add arrow at start (default: false)
 * @param {boolean} options.arrowEnd - Add arrow at end (default: false)
 * @param {number} options.arrowSize - Arrow size in pixels (default: 6)
 * @param {string} options.arrowColor - Arrow color (defaults to stroke color)
 * 
 * @returns {d3.Selection} D3 selection of the created path element
 * 
 * @example
 * // Basic straight line
 * const svg = d3.select('svg');
 * line(svg, {x: 10, y: 10}, {x: 100, y: 50});
 * 
 * @example
 * // Polyline with multiple points
 * line(svg, {x: 10, y: 10}, {x: 100, y: 100}, {
 *   middlePoints: [{x: 50, y: 20}, {x: 80, y: 60}],
 *   stroke: 'blue',
 *   strokeWidth: 3
 * });
 * 
 * @example
 * // Curved line with rounded corners
 * line(svg, {x: 10, y: 10}, {x: 100, y: 100}, {
 *   middlePoints: [{x: 50, y: 20}, {x: 80, y: 60}],
 *   curveBeta: 15,  // Smooth rounded corners
 *   stroke: 'red',
 *   arrowStart: true,
 *   arrowEnd: true
 * });
 * 
 * @example
 * // Arrow with custom styling
 * line(svg, {x: 0, y: 0}, {x: 200, y: 100}, {
 *   stroke: '#333',
 *   strokeWidth: 4,
 *   arrowStart: true,
 *   arrowEnd: true,
 *   arrowSize: 10,
 *   arrowColor: '#ff0000'
 * });
 */
function line(container, start, end, options = {}) {
  if (typeof d3 === 'undefined') {
    throw new Error('d3 must be loaded before using line()');
  }

  const opts = Object.assign({
    type: 'line',
    beta: 0.3,
    curveBeta: 0,
    middlePoints: null,
    stroke: '#000',
    strokeWidth: 2,
    opacity: 1,
    dasharray: null,
    dashoffset: 0,
    fill: 'none',
    arrowStart: false,
    arrowEnd: false,
    arrowSize: 6,
    class: ''
  }, options || {});

  const x1 = start.x, y1 = start.y;
  const x2 = end.x, y2 = end.y;

  // Ensure defs for markers exists on the root SVG
  const rootSvg = container.node().ownerSVGElement || container;
  const defs = ensureDefs(d3.select(rootSvg));

  // Prepare marker ids unique to color/size/width to avoid duplication
  const strokeColor = opts.stroke;
  const strokeWidth = +opts.strokeWidth;
  const arrowColor = opts.arrowColor || strokeColor;
  const arrowSize = +opts.arrowSize;
  const markerStartId = `arrow-start-${sanitizeId(arrowColor)}-${arrowSize}-${strokeWidth}`;
  const markerEndId = `arrow-end-${sanitizeId(arrowColor)}-${arrowSize}-${strokeWidth}`;

  if (opts.arrowStart) {
    ensureArrowMarker(defs, markerStartId, arrowColor, arrowSize, strokeWidth, true);
  }
  if (opts.arrowEnd) {
    ensureArrowMarker(defs, markerEndId, arrowColor, arrowSize, strokeWidth, false);
  }

  // Optionally shorten the line so arrow markers don't overshoot
  let sx = x1, sy = y1, ex = x2, ey = y2;

  const middlePoints = Array.isArray(opts.middlePoints) ? opts.middlePoints : null;
  const curveBeta = typeof opts.curveBeta === 'number' ? Math.max(0, opts.curveBeta) : 0;

  let d;
  if (!middlePoints || middlePoints.length === 0) {
    // Straight line case: shorten along global direction
    if (opts.arrowStart || opts.arrowEnd) {
      const dx = x2 - x1;
      const dy = y2 - y1;
      const len = Math.hypot(dx, dy) || 1;
      const ux = dx / len;
      const uy = dy / len;
      if (opts.arrowStart) { sx = x1 + ux * arrowSize; sy = y1 + uy * arrowSize; }
      if (opts.arrowEnd) { ex = x2 - ux * arrowSize; ey = y2 - uy * arrowSize; }
    }
    d = `M ${sx} ${sy} L ${ex} ${ey}`;
  } else if (curveBeta === 0) {
    // Polyline case: shorten along first and last segment directions
    const polyPts = [{ x: x1, y: y1 }, ...middlePoints, { x: x2, y: y2 }];
    if (opts.arrowStart && polyPts.length >= 2) {
      const n = polyPts[0];
      const t = polyPts[1];
      const dx = t.x - n.x, dy = t.y - n.y;
      const len = Math.hypot(dx, dy) || 1;
      sx = n.x + (dx / len) * arrowSize;
      sy = n.y + (dy / len) * arrowSize;
    }
    if (opts.arrowEnd && polyPts.length >= 2) {
      const p = polyPts[polyPts.length - 2];
      const n = polyPts[polyPts.length - 1];
      const dx = n.x - p.x, dy = n.y - p.y;
      const len = Math.hypot(dx, dy) || 1;
      ex = n.x - (dx / len) * arrowSize;
      ey = n.y - (dy / len) * arrowSize;
    }
    const pts = [{ x: sx, y: sy }, ...middlePoints, { x: ex, y: ey }];
    d = `M ${pts[0].x} ${pts[0].y}` + pts.slice(1).map(p => ` L ${p.x} ${p.y}`).join('');
  } else {
    // Curved case: shorten using tangents at start (toward first entry) and end (from last exit)
    const basePts = [{ x: x1, y: y1 }, ...middlePoints, { x: x2, y: y2 }];
    // Compute first and last rounded corners for tangent directions
    const firstCorner = computeRoundedCorner(
      basePts[0],
      basePts[1],
      basePts[2] || basePts[basePts.length - 1],
      curveBeta
    );
    const lastCorner = computeRoundedCorner(
      basePts[basePts.length - 3] || basePts[0],
      basePts[basePts.length - 2],
      basePts[basePts.length - 1],
      curveBeta
    );
    if (opts.arrowStart) {
      const dx = firstCorner.entry.x - x1;
      const dy = firstCorner.entry.y - y1;
      const len = Math.hypot(dx, dy) || 1;
      const shorten = Math.min(arrowSize, len - 1e-6);
      sx = x1 + (dx / len) * shorten;
      sy = y1 + (dy / len) * shorten;
    }
    if (opts.arrowEnd) {
      const dx = x2 - lastCorner.exit.x;
      const dy = y2 - lastCorner.exit.y;
      const len = Math.hypot(dx, dy) || 1;
      const shorten = Math.min(arrowSize, len - 1e-6);
      ex = x2 - (dx / len) * shorten;
      ey = y2 - (dy / len) * shorten;
    }
    const pts = [{ x: sx, y: sy }, ...middlePoints, { x: ex, y: ey }];
    let parts = [`M ${pts[0].x} ${pts[0].y}`];
    let cursor = { x: pts[0].x, y: pts[0].y };

    for (let i = 0; i < pts.length - 2; i++) {
      const prev = pts[i];
      const curr = pts[i + 1];
      const next = pts[i + 2];

      const rc = computeRoundedCorner(prev, curr, next, curveBeta);
      parts.push(` L ${rc.entry.x} ${rc.entry.y}`);
      parts.push(` C ${rc.c1.x} ${rc.c1.y}, ${rc.c2.x} ${rc.c2.y}, ${rc.exit.x} ${rc.exit.y}`);
      cursor = rc.exit;
    }
    const last = pts[pts.length - 1];
    parts.push(` L ${last.x} ${last.y}`);
    d = parts.join('');
  }

  const path = container.append('path')
    .attr('d', d)
    .attr('fill', opts.fill)
    .attr('stroke', opts.stroke)
    .attr('stroke-width', opts.strokeWidth)
    .attr('opacity', opts.opacity)
    .attr('class', opts.class);

  if (opts.dasharray != null) {
    const dashPattern = Array.isArray(opts.dasharray)
      ? opts.dasharray.join(' ')
      : String(opts.dasharray);
    path.attr('stroke-dasharray', dashPattern);
  }
  if (opts.dashoffset) {
    path.attr('stroke-dashoffset', opts.dashoffset);
  }

  if (opts.arrowStart) path.attr('marker-start', `url(#${markerStartId})`);
  if (opts.arrowEnd) path.attr('marker-end', `url(#${markerEndId})`);

  return path;
}

/**
 * Draws a smooth single-arch curved line between two points.
 *
 * Arguments mirror `line(container, start, end, options)` but the curve's
 * middle control point is computed automatically by offsetting the midpoint
 * perpendicular to the straight line from start to end.
 *
 * Options specific to curvature:
 * - curveSide | bendSide: 'left' | 'right' (default: 'left') relative to the
 *   direction from start -> end
 * - curveDistance | bendDistance | offset: number (default: 40) distance in px
 *   from the straight line to the curve apex (perpendicular offset)
 *
 * Styling and arrows behave like in `line()`.
 *
 * @param {d3.Selection} container
 * @param {{x:number,y:number}} start
 * @param {{x:number,y:number}} end
 * @param {Object} options
 * @returns {d3.Selection} path
 */
function curvedLine(container, start, end, options = {}) {
  if (typeof d3 === 'undefined') {
    throw new Error('d3 must be loaded before using curvedLine()');
  }

  const opts = Object.assign({
    stroke: '#000',
    strokeWidth: 2,
    opacity: 1,
    dasharray: null,
    dashoffset: 0,
    fill: 'none',
    arrowStart: false,
    arrowEnd: false,
    arrowSize: 6,
    class: '',
    // curvature controls (aliases supported)
    curveSide: 'left',
    bendSide: undefined,
    curveDistance: 40,
    bendDistance: undefined,
    offset: undefined
  }, options || {});

  const x1 = start.x, y1 = start.y;
  const x2 = end.x, y2 = end.y;

  // Decide side and distance using aliases
  const side = (opts.bendSide || opts.curveSide || 'left');
  const distance = Number(
    (opts.offset != null ? opts.offset : (opts.bendDistance != null ? opts.bendDistance : opts.curveDistance))
  );
  const bend = isFinite(distance) ? Math.max(0, distance) : 40;

  // Ensure defs for markers exists on the root SVG
  const rootSvg = container.node().ownerSVGElement || container;
  const defs = ensureDefs(d3.select(rootSvg));

  // Prepare marker ids unique to color/size/width
  const strokeColor = opts.stroke;
  const strokeWidth = +opts.strokeWidth;
  const arrowColor = opts.arrowColor || strokeColor;
  const arrowSize = +opts.arrowSize;
  const markerStartId = `arrow-start-${sanitizeId(arrowColor)}-${arrowSize}-${strokeWidth}`;
  const markerEndId = `arrow-end-${sanitizeId(arrowColor)}-${arrowSize}-${strokeWidth}`;

  if (opts.arrowStart) ensureArrowMarker(defs, markerStartId, arrowColor, arrowSize, strokeWidth, true);
  if (opts.arrowEnd) ensureArrowMarker(defs, markerEndId, arrowColor, arrowSize, strokeWidth, false);

  // Geometry
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;

  // Perpendicular unit vectors (screen coords: y increases downward)
  const leftNx = -uy, leftNy = ux;
  const rightNx = uy, rightNy = -ux;
  // Flip side mapping as requested: 'left' now produces the former 'right' bend and vice versa
  const nx = side === 'right' ? leftNx : rightNx;
  const ny = side === 'right' ? leftNy : rightNy;

  // Quadratic control point at offset midpoint
  const midx = (x1 + x2) / 2;
  const midy = (y1 + y2) / 2;
  const qcx = midx + nx * bend;
  const qcy = midy + ny * bend;

  // Shorten endpoints if arrows are present, along curve tangents
  let sx = x1, sy = y1, ex = x2, ey = y2;
  if (opts.arrowStart) {
    const t0x = qcx - x1;
    const t0y = qcy - y1;
    const t0len = Math.hypot(t0x, t0y) || 1;
    const shorten = Math.min(arrowSize, t0len - 1e-6);
    sx = x1 + (t0x / t0len) * shorten;
    sy = y1 + (t0y / t0len) * shorten;
  }
  if (opts.arrowEnd) {
    const t1x = x2 - qcx;
    const t1y = y2 - qcy;
    const t1len = Math.hypot(t1x, t1y) || 1;
    const shorten = Math.min(arrowSize, t1len - 1e-6);
    ex = x2 - (t1x / t1len) * shorten;
    ey = y2 - (t1y / t1len) * shorten;
  }

  // Use cubic Bezier equivalent of a quadratic (Q -> C conversion)
  // C1 = S + 2/3 * (Q - S); C2 = E + 2/3 * (Q - E)
  const c1x = sx + (2 / 3) * (qcx - sx);
  const c1y = sy + (2 / 3) * (qcy - sy);
  const c2x = ex + (2 / 3) * (qcx - ex);
  const c2y = ey + (2 / 3) * (qcy - ey);

  const d = `M ${sx} ${sy} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${ex} ${ey}`;

  const path = container.append('path')
    .attr('d', d)
    .attr('fill', opts.fill)
    .attr('stroke', opts.stroke)
    .attr('stroke-width', opts.strokeWidth)
    .attr('opacity', opts.opacity)
    .attr('class', opts.class);

  if (opts.dasharray != null) {
    const dashPattern = Array.isArray(opts.dasharray)
      ? opts.dasharray.join(' ')
      : String(opts.dasharray);
    path.attr('stroke-dasharray', dashPattern);
  }
  if (opts.dashoffset) {
    path.attr('stroke-dashoffset', opts.dashoffset);
  }

  if (opts.arrowStart) path.attr('marker-start', `url(#${markerStartId})`);
  if (opts.arrowEnd) path.attr('marker-end', `url(#${markerEndId})`);

  return path;
}

/**
 * Computes control points for rounded corners in bezier curves
 * 
 * This function calculates the entry/exit points and control points (c1, c2)
 * needed to create smooth rounded corners between line segments using cubic bezier curves.
 * 
 * @param {Object} prev - Previous point {x, y}
 * @param {Object} curr - Current corner point {x, y}
 * @param {Object} next - Next point {x, y}
 * @param {number} radius - Corner radius
 * @returns {Object} Object with entry, c1, c2, exit points for bezier curve
 */
function computeRoundedCorner(prev, curr, next, radius) {
  const vIn = { x: curr.x - prev.x, y: curr.y - prev.y };
  const vOut = { x: next.x - curr.x, y: next.y - curr.y };

  const lenIn = Math.hypot(vIn.x, vIn.y) || 1;
  const lenOut = Math.hypot(vOut.x, vOut.y) || 1;

  const uIn = { x: vIn.x / lenIn, y: vIn.y / lenIn };
  const uOut = { x: vOut.x / lenOut, y: vOut.y / lenOut };

  const rEff = Math.min(radius, lenIn / 2, lenOut / 2);

  const entry = { x: curr.x - uIn.x * rEff, y: curr.y - uIn.y * rEff };
  const exit = { x: curr.x + uOut.x * rEff, y: curr.y + uOut.y * rEff };

  const dot = Math.max(-1, Math.min(1, uIn.x * uOut.x + uIn.y * uOut.y));
  const phi = Math.acos(dot);
  const k = (4 / 3) * Math.tan(phi / 4) * rEff;

  const c1 = { x: entry.x + uIn.x * k, y: entry.y + uIn.y * k };
  const c2 = { x: exit.x - uOut.x * k, y: exit.y - uOut.y * k };

  return { entry, c1, c2, exit };
}

/**
 * Ensures SVG defs element exists for markers and gradients
 * @param {d3.Selection} svgSel - D3 selection of SVG element
 * @returns {d3.Selection} D3 selection of defs element
 */
function ensureDefs(svgSel) {
  let defs = svgSel.select('defs');
  if (defs.empty()) defs = svgSel.append('defs');
  return defs;
}

/**
 * Sanitizes string for use as SVG element ID
 * @param {string} s - String to sanitize
 * @returns {string} Sanitized string safe for SVG IDs
 */
function sanitizeId(s) {
  return String(s).replace(/[^a-zA-Z0-9_-]/g, '_');
}

/**
 * Creates or retrieves arrow marker definition for SVG paths
 * @param {d3.Selection} defs - D3 selection of defs element
 * @param {string} id - Unique marker ID
 * @param {string} color - Arrow color
 * @param {number} size - Arrow size in pixels
 * @param {number} strokeWidth - Line stroke width
 * @param {boolean} isStart - Whether this is a start arrow (true) or end arrow (false)
 * @returns {d3.Selection} D3 selection of marker element
 */
function ensureArrowMarker(defs, id, color, size, strokeWidth, isStart) {
  let marker = defs.select(`#${id}`);
  if (!marker.empty()) return marker;

  // Marker dimensions in user space units (pixels)
  const markerWidth = size;
  const markerHeight = size;

  marker = defs.append('marker')
    .attr('id', id)
    .attr('viewBox', `0 0 ${markerWidth} ${markerHeight}`)
    .attr('markerUnits', 'userSpaceOnUse')
    .attr('refX', isStart ? markerWidth : 0)
    .attr('refY', markerHeight / 2)
    .attr('markerWidth', markerWidth)
    .attr('markerHeight', markerHeight)
    .attr('orient', 'auto')

  // Draw a triangle
  const pathData = isStart
    // Tip at (0, markerHeight/2)
    ? `M ${markerWidth} 0 L 0 ${markerHeight / 2} L ${markerWidth} ${markerHeight} z`
    // Tip at (markerWidth, markerHeight/2)
    : `M 0 0 L ${markerWidth} ${markerHeight / 2} L 0 ${markerHeight} z`;

  marker.append('path')
    .attr('d', pathData)
    .attr('fill', 'context-stroke')
    // .attr('stroke', color)
    // .attr('stroke-width', Math.max(1, strokeWidth * 0.8));

  return marker;
}

// Expose globally
if (typeof window !== 'undefined') {
  window.SVGLines = { line, curvedLine };
  // Backward-compatible helper
  if (!window.line) window.line = line;
}


