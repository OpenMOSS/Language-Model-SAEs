function coordinates(width, height, startX = 0, startY = 0) {
  let x = d3.scaleLinear().domain([0, 1]).range([startX, width]);
  let y = d3.scaleLinear().domain([0, 1]).range([startY, height]);
  return { x, y };
}

function add(a, b) {
  return { x: (a.x || 0) + (b.x || 0), y: (a.y || 0) + (b.y || 0) };
}

// --- Shape anchor utilities ---
// Accepts either:
// - An SVG element (e.g., <rect>, <circle>, <g> child). Computes anchors in SVG user coordinates
//   by transforming the element's local bbox points to the root SVG space.
// - A plain object describing a rect: { type: 'rect', x, y, width, height } in absolute SVG coords
// - A plain object describing a circle: { type: 'circle', cx, cy, r } in absolute SVG coords

function unwrapTarget(obj) {
  // Unwrap d3 selection -> DOM node
  if (obj && typeof obj.node === 'function') {
    const n = obj.node();
    if (n) return n;
  }
  return obj;
}

function isSvgElement(obj) {
  const el = unwrapTarget(obj);
  return !!(el && (
    el.ownerSVGElement ||
    (el.tagName && String(el.tagName).toLowerCase() === 'svg') ||
    (el.nodeType === 1 && el.namespaceURI && /svg/i.test(el.namespaceURI))
  ));
}

function isNumberLike(v) { return v !== null && v !== undefined && !Number.isNaN(+v); }

function isRectObject(obj) {
  return obj && (obj.type === 'rect' || (isNumberLike(obj.x) && isNumberLike(obj.y) && isNumberLike(obj.width) && isNumberLike(obj.height)));
}

function isCircleObject(obj) {
  return obj && (obj.type === 'circle' || (isNumberLike(obj.cx) && isNumberLike(obj.cy) && isNumberLike(obj.r)));
}

function svgRoot(el) {
  return el.ownerSVGElement || (el.tagName && el.tagName.toLowerCase() === 'svg' ? el : null);
}

function toGlobalPoint(el, x, y) {
  const root = svgRoot(el) || el;
  const pt = (root.createSVGPoint ? root.createSVGPoint() : null);
  if (!pt || !el.getScreenCTM || !root.getScreenCTM) {
    // Fallback: assume already in absolute coordinates
    return { x, y };
  }
  pt.x = x;
  pt.y = y;
  const screenMat = el.getScreenCTM();
  const rootMat = root.getScreenCTM();
  if (!screenMat || !rootMat || !rootMat.inverse) return { x, y };
  // Convert local point -> screen -> root svg user space
  const screenPt = pt.matrixTransform(screenMat);
  const svgPt = screenPt.matrixTransform(rootMat.inverse());
  return { x: svgPt.x, y: svgPt.y };
}

function bboxFor(elOrObj) {
  const t = unwrapTarget(elOrObj);
  if (isSvgElement(t)) {
    // Use element's bbox in its local coords
    const bb = (typeof t.getBBox === 'function') ? t.getBBox() : null;
    if (bb) {
      return { type: 'rect', x: bb.x, y: bb.y, width: bb.width, height: bb.height, _el: t };
    }
  } else if (isRectObject(t)) {
    return { type: 'rect', x: +t.x, y: +t.y, width: +t.width, height: +t.height, _el: null };
  } else if (isCircleObject(t)) {
    return { type: 'circle', cx: +t.cx, cy: +t.cy, r: +t.r, _el: null };
  }
  // Unknown, return zero rect
  throw new Error(`Unknown target type: ${Object.prototype.toString.call(t)}`);
}

function anchorsForRect(rect) {
  const { x, y, width, height, _el } = rect;
  const points = {
    topLeft: { x: x, y: y },
    top: { x: x + width / 2, y: y },
    topRight: { x: x + width, y: y },
    left: { x: x, y: y + height / 2 },
    center: { x: x + width / 2, y: y + height / 2 },
    right: { x: x + width, y: y + height / 2 },
    bottomLeft: { x: x, y: y + height },
    bottom: { x: x + width / 2, y: y + height },
    bottomRight: { x: x + width, y: y + height }
  };
  if (_el) {
    // Transform all points to global SVG space
    const out = {};
    for (const k in points) out[k] = toGlobalPoint(_el, points[k].x, points[k].y);
    return out;
  }
  return points;
}

function anchorsForCircle(circle) {
  const { cx, cy, r, _el } = circle;
  const points = {
    top: { x: cx, y: cy - r },
    bottom: { x: cx, y: cy + r },
    left: { x: cx - r, y: cy },
    right: { x: cx + r, y: cy },
    center: { x: cx, y: cy },
    topLeft: { x: cx - r * Math.SQRT1_2, y: cy - r * Math.SQRT1_2 },
    topRight: { x: cx + r * Math.SQRT1_2, y: cy - r * Math.SQRT1_2 },
    bottomLeft: { x: cx - r * Math.SQRT1_2, y: cy + r * Math.SQRT1_2 },
    bottomRight: { x: cx + r * Math.SQRT1_2, y: cy + r * Math.SQRT1_2 }
  };
  if (_el) {
    const out = {};
    for (const k in points) out[k] = toGlobalPoint(_el, points[k].x, points[k].y);
    return out;
  }
  return points;
}

function computeAnchors(target) {
  const bb = bboxFor(target);
  if (bb.type === 'rect') return anchorsForRect(bb);
  if (bb.type === 'circle') return anchorsForCircle(bb);
  throw new Error('Unknown target type');
}

function getTop(target) { return computeAnchors(target).top; }
function getBottom(target) { return computeAnchors(target).bottom; }
function getLeft(target) { return computeAnchors(target).left; }
function getRight(target) { return computeAnchors(target).right; }
function getCenter(target) { return computeAnchors(target).center; }
function getTopLeft(target) { return computeAnchors(target).topLeft; }
function getTopRight(target) { return computeAnchors(target).topRight; }
function getBottomLeft(target) { return computeAnchors(target).bottomLeft; }
function getBottomRight(target) { return computeAnchors(target).bottomRight; }

// Expose globally
if (typeof window !== 'undefined') {
  window.SVGCoordinates = Object.assign(window.SVGCoordinates || {}, {
    coordinates,
    getTop,
    getBottom,
    getLeft,
    getRight,
    getCenter,
    getTopLeft,
    getTopRight,
    getBottomLeft,
    getBottomRight
  });
  // Backward-compatible helpers
  if (!window.getTop) window.getTop = getTop;
  if (!window.getBottom) window.getBottom = getBottom;
  if (!window.getLeft) window.getLeft = getLeft;
  if (!window.getRight) window.getRight = getRight;
  if (!window.getCenter) window.getCenter = getCenter;
  if (!window.getTopLeft) window.getTopLeft = getTopLeft;
  if (!window.getTopRight) window.getTopRight = getTopRight;
  if (!window.getBottomLeft) window.getBottomLeft = getBottomLeft;
  if (!window.getBottomRight) window.getBottomRight = getBottomRight;
}

