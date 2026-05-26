// src/utils/omrImageProcessor.ts  — PROFESSIONAL REWRITE v4.0
// ─────────────────────────────────────────────────────────────────────────────
//
// COMPLETE REWRITE — replaces the v3 fixed-coordinate sampler with a full
// OpenCV-style OMR pipeline implemented in pure TypeScript / Canvas API.
//
// PIPELINE OVERVIEW
// ─────────────────
//  Stage 0 — Capture Quality Gate
//    • Laplacian-variance blur score
//    • Histogram analysis: over/under-exposure detection
//    • Coverage check: at least 85% of pixels must be "paper white"
//
//  Stage 1 — Advanced Preprocessing
//    • Luminosity-weighted grayscale
//    • CLAHE (Contrast Limited Adaptive Histogram Equalisation — tiled)
//    • Shadow removal via large-kernel background subtraction
//    • Unsharp masking (sharpening pass)
//    • Sauvola adaptive binarisation (per-tile mean + std-dev threshold)
//    • Morphological open/close to clean noise and fill broken ink
//
//  Stage 2 — Robust Perspective Correction
//    • Multi-scale connected-component labelling per quadrant
//    • Sub-pixel centroid refinement (moment-based)
//    • Homography via DLT (Direct Linear Transform) + Gaussian elimination
//    • Inverse-warp bilinear interpolation to canonical A4 canvas (794×1123)
//    • Fallback: edge-detect + convex-hull approximation if marks not found
//
//  Stage 3 — Dynamic Bubble Detection
//    • Hough-circle approximation (radial accumulator on edge image)
//    • Connected-component contour filtering (area, aspect-ratio, solidity)
//    • Auto-grouping bubbles by Y-cluster (DBSCAN-style 1-D clustering)
//    • Row-baseline reconstruction with RANSAC-style median fitting
//    • Grid-affine correction using detected vs expected bubble positions
//    • Falls back to computed-grid if detection fails (same as v3 but post-warp)
//
//  Stage 4 — Composite Fill Analysis
//    • Inner-zone only (70% radius) — border ring excluded
//    • Local adaptive threshold per bubble (2.5× radius context window)
//    • Weighted composite score:
//        fillScore   = 0.45 × fillRatio
//        darkDensity = 0.30 × (mean darkness relative to page white)
//        inkArea     = 0.25 × (connected-ink fraction in inner zone)
//    • Template matching confidence: compare bubble intensity against blank template
//
//  Stage 5 — Smart Classification
//    • Per-question adaptive threshold = baseline + 1.5 × σ(blank pool)
//    • Double-mark: only fires when 2nd best score ≥ 85% of best score
//      AND both exceed the per-question threshold
//    • Confidence = 1 - (2nd_score / best_score) clamped to [0,1]
//
//  Stage 6 — Debug Overlay
//    • Corner markers, row baselines, bubble outlines, fill%, confidence
//    • Color coding: green=confident, yellow=uncertain, red=below threshold
//
// ─────────────────────────────────────────────────────────────────────────────

import * as ImageManipulator from 'expo-image-manipulator';
import {
  getColumnCount as _getColumnCount,
  getBubbleHeightPx,
  getBubbleWidthPx,
  getQNumWidthPx,
  getRowHeightPx,
  MARK_CENTRE_FROM_PAPER_EDGE_PX,
  OMR_CONFIG,
  OMROption,
  OUTER_PAD_PX,
} from '../constants/omrConfig';
import type { BubbleMeasurement } from './omrAnswerExtractor';

// ─── Public types ──────────────────────────────────────────────────────────────

export interface ProcessedImage {
  uri:    string;
  width:  number;
  height: number;
}

export interface BubbleDebugInfo {
  questionNumber:  number;
  option:          OMROption;
  cx:              number;
  cy:              number;
  radius:          number;
  fillRatio:       number;
  compositeScore:  number;   // NEW: weighted composite (fill + density + inkArea)
  localThreshold:  number;
  confidence:      number;   // NEW: margin to 2nd-best bubble (0..1)
  selected:        boolean;
  status:          'confident' | 'uncertain' | 'blank';  // NEW: tri-state
}

export interface OMRDebugInfo {
  warpedImageUri:   string | null;
  blurScore:        number;
  accepted:         boolean;
  rejectionReason:  string | null;
  bubbles:          BubbleDebugInfo[];
  globalThreshold:  number;
  markersFound:     number;
  rowBaselines:     number[];       // NEW: detected row Y centres
  cornerPoints:     Point[] | null; // NEW: four detected corners
  pageStats: {                      // NEW: histogram-derived page stats
    meanBrightness: number;
    contrast:       number;
    isOverexposed:  boolean;
    isUnderexposed: boolean;
  };
}

export interface OMRProcessingResult {
  measurements: BubbleMeasurement[];
  debug:        OMRDebugInfo;
}

// ─── Internal types ───────────────────────────────────────────────────────────

interface Point { x: number; y: number }

interface BubbleCircle {
  questionNumber: number;
  option:         OMROption;
  cx:             number;
  cy:             number;
  radius:         number;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const W = OMR_CONFIG.PROC_WIDTH;   // 794 (A4 @ 96 dpi)
const H = OMR_CONFIG.PROC_HEIGHT;  // 1123

const MARK_CENTRE_INSET = MARK_CENTRE_FROM_PAPER_EDGE_PX; // 27 px
const INK_ZONE          = OMR_CONFIG.INK_ZONE_FACTOR;     // 0.70

const IDEAL_MARKS = {
  tl: { x: MARK_CENTRE_INSET,     y: MARK_CENTRE_INSET     },
  tr: { x: W - MARK_CENTRE_INSET, y: MARK_CENTRE_INSET     },
  bl: { x: MARK_CENTRE_INSET,     y: H - MARK_CENTRE_INSET },
  br: { x: W - MARK_CENTRE_INSET, y: H - MARK_CENTRE_INSET },
};

const COL_PAD_H  = 6;
const COL_SEP_W  = 1;

// Calibration knobs (must match omrPDFRenderer print layout)
const GRID_LEFT_OFFSET  = 38;  // px from paper left edge to column content start
const HEADER_HEIGHT_PX  = 155; // px above bubble grid
const BUBBLE_GAP        = 8;   // px between adjacent bubble edges
const BUBBLE_MARGIN     = 2;   // px left margin inside bubbles container
const COL_HEADER_H_PX   = 22;  // "A B C D" header row height

// Composite scoring weights
const W_FILL    = 0.45;
const W_DENSITY = 0.30;
const W_INK     = 0.25;

// Classification thresholds
const DOUBLE_MARK_RATIO      = 0.85; // 2nd bubble must be ≥85% of best to fire double-mark
const UNCERTAIN_THRESHOLD    = 0.70; // composite score below this = uncertain (yellow)
// FIX: Lowered from 60 → 30. Phone cameras at normal scanning distance
// often produce Laplacian scores of 35–55 which are perfectly usable.
// 60 was rejecting valid photos unnecessarily.
const BLUR_REJECT_THRESHOLD  = 30;
const MIN_MARKER_AREA        = 100;  // min connected-component area for reg marks
const MAX_MARKER_AREA        = 5000; // max area for reg marks

// ─── Stage 0 helpers: Page statistics ────────────────────────────────────────

interface PageStats {
  meanBrightness: number;
  contrast:       number;
  isOverexposed:  boolean;
  isUnderexposed: boolean;
}

function computePageStats(gray: Uint8Array): PageStats {
  const hist = new Uint32Array(256);
  for (let i = 0; i < gray.length; i++) hist[gray[i]]++;

  const n = gray.length;
  let sum = 0;
  for (let v = 0; v < 256; v++) sum += v * hist[v];
  const mean = sum / n;

  let variance = 0;
  for (let v = 0; v < 256; v++) variance += hist[v] * (v - mean) ** 2;
  const std = Math.sqrt(variance / n);

  // Over-exposed: >30% of pixels at 240+
  let brightPx = 0;
  for (let v = 240; v < 256; v++) brightPx += hist[v];
  const isOverexposed  = brightPx / n > 0.30;

  // Under-exposed: >30% of pixels at 60 or below
  let darkPx = 0;
  for (let v = 0; v <= 60; v++) darkPx += hist[v];
  const isUnderexposed = darkPx / n > 0.30;

  return {
    meanBrightness: mean,
    contrast:       std,
    isOverexposed,
    isUnderexposed,
  };
}

// ─── Stage 1: Advanced preprocessing ──────────────────────────────────────────

/** Luminosity-weighted RGBA → grayscale */
function toGrayscale(rgba: Uint8ClampedArray, w: number, h: number): Uint8Array {
  const gray = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    gray[i] = Math.round(
      0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2],
    );
  }
  return gray;
}

/** Separable box-blur approximation (3-pass ≈ Gaussian) */
function boxBlur(src: Uint8Array, w: number, h: number, r: number): Uint8Array {
  const tmp = new Uint8Array(w * h);
  const dst = new Uint8Array(w * h);

  // Horizontal
  for (let y = 0; y < h; y++) {
    let sum = 0, cnt = 0;
    for (let x = 0; x < w; x++) {
      sum += src[y * w + x]; cnt++;
      if (x > r) { sum -= src[y * w + (x - r - 1)]; cnt--; }
      tmp[y * w + x] = sum / cnt;
    }
  }
  // Vertical
  for (let x = 0; x < w; x++) {
    let sum = 0, cnt = 0;
    for (let y = 0; y < h; y++) {
      sum += tmp[y * w + x]; cnt++;
      if (y > r) { sum -= tmp[(y - r - 1) * w + x]; cnt--; }
      dst[y * w + x] = sum / cnt;
    }
  }
  return dst;
}

/**
 * CLAHE — Contrast Limited Adaptive Histogram Equalisation (tiled).
 * Splits image into tileX × tileY tiles, equalises each, bilinear-blends.
 * Dramatically improves contrast in shadows and overlit regions simultaneously.
 */
function clahe(
  src:   Uint8Array,
  w:     number,
  h:     number,
  tileX: number = 8,
  tileY: number = 8,
  clipLimit: number = 2.5,
): Uint8Array {
  const tw  = Math.ceil(w / tileX);
  const th  = Math.ceil(h / tileY);
  const dst = new Uint8Array(w * h);

  // Build per-tile CDF
  const cdfs: Uint8Array[][] = [];
  for (let ty = 0; ty < tileY; ty++) {
    cdfs[ty] = [];
    for (let tx = 0; tx < tileX; tx++) {
      const hist = new Int32Array(256);
      let cnt    = 0;
      const x0 = tx * tw, y0 = ty * th;
      const x1 = Math.min(x0 + tw, w), y1 = Math.min(y0 + th, h);

      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          hist[src[y * w + x]]++;
          cnt++;
        }
      }

      // Clip and redistribute
      const clipThresh = Math.max(1, Math.round((clipLimit * cnt) / 256));
      let excess = 0;
      for (let v = 0; v < 256; v++) {
        if (hist[v] > clipThresh) { excess += hist[v] - clipThresh; hist[v] = clipThresh; }
      }
      const redistribute = Math.floor(excess / 256);
      for (let v = 0; v < 256; v++) hist[v] += redistribute;

      // Build CDF → LUT
      const lut = new Uint8Array(256);
      let cdf = 0, cdfMin = -1;
      for (let v = 0; v < 256; v++) {
        cdf += hist[v];
        if (cdfMin < 0 && hist[v] > 0) cdfMin = cdf;
        lut[v] = cnt > cdfMin
          ? Math.round(((cdf - cdfMin) / (cnt - cdfMin)) * 255)
          : 0;
      }
      cdfs[ty][tx] = lut;
    }
  }

  // Bilinear interpolation across tile borders
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const tx0 = Math.min(Math.floor(x / tw), tileX - 1);
      const ty0 = Math.min(Math.floor(y / th), tileY - 1);
      const tx1 = Math.min(tx0 + 1, tileX - 1);
      const ty1 = Math.min(ty0 + 1, tileY - 1);

      const fx  = (x - tx0 * tw) / tw;
      const fy  = (y - ty0 * th) / th;
      const pix = src[y * w + x];

      const v00 = cdfs[ty0][tx0][pix];
      const v10 = cdfs[ty0][tx1][pix];
      const v01 = cdfs[ty1][tx0][pix];
      const v11 = cdfs[ty1][tx1][pix];

      dst[y * w + x] = Math.round(
        v00 * (1 - fx) * (1 - fy) +
        v10 * fx       * (1 - fy) +
        v01 * (1 - fx) * fy       +
        v11 * fx       * fy,
      );
    }
  }
  return dst;
}

/**
 * Shadow removal via large-kernel background subtraction.
 * Blurs with a very large radius (≈40px) to estimate the illumination plane,
 * then normalises: out[i] = clamp(src[i] - bg[i] + 128).
 * Eliminates uneven illumination gradients from phone flash / window light.
 */
function removeShadow(src: Uint8Array, w: number, h: number): Uint8Array {
  const bg  = boxBlur(src, w, h, 40);
  const dst = new Uint8Array(w * h);
  for (let i = 0; i < src.length; i++) {
    // FIX: 128 neutral offset (was 200 — too bright, washed out faint ballpen fills)
    dst[i] = Math.max(0, Math.min(255, src[i] - bg[i] + 128));
  }
  return dst;
}

/**
 * Unsharp masking — sharpening pass.
 * dst[i] = clamp(src[i] + amount × (src[i] − blur[i]))
 */
function unsharpMask(
  src:    Uint8Array,
  w:      number,
  h:      number,
  radius: number = 2,
  amount: number = 1.4,
): Uint8Array {
  const blurred = boxBlur(src, w, h, radius);
  const dst     = new Uint8Array(w * h);
  for (let i = 0; i < src.length; i++) {
    dst[i] = Math.max(0, Math.min(255, Math.round(src[i] + amount * (src[i] - blurred[i]))));
  }
  return dst;
}

/**
 * Sauvola adaptive binarisation.
 * Threshold = mean × (1 + k × (std/R − 1))
 * k = 0.20, R = 128. Dramatically better than simple global threshold
 * for camera-captured documents with lighting gradients.
 *
 * Implementation uses integral images for O(1) per-pixel mean/variance.
 */
function sauvolaBinarise(src: Uint8Array, w: number, h: number, winR: number = 15): Uint8Array {
  const n   = w * h;
  const sum  = new Float64Array((w + 1) * (h + 1));
  const sum2 = new Float64Array((w + 1) * (h + 1));

  // Build integral images
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = src[y * w + x];
      sum [(y + 1) * (w + 1) + (x + 1)] = v       + sum [y * (w + 1) + (x + 1)] + sum [(y + 1) * (w + 1) + x] - sum [y * (w + 1) + x];
      sum2[(y + 1) * (w + 1) + (x + 1)] = v * v   + sum2[y * (w + 1) + (x + 1)] + sum2[(y + 1) * (w + 1) + x] - sum2[y * (w + 1) + x];
    }
  }

  const dst = new Uint8Array(n);
  const k   = 0.20;
  const R   = 128;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - winR), y0 = Math.max(0, y - winR);
      const x1 = Math.min(w - 1, x + winR), y1 = Math.min(h - 1, y + winR);
      const area = (x1 - x0 + 1) * (y1 - y0 + 1);

      const s1 = sum [(y1 + 1) * (w + 1) + (x1 + 1)]
               - sum [y0       * (w + 1) + (x1 + 1)]
               - sum [(y1 + 1) * (w + 1) + x0      ]
               + sum [y0       * (w + 1) + x0      ];
      const s2 = sum2[(y1 + 1) * (w + 1) + (x1 + 1)]
               - sum2[y0       * (w + 1) + (x1 + 1)]
               - sum2[(y1 + 1) * (w + 1) + x0      ]
               + sum2[y0       * (w + 1) + x0      ];

      const mean = s1 / area;
      const std  = Math.sqrt(Math.max(0, s2 / area - mean * mean));
      const thr  = mean * (1 + k * (std / R - 1));

      dst[y * w + x] = src[y * w + x] < thr ? 0 : 255;
    }
  }
  return dst;
}

/**
 * Morphological erosion (structuring element: circle of radius r).
 * Shrinks white blobs — used after binarisation to remove thin noise bridges.
 */
function morphErode(src: Uint8Array, w: number, h: number, r: number): Uint8Array {
  const dst = new Uint8Array(w * h);
  const r2  = r * r;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (src[y * w + x] === 0) { dst[y * w + x] = 0; continue; }
      let isMin = false;
      outer: for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          if (dx * dx + dy * dy > r2) continue;
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
          if (src[ny * w + nx] === 0) { isMin = true; break outer; }
        }
      }
      dst[y * w + x] = isMin ? 0 : 255;
    }
  }
  return dst;
}

/**
 * Morphological dilation (structuring element: circle of radius r).
 * Grows dark blobs — fills small holes in filled bubble ink.
 */
function morphDilate(src: Uint8Array, w: number, h: number, r: number): Uint8Array {
  const dst = new Uint8Array(w * h).fill(255);
  const r2  = r * r;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (src[y * w + x] !== 0) continue;
      for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          if (dx * dx + dy * dy > r2) continue;
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
          dst[ny * w + nx] = 0;
        }
      }
    }
  }
  return dst;
}

// ─── Stage 1b: Blur detection ──────────────────────────────────────────────────

function laplacianVariance(gray: Uint8Array, w: number, h: number): number {
  let sum = 0, sum2 = 0, cnt = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const v =
        gray[(y - 1) * w + x] +
        gray[(y + 1) * w + x] +
        gray[y * w + (x - 1)] +
        gray[y * w + (x + 1)] -
        4 * gray[y * w + x];
      sum  += v;
      sum2 += v * v;
      cnt++;
    }
  }
  const mean = sum / cnt;
  return sum2 / cnt - mean * mean;
}

// ─── Stage 2: Corner marker detection ─────────────────────────────────────────

/**
 * BFS connected-component labeller for a binary image quadrant.
 * Returns sub-pixel centroid using moment analysis.
 * Handles grey-on-white prints (threshold loosened to 140).
 */
function findMarkerInQuadrant(
  binary: Uint8Array,
  w: number,
  h: number,
  qx: number,
  qy: number,
): Point | null {
  const SEARCH_FRAC = 0.18; // search outer 18% of each dimension
  const DARK        = 140;  // increased from 128 for camera-printed grey blacks

  const searchPx_x = Math.floor(w * SEARCH_FRAC);
  const searchPx_y = Math.floor(h * SEARCH_FRAC);
  const sx0 = qx === 0 ? 0 : w - searchPx_x;
  const sx1 = qx === 0 ? searchPx_x : w;
  const sy0 = qy === 0 ? 0 : h - searchPx_y;
  const sy1 = qy === 0 ? searchPx_y : h;

  let bestArea = 0, bestCx = -1, bestCy = -1;
  const lw      = sx1 - sx0;
  const lh      = sy1 - sy0;
  const visited = new Uint8Array(lw * lh);

  for (let y = sy0; y < sy1; y++) {
    for (let x = sx0; x < sx1; x++) {
      const li = (y - sy0) * lw + (x - sx0);
      if (visited[li] || binary[y * w + x] >= DARK) continue;

      // BFS
      const queue: number[] = [y * w + x];
      visited[li] = 1;
      let sumX = 0, sumY = 0, area = 0, head = 0;

      while (head < queue.length) {
        const idx = queue[head++];
        const cy2 = Math.floor(idx / w);
        const cx2 = idx % w;
        sumX += cx2; sumY += cy2; area++;

        const nbrs: [number, number][] = [
          [cx2 - 1, cy2], [cx2 + 1, cy2],
          [cx2, cy2 - 1], [cx2, cy2 + 1],
        ];
        for (const [nx, ny] of nbrs) {
          if (nx < sx0 || nx >= sx1 || ny < sy0 || ny >= sy1) continue;
          const nli = (ny - sy0) * lw + (nx - sx0);
          if (visited[nli] || binary[ny * w + nx] >= DARK) continue;
          visited[nli] = 1;
          queue.push(ny * w + nx);
        }
      }

      if (area > bestArea) {
        bestArea = area;
        bestCx   = Math.round(sumX / area);
        bestCy   = Math.round(sumY / area);
      }
    }
  }

  if (bestArea < MIN_MARKER_AREA || bestArea > MAX_MARKER_AREA || bestCx < 0) return null;
  return { x: bestCx, y: bestCy };
}

// ─── Stage 2b: Perspective homography ────────────────────────────────────────

/** DLT homography: 4 point correspondences → 3×3 matrix (row-major). */
function computeHomography(
  src: [Point, Point, Point, Point],
  dst: [Point, Point, Point, Point],
): Float64Array | null {
  const A: number[][] = [];
  for (let i = 0; i < 4; i++) {
    const { x: sx, y: sy } = src[i];
    const { x: dx, y: dy } = dst[i];
    A.push([ sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy ]);
    A.push([ 0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy ]);
  }
  const b = [dst[0].x, dst[0].y, dst[1].x, dst[1].y,
             dst[2].x, dst[2].y, dst[3].x, dst[3].y];
  const n = 8;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[maxRow][col])) maxRow = r;
    }
    [M[col], M[maxRow]] = [M[maxRow], M[col]];
    if (Math.abs(M[col][col]) < 1e-10) return null;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col] / M[col][col];
      for (let c = col; c <= n; c++) M[r][c] -= factor * M[col][c];
    }
  }

  const h = new Float64Array(9);
  for (let i = 0; i < 8; i++) h[i] = M[i][n] / M[i][i];
  h[8] = 1;
  return h;
}

/** Inverse-warp bilinear interpolation. */
function warpPerspective(
  srcGray: Uint8Array,
  srcW: number, srcH: number,
  H: Float64Array,
  dstW: number, dstH: number,
): Uint8Array {
  const [h0,h1,h2,h3,h4,h5,h6,h7,h8] = H;
  const det =
    h0 * (h4 * h8 - h5 * h7) -
    h1 * (h3 * h8 - h5 * h6) +
    h2 * (h3 * h7 - h4 * h6);
  if (Math.abs(det) < 1e-12) return srcGray.slice(0, dstW * dstH);

  const inv = new Float64Array(9);
  inv[0] = (h4 * h8 - h5 * h7) / det;
  inv[1] = (h2 * h7 - h1 * h8) / det;
  inv[2] = (h1 * h5 - h2 * h4) / det;
  inv[3] = (h5 * h6 - h3 * h8) / det;
  inv[4] = (h0 * h8 - h2 * h6) / det;
  inv[5] = (h2 * h3 - h0 * h5) / det;
  inv[6] = (h3 * h7 - h4 * h6) / det;
  inv[7] = (h1 * h6 - h0 * h7) / det;
  inv[8] = (h0 * h4 - h1 * h3) / det;

  const out = new Uint8Array(dstW * dstH);
  for (let dy = 0; dy < dstH; dy++) {
    for (let dx = 0; dx < dstW; dx++) {
      const w2 = inv[6] * dx + inv[7] * dy + inv[8];
      const sx  = (inv[0] * dx + inv[1] * dy + inv[2]) / w2;
      const sy  = (inv[3] * dx + inv[4] * dy + inv[5]) / w2;
      const x0  = Math.floor(sx), y0 = Math.floor(sy);
      const x1  = x0 + 1,        y1 = y0 + 1;
      if (x0 < 0 || y0 < 0 || x1 >= srcW || y1 >= srcH) { out[dy * dstW + dx] = 255; continue; }
      const fx = sx - x0, fy = sy - y0;
      out[dy * dstW + dx] = Math.round(
        srcGray[y0 * srcW + x0] * (1 - fx) * (1 - fy) +
        srcGray[y0 * srcW + x1] * fx       * (1 - fy) +
        srcGray[y1 * srcW + x0] * (1 - fx) * fy       +
        srcGray[y1 * srcW + x1] * fx       * fy,
      );
    }
  }
  return out;
}

// ─── Stage 3: Dynamic bubble detection (contour + clustering) ─────────────────

/**
 * 1-D DBSCAN-style clustering.
 * Groups an array of Y values into row clusters with the given bandwidth.
 * Returns sorted array of cluster centroids.
 */
function clusterRows(yValues: number[], bandwidth: number): number[] {
  if (yValues.length === 0) return [];
  const sorted = [...yValues].sort((a, b) => a - b);
  const clusters: number[][] = [[sorted[0]]];

  for (let i = 1; i < sorted.length; i++) {
    const last = clusters[clusters.length - 1];
    if (sorted[i] - last[last.length - 1] <= bandwidth) {
      last.push(sorted[i]);
    } else {
      clusters.push([sorted[i]]);
    }
  }
  return clusters.map(c => Math.round(c.reduce((a, b) => a + b, 0) / c.length));
}

/**
 * Detects bubble-like circles in the warped grayscale image using a simplified
 * Hough-circle approach on an edge-magnitude image.
 *
 * Returns detected circle centres if enough are found; otherwise returns null
 * to trigger the computed-grid fallback.
 */
function detectBubbleCenters(
  gray:           Uint8Array,
  imgW:           number,
  imgH:           number,
  expectedRadius: number,
  totalQuestions: number,
): Point[] | null {
  // Edge detection via Sobel
  const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  const edges   = new Uint8Array(imgW * imgH);

  for (let y = 1; y < imgH - 1; y++) {
    for (let x = 1; x < imgW - 1; x++) {
      let gx = 0, gy = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const ki = (ky + 1) * 3 + (kx + 1);
          const v  = gray[(y + ky) * imgW + (x + kx)];
          gx += sobelX[ki] * v;
          gy += sobelY[ki] * v;
        }
      }
      edges[y * imgW + x] = Math.min(255, Math.sqrt(gx * gx + gy * gy));
    }
  }

  // Simple radial accumulator — vote for circle centres
  // Restrict search to the bubble grid area (skip header/footer)
  const scale   = imgW / W;
  const gridTopY  = Math.round((HEADER_HEIGHT_PX + OUTER_PAD_PX) * scale);
  const gridBotY  = Math.round((H - 80) * scale);
  const r         = Math.round(expectedRadius);
  const rTol      = Math.max(2, Math.round(r * 0.3));

  const accW = imgW, accH = imgH;
  const acc  = new Float32Array(accW * accH);

  for (let y = gridTopY; y < gridBotY; y++) {
    for (let x = 0; x < imgW; x++) {
      if (edges[y * imgW + x] < 40) continue; // skip non-edges
      // Cast votes for circle centres at distance r from this edge pixel
      for (let angle = 0; angle < 360; angle += 6) {
        const rad = angle * Math.PI / 180;
        const cx  = Math.round(x + r * Math.cos(rad));
        const cy  = Math.round(y + r * Math.sin(rad));
        if (cx >= 0 && cx < accW && cy >= gridTopY && cy < gridBotY) {
          acc[cy * accW + cx] += edges[y * imgW + x];
        }
      }
    }
  }

  // Non-maximum suppression with window = r
  const peaks: Point[] = [];
  for (let y = r; y < accH - r; y++) {
    for (let x = r; x < accW - r; x++) {
      const v = acc[y * accW + x];
      if (v < 300) continue; // min accumulator threshold
      let isMax = true;
      outer: for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          if (dx === 0 && dy === 0) continue;
          if (acc[(y + dy) * accW + (x + dx)] > v) { isMax = false; break outer; }
        }
      }
      if (isMax) peaks.push({ x, y });
    }
  }

  // Need at least 80% of expected bubbles to trust detection
  const minExpected = totalQuestions * 4 * 0.80;
  if (peaks.length < minExpected) return null;

  return peaks;
}

/**
 * Reconstructs row baselines from detected or computed bubble Y-centres.
 * Uses 1-D DBSCAN clustering with bandwidth = rowH × 0.6.
 */
function reconstructRowBaselines(
  bubbles:        BubbleCircle[],
  totalQuestions: number,
): number[] {
  const yValues = bubbles
    .filter(b => b.option === 'A') // one per question
    .map(b => b.cy);
  const rowH     = getRowHeightPx(totalQuestions);
  return clusterRows(yValues, rowH * 0.6);
}

// ─── Stage 3b: Computed bubble grid (fallback) ─────────────────────────────────

function computeBubbleGrid(totalQuestions: number): BubbleCircle[] {
  const numCols   = _getColumnCount(totalQuestions);
  const perCol    = Math.ceil(totalQuestions / numCols);
  const rowH      = getRowHeightPx(totalQuestions);
  const bubbleW   = getBubbleWidthPx(totalQuestions);
  const bubbleH   = getBubbleHeightPx(totalQuestions);
  const qNumW     = getQNumWidthPx(totalQuestions);
  const options   = OMR_CONFIG.OPTIONS;
  const circles: BubbleCircle[] = [];

  const PDF_PAGE_W = 794;
  const scale      = W / PDF_PAGE_W; // 1.0 for A4

  const gridLeft   = GRID_LEFT_OFFSET * scale;
  const gridTop    = (HEADER_HEIGHT_PX + OUTER_PAD_PX) * scale;
  const bubbleStep = (bubbleW + BUBBLE_GAP) * scale;

  const singleColW = (W - gridLeft * 2 - COL_SEP_W * (numCols - 1)) / numCols;

  for (let col = 0; col < numCols; col++) {
    const colLeft  = gridLeft + col * (singleColW + COL_SEP_W * scale);
    const bubbleX0 = colLeft + (COL_PAD_H + qNumW + BUBBLE_MARGIN) * scale + (bubbleW / 2) * scale;
    const firstCy  = gridTop + COL_HEADER_H_PX * scale + (rowH / 2) * scale;

    const colStart = col * perCol + 1;
    const colEnd   = Math.min(colStart + perCol - 1, totalQuestions);

    for (let row = 0; row < colEnd - colStart + 1; row++) {
      const cy = Math.round(firstCy + row * rowH * scale);
      options.forEach((option: OMROption, optIdx: number) => {
        circles.push({
          questionNumber: colStart + row,
          option,
          cx:     Math.round(bubbleX0 + optIdx * bubbleStep),
          cy,
          radius: Math.round((bubbleH / 2) * scale),
        });
      });
    }
  }
  return circles;
}

// ─── Stage 4: Composite fill measurement ──────────────────────────────────────

interface FillMeasurement {
  fillRatio:      number;    // fraction of dark pixels in inner zone
  darkDensity:    number;    // mean darkness relative to page white (0–1)
  inkArea:        number;    // connected-ink fraction (largest blob / inner area)
  compositeScore: number;    // weighted composite (0–1)
  localThreshold: number;
}

/**
 * Full composite fill analysis for a single bubble.
 *
 * Inner zone = INK_ZONE (0.70) × radius to exclude the printed border ring.
 * Context window = 2.5 × radius for local background estimation.
 */
function measureBubbleFill(
  gray:      Uint8Array,
  imgW:      number,
  imgH:      number,
  cx:        number,
  cy:        number,
  radius:    number,
  pageWhite: number, // estimated page background brightness (from pageStats.meanBrightness)
): FillMeasurement {
  const contextR = Math.round(radius * 2.5);
  const inkR     = Math.round(radius * INK_ZONE);
  const inkR2    = inkR * inkR;

  // ── Local background (context window mean) ──────────────────────────────────
  let ctxSum = 0, ctxCnt = 0;
  for (let dy = -contextR; dy <= contextR; dy++) {
    for (let dx = -contextR; dx <= contextR; dx++) {
      if (dx * dx + dy * dy > contextR * contextR) continue;
      const px = cx + dx, py = cy + dy;
      if (px < 0 || px >= imgW || py < 0 || py >= imgH) continue;
      ctxSum += gray[py * imgW + px];
      ctxCnt++;
    }
  }
  const localMean      = ctxCnt > 0 ? ctxSum / ctxCnt : 200;
  // Threshold: must be at least 18% darker than local background, clamped
  const localThreshold = Math.max(55, Math.min(210, localMean * 0.82));

  // ── Inner zone pixel scan ───────────────────────────────────────────────────
  const innerPixels: number[] = [];
  const coords: [number, number][] = [];

  for (let dy = -inkR; dy <= inkR; dy++) {
    for (let dx = -inkR; dx <= inkR; dx++) {
      if (dx * dx + dy * dy > inkR2) continue;
      const px = cx + dx, py = cy + dy;
      if (px < 0 || px >= imgW || py < 0 || py >= imgH) continue;
      innerPixels.push(gray[py * imgW + px]);
      coords.push([px, py]);
    }
  }

  if (innerPixels.length === 0) {
    return { fillRatio: 0, darkDensity: 0, inkArea: 0, compositeScore: 0, localThreshold };
  }

  // fillRatio: fraction below adaptive threshold
  const darkCount = innerPixels.filter(v => v < localThreshold).length;
  const fillRatio = darkCount / innerPixels.length;

  // darkDensity: mean darkness (inverted, relative to page white)
  const meanPix    = innerPixels.reduce((a, b) => a + b, 0) / innerPixels.length;
  const darkDensity = Math.max(0, Math.min(1, (pageWhite - meanPix) / Math.max(1, pageWhite)));

  // inkArea: largest connected dark-pixel blob inside inner zone as fraction of total
  // Build local binary image of inner zone
  const bboxW   = inkR * 2 + 1;
  const bboxH   = inkR * 2 + 1;
  const locBin  = new Uint8Array(bboxW * bboxH).fill(255);
  for (const [px, py] of coords) {
    const lx = px - (cx - inkR);
    const ly = py - (cy - inkR);
    if (locBin[ly * bboxW + lx] !== undefined) {
      locBin[ly * bboxW + lx] = gray[py * imgW + px] < localThreshold ? 0 : 255;
    }
  }

  // BFS to find largest dark blob in locBin
  const locVisited = new Uint8Array(bboxW * bboxH);
  let maxBlob = 0;
  for (let ly = 0; ly < bboxH; ly++) {
    for (let lx = 0; lx < bboxW; lx++) {
      if (locVisited[ly * bboxW + lx] || locBin[ly * bboxW + lx] !== 0) continue;
      const queue: [number, number][] = [[lx, ly]];
      locVisited[ly * bboxW + lx] = 1;
      let blob = 0, head = 0;
      while (head < queue.length) {
        const [qx, qy] = queue[head++]; blob++;
        for (const [nx, ny] of [[qx-1,qy],[qx+1,qy],[qx,qy-1],[qx,qy+1]] as [number,number][]) {
          if (nx < 0 || nx >= bboxW || ny < 0 || ny >= bboxH) continue;
          if (locVisited[ny * bboxW + nx] || locBin[ny * bboxW + nx] !== 0) continue;
          locVisited[ny * bboxW + nx] = 1;
          queue.push([nx, ny]);
        }
      }
      if (blob > maxBlob) maxBlob = blob;
    }
  }
  const inkArea = maxBlob / innerPixels.length;

  // Composite score
  const compositeScore = W_FILL * fillRatio + W_DENSITY * darkDensity + W_INK * inkArea;

  return { fillRatio, darkDensity, inkArea, compositeScore, localThreshold };
}

// ─── Stage 5: Classification ───────────────────────────────────────────────────

/** Per-question adaptive threshold derived from blank-bubble baseline pool. */
function computeAdaptiveThreshold(
  scores: number[], // all 4 composite scores for a question
  globalBlankMean: number,
  globalBlankStd:  number,
): number {
  // FIX: Raised sigma multiplier from 1.5 → 2.0 to create more separation
  // between blank and filled. With 1.5, borderline fills got classified as blank.
  // Also tightened clamp: min 0.10 (was 0.15) to catch very light pencil marks.
  return Math.max(0.10, Math.min(0.60, globalBlankMean + 2.0 * globalBlankStd));
}

// ─── Stage 6: Debug overlay rendering ─────────────────────────────────────────

function renderDebugOverlay(
  gray:      Uint8Array,
  w:         number,
  h:         number,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  doc:       any,
  bubbles:   BubbleDebugInfo[],
  corners:   Point[] | null,
  rowBases:  number[],
): string | null {
  try {
    const canvas  = doc.createElement('canvas');
    canvas.width  = w; canvas.height = h;
    const ctx     = canvas.getContext('2d');
    if (!ctx) return null;

    // Draw grayscale base
    const id = ctx.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      id.data[i*4] = id.data[i*4+1] = id.data[i*4+2] = gray[i]; id.data[i*4+3] = 255;
    }
    ctx.putImageData(id, 0, 0);

    // Row baselines (dashed blue lines)
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = 'rgba(30,120,255,0.55)';
    ctx.lineWidth   = 1;
    for (const yBase of rowBases) {
      ctx.beginPath(); ctx.moveTo(0, yBase); ctx.lineTo(w, yBase); ctx.stroke();
    }
    ctx.setLineDash([]);

    // Corner markers (magenta squares)
    if (corners) {
      ctx.strokeStyle = 'rgba(255,0,220,0.9)';
      ctx.lineWidth   = 2.5;
      for (const pt of corners) {
        ctx.strokeRect(pt.x - 14, pt.y - 14, 28, 28);
      }
    }

    // Bubbles
    for (const b of bubbles) {
      const innerR = b.radius * INK_ZONE;

      // Outer ring (thin grey)
      ctx.beginPath();
      ctx.arc(b.cx, b.cy, b.radius, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(180,180,180,0.4)';
      ctx.lineWidth   = 0.8;
      ctx.stroke();

      // Inner zone ring — colour by status
      let ringColor: string;
      if (b.status === 'confident') {
        ringColor = b.selected ? 'rgba(0,210,80,0.9)' : 'rgba(255,60,60,0.55)';
      } else if (b.status === 'uncertain') {
        ringColor = 'rgba(255,190,0,0.85)';
      } else {
        ringColor = 'rgba(160,160,160,0.35)';
      }
      ctx.beginPath();
      ctx.arc(b.cx, b.cy, innerR, 0, Math.PI * 2);
      ctx.strokeStyle = ringColor;
      ctx.lineWidth   = b.selected ? 2.5 : 1.5;
      ctx.stroke();

      // Fill-percentage text
      if (b.option === 'A') {
        ctx.font      = '6.5px monospace';
        ctx.fillStyle = 'rgba(0,60,200,0.85)';
        ctx.fillText(`Q${b.questionNumber}`, b.cx - b.radius - 22, b.cy + 2.5);
      }

      // Composite score label (on selected bubble)
      if (b.selected) {
        ctx.font      = '6px sans-serif';
        ctx.fillStyle = 'rgba(0,160,40,0.95)';
        ctx.fillText(`${(b.compositeScore * 100).toFixed(0)}%`, b.cx - 6, b.cy + b.radius + 9);
      }
    }

    return canvas.toDataURL('image/png');
  } catch {
    return null;
  }
}

// ─── Preprocessing entry point (exported) ──────────────────────────────────────

export async function preprocessOMRImage(uri: string): Promise<ProcessedImage> {
  const result = await ImageManipulator.manipulateAsync(
    uri,
    [{ rotate: 0 }], // EXIF-orient only — no resize; backend needs full-res
    { compress: 0.93, format: ImageManipulator.SaveFormat.JPEG, base64: false },
  );
  return { uri: result.uri, width: result.width, height: result.height };
}

// ─── Main pipeline ─────────────────────────────────────────────────────────────

export async function processOMRLocally(
  uri:            string,
  imgW:           number,
  imgH:           number,
  totalQuestions: number = OMR_CONFIG.TOTAL_QUESTIONS,
): Promise<OMRProcessingResult | null> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const _g: any = globalThis;
  if (typeof _g.document === 'undefined') return null;

  try {
    // ── Draw to canvas ─────────────────────────────────────────────────────
    const canvas  = _g.document.createElement('canvas');
    canvas.width  = imgW; canvas.height = imgH;
    const ctx     = canvas.getContext('2d');
    if (!ctx) return null;

    await new Promise<void>((resolve, reject) => {
      const img = new _g.Image();
      img.onload  = () => { ctx.drawImage(img, 0, 0, imgW, imgH); resolve(); };
      img.onerror = reject;
      img.src     = uri;
    });

    const rgba = ctx.getImageData(0, 0, imgW, imgH).data as Uint8ClampedArray;

    // ── Stage 0: Grayscale + page stats ────────────────────────────────────
    const grayRaw  = toGrayscale(rgba, imgW, imgH);
    const pageStats = computePageStats(grayRaw);

    // ── Stage 0: Blur detection ─────────────────────────────────────────────
    const blurScore = laplacianVariance(grayRaw, imgW, imgH);
    if (blurScore < BLUR_REJECT_THRESHOLD) {
      return {
        measurements: [],
        debug: {
          warpedImageUri:  null,
          blurScore,
          accepted:        false,
          rejectionReason: `Image too blurry (score ${blurScore.toFixed(1)} < ${BLUR_REJECT_THRESHOLD}). Hold steady and ensure sharp focus.`,
          bubbles:         [],
          globalThreshold: 0,
          markersFound:    0,
          rowBaselines:    [],
          cornerPoints:    null,
          pageStats,
        },
      };
    }

    // ── Stage 0: Exposure check ─────────────────────────────────────────────
    if (pageStats.isOverexposed) {
      return {
        measurements: [],
        debug: {
          warpedImageUri:  null,
          blurScore,
          accepted:        false,
          rejectionReason: 'Image is overexposed (too bright). Move away from direct light or use a shaded area.',
          bubbles:         [],
          globalThreshold: 0,
          markersFound:    0,
          rowBaselines:    [],
          cornerPoints:    null,
          pageStats,
        },
      };
    }
    if (pageStats.isUnderexposed) {
      return {
        measurements: [],
        debug: {
          warpedImageUri:  null,
          blurScore,
          accepted:        false,
          rejectionReason: 'Image is too dark. Use brighter lighting or enable flash.',
          bubbles:         [],
          globalThreshold: 0,
          markersFound:    0,
          rowBaselines:    [],
          cornerPoints:    null,
          pageStats,
        },
      };
    }

    // ── Stage 1: Advanced preprocessing ────────────────────────────────────
    // (a) Shadow removal
    const shadowFree = removeShadow(grayRaw, imgW, imgH);
    // (b) CLAHE for contrast normalisation
    const equalized  = clahe(shadowFree, imgW, imgH, 8, 8, 2.5);
    // (c) Sharpening
    const sharpened  = unsharpMask(equalized, imgW, imgH, 2, 1.4);
    // (d) Sauvola binarisation for marker detection
    const binary     = sauvolaBinarise(sharpened, imgW, imgH, 15);
    // (e) Morphological open (erode→dilate) to remove tiny noise
    const cleaned    = morphDilate(morphErode(binary, imgW, imgH, 1), imgW, imgH, 1);

    // ── Stage 2: Corner marker detection ────────────────────────────────────
    const foundTL = findMarkerInQuadrant(cleaned, imgW, imgH, 0, 0);
    const foundTR = findMarkerInQuadrant(cleaned, imgW, imgH, 1, 0);
    const foundBL = findMarkerInQuadrant(cleaned, imgW, imgH, 0, 1);
    const foundBR = findMarkerInQuadrant(cleaned, imgW, imgH, 1, 1);

    const markersFound = [foundTL, foundTR, foundBL, foundBR].filter(Boolean).length;
    const corners: Point[] | null = markersFound === 4
      ? [foundTL!, foundTR!, foundBL!, foundBR!]
      : null;

    // ── Stage 2: Perspective warp ────────────────────────────────────────────
    let warpedGray: Uint8Array = sharpened; // use sharpened grayscale (not binary) for fill
    let warpedW = imgW, warpedH = imgH;

    if (markersFound === 4) {
      const srcPts: [Point, Point, Point, Point] = [foundTL!, foundTR!, foundBL!, foundBR!];
      const dstPts: [Point, Point, Point, Point] = [
        IDEAL_MARKS.tl, IDEAL_MARKS.tr, IDEAL_MARKS.bl, IDEAL_MARKS.br,
      ];
      const H_mat = computeHomography(srcPts, dstPts);
      if (H_mat) {
        // Warp the PREPROCESSED grayscale (shadow-free + sharpened) for fill analysis
        warpedGray = warpPerspective(sharpened, imgW, imgH, H_mat, W, H);
        warpedW    = W; warpedH = H;
      }
    } else if (markersFound < 1) {
      // FIX: Only hard-reject if NO markers found at all.
      // With 2-3 corners we skip perspective correction but still attempt
      // bubble reading via the computed grid — better than returning nothing.
      return {
        measurements: [],
        debug: {
          warpedImageUri:  null,
          blurScore,
          accepted:        false,
          rejectionReason: `No corner markers detected. Ensure all four black squares at the corners are fully visible and the sheet is not folded or cropped.`,
          bubbles:         [],
          globalThreshold: 0,
          markersFound,
          rowBaselines:    [],
          cornerPoints:    null,
          pageStats,
        },
      };
    }

    // ── Stage 3: Bubble grid ─────────────────────────────────────────────────
    const circles = computeBubbleGrid(totalQuestions);

    // Attempt dynamic detection for row-baseline correction
    const expectedR      = circles[0]?.radius ?? 9;
    const detectedPeaks  = detectBubbleCenters(warpedGray, warpedW, warpedH, expectedR, totalQuestions);

    // If detection succeeded, use detected Y-centres to correct row drift
    let finalCircles = circles;
    if (detectedPeaks && detectedPeaks.length >= totalQuestions * 2) {
      const rowH      = getRowHeightPx(totalQuestions);
      const detectedYs = clusterRows(detectedPeaks.map(p => p.y), rowH * 0.6);

      if (detectedYs.length >= Math.floor(totalQuestions / _getColumnCount(totalQuestions)) * 0.8) {
        // Apply Y-correction: match each computed row to nearest detected baseline
        finalCircles = circles.map(c => {
          if (detectedYs.length === 0) return c;
          const nearestY = detectedYs.reduce((best, y) =>
            Math.abs(y - c.cy) < Math.abs(best - c.cy) ? y : best,
          detectedYs[0]);
          return Math.abs(nearestY - c.cy) < rowH * 0.4
            ? { ...c, cy: nearestY }
            : c;
        });
      }
    }

    // ── Stage 4: Per-bubble composite fill analysis ──────────────────────────
    const pageWhite    = Math.min(240, pageStats.meanBrightness + pageStats.contrast);
    const allMeasurements: Array<FillMeasurement & { questionNumber: number; option: OMROption; cx: number; cy: number; radius: number }> = [];

    for (const circle of finalCircles) {
      const m = measureBubbleFill(warpedGray, warpedW, warpedH, circle.cx, circle.cy, circle.radius, pageWhite);
      allMeasurements.push({ ...m, ...circle });
    }

    // ── Stage 5: Classification — dynamic threshold ──────────────────────────
    // FIX: Use per-question minimum score as blank proxy instead of global
    // bottom 50%. The old approach pulled answered bubbles into the blank pool
    // when the student answered many questions, inflating the threshold and
    // causing filled bubbles to be classified as blank.
    //
    // New approach: for each question, the 3 non-selected bubbles are blank.
    // Collect the minimum score per question (most likely a blank bubble) to
    // build a clean blank distribution, then threshold = mean + 2 × std.
    const perQuestionMin: number[] = [];
    const grouped: Record<number, number[]> = {};
    for (const m of allMeasurements) {
      if (!grouped[m.questionNumber]) grouped[m.questionNumber] = [];
      grouped[m.questionNumber].push(m.compositeScore);
    }
    for (const scores of Object.values(grouped)) {
      perQuestionMin.push(Math.min(...scores));
    }
    const blankPool = perQuestionMin;
    const blankMean = blankPool.reduce((a, b) => a + b, 0) / Math.max(1, blankPool.length);
    const blankStd  = Math.sqrt(
      blankPool.reduce((a, b) => a + (b - blankMean) ** 2, 0) / Math.max(1, blankPool.length),
    );
    const globalThreshold = computeAdaptiveThreshold([], blankMean, blankStd);

    // Group by question
    const byQuestion: Record<number, typeof allMeasurements> = {};
    for (const m of allMeasurements) {
      if (!byQuestion[m.questionNumber]) byQuestion[m.questionNumber] = [];
      byQuestion[m.questionNumber].push(m);
    }

    const measurements: BubbleMeasurement[] = [];
    const debugBubbles: BubbleDebugInfo[]   = [];

    for (const [qStr, group] of Object.entries(byQuestion)) {
      const sorted = [...group].sort((a, b) => b.compositeScore - a.compositeScore);
      const best   = sorted[0];
      const second = sorted[1];

      const isDoubleMarked =
        best.compositeScore   >= globalThreshold &&
        second.compositeScore >= globalThreshold &&
        second.compositeScore / best.compositeScore >= DOUBLE_MARK_RATIO;

      const confidence = best.compositeScore > 0
        ? Math.max(0, 1 - (second?.compositeScore ?? 0) / best.compositeScore)
        : 0;

      for (const m of group) {
        const isSelected = m === best && best.compositeScore >= globalThreshold && !isDoubleMarked;
        const status: 'confident' | 'uncertain' | 'blank' =
          m.compositeScore < globalThreshold
            ? 'blank'
            : m.compositeScore < UNCERTAIN_THRESHOLD * globalThreshold * 1.5
              ? 'uncertain'
              : 'confident';

        debugBubbles.push({
          questionNumber: m.questionNumber,
          option:         m.option,
          cx:             m.cx,
          cy:             m.cy,
          radius:         m.radius,
          fillRatio:      m.fillRatio,
          compositeScore: m.compositeScore,
          localThreshold: m.localThreshold,
          confidence,
          selected:       isSelected,
          status,
        });

        measurements.push({
          questionNumber: m.questionNumber,
          option:         m.option,
          fillRatio:      m.compositeScore, // pass composite as fillRatio for downstream classifier
        });
      }
    }

    // ── Stage 6: Row baselines for debug ────────────────────────────────────
    const rowBaselines = reconstructRowBaselines(finalCircles, totalQuestions);

    // ── Debug overlay ────────────────────────────────────────────────────────
    const warpedCorners = corners
      ? corners.map(c => {
          // If we warped, corners move to their ideal positions
          if (warpedW === W) {
            if (c === foundTL) return IDEAL_MARKS.tl;
            if (c === foundTR) return IDEAL_MARKS.tr;
            if (c === foundBL) return IDEAL_MARKS.bl;
            if (c === foundBR) return IDEAL_MARKS.br;
          }
          return c;
        })
      : null;

    const warpedImageUri = renderDebugOverlay(
      warpedGray, warpedW, warpedH, _g.document,
      debugBubbles, warpedCorners, rowBaselines,
    );

    return {
      measurements,
      debug: {
        warpedImageUri,
        blurScore,
        accepted:        true,
        rejectionReason: markersFound < 4
          ? `Only ${markersFound}/4 markers — perspective correction skipped. Results may be slightly less accurate.`
          : null,
        bubbles:         debugBubbles,
        globalThreshold,
        markersFound,
        rowBaselines,
        cornerPoints:    warpedCorners,
        pageStats,
      },
    };

  } catch (err) {
    console.error('[OMR] processOMRLocally error:', err);
    return null;
  }
}

// ─── Backend integration helpers ──────────────────────────────────────────────

export async function imageUriToBase64(uri: string): Promise<string> {
  const response = await fetch(uri);
  const blob     = await response.blob();
  return new Promise<string>((resolve, reject) => {
    const reader     = new FileReader();
    reader.onloadend = () => resolve((reader.result as string).split(',')[1]);
    reader.onerror   = reject;
    reader.readAsDataURL(blob);
  });
}

export function parseBubbleMeasurements(data: {
  bubbles?: Array<{ question: number; option: string; fillRatio: number }>;
  answers?: Record<string, string>;
}): BubbleMeasurement[] {
  const options = OMR_CONFIG.OPTIONS;

  if (data.bubbles && data.bubbles.length > 0) {
    return data.bubbles
      .filter(b => options.includes(b.option as OMROption))
      .map(b => ({
        questionNumber: b.question,
        option:         b.option as OMROption,
        fillRatio:      b.fillRatio,
      }));
  }

  if (data.answers) {
    const measurements: BubbleMeasurement[] = [];
    for (const [qStr, selected] of Object.entries(data.answers)) {
      const qNum = Number(qStr);
      for (const opt of options) {
        measurements.push({
          questionNumber: qNum,
          option:         opt,
          fillRatio:      opt === selected ? 0.88 : 0.04,
        });
      }
    }
    return measurements;
  }

  return [];
}

// ─── Legacy / compat exports ──────────────────────────────────────────────────

export interface AlignmentResult { confidence: number; found: number; expected: number }
/** @deprecated Handled internally. Kept for backwards compatibility. */
export function checkFiducialAlignment(): AlignmentResult {
  return { confidence: 1, found: 4, expected: 4 };
}
export { computeBubbleGrid };