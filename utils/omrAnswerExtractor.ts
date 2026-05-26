// src/utils/omrAnswerExtractor.ts  (FIXED v2)
// ─────────────────────────────────────────────────────────────────────────────
// Drop-in replacement for omrAnswerExtractor.ts.
// All existing exports are preserved — no callers need to change.
//
// What changed vs the original:
//
//  FIX 1 (High): computeAdaptiveThreshold() — NEW exported function.
//         Instead of the hardcoded FILL_THRESHOLD = 0.45, the threshold is now
//         derived per-scan from the distribution of fill ratios:
//           threshold = baselineMean + 3 × baselineStd
//         where baseline = the lowest fill ratio per question (blank bubbles).
//         Clamped to [0.20, 0.65] so extreme images don't produce nonsense.
//         Result: works correctly in dim classrooms, on yellowish paper, and
//         with both light pencil marks and heavy ballpen fills.
//
//  FIX 2 (High): classifyBubbles() now calls computeAdaptiveThreshold() by
//         default when no threshold is explicitly supplied. Callers that pass
//         an explicit threshold (e.g. from server config) are unaffected.
//
//  FIX 3 (Medium): INVALID detection is now margin-aware.
//         Previously two bubbles both above the threshold = INVALID with no
//         further analysis. Now: if two bubbles are above threshold BUT one is
//         clearly darker (margin > adaptiveMargin), the darker one wins — this
//         handles sheets where the student erased and re-filled.
//         If the two are truly close (margin ≤ adaptiveMargin), it is INVALID.
//
//  FIX 4 (Low): summariseClassifications() now also returns a doubleMarked count
//         so the UI can display a warning badge when relevant.
// ─────────────────────────────────────────────────────────────────────────────

import { OMR_CONFIG, OMROption, OMRScanResult, OMRSheetMeta } from '../constants/omrConfig';
import { omrAnswersToFlatKey } from './omrSheetGenerator';

// ─── Types ────────────────────────────────────────────────────────────────────

/** One bubble's fill measurement — produced by omrImageProcessor. */
export interface BubbleMeasurement {
  questionNumber: number;
  option:         OMROption;
  /** 0–1: fraction of pixels darker than the threshold inside the bubble circle. */
  fillRatio:      number;
}

/** Per-question classification result. */
export interface QuestionClassification {
  questionNumber: number;
  selected:       OMROption | 'BLANK' | 'INVALID';
  fillRatios:     Record<OMROption, number>;
  /** True when two or more bubbles exceeded the threshold AND were too close to call. */
  multipleMarked: boolean;
}

/** Result of computeAdaptiveThreshold — exposed so the UI can log/display it. */
export interface ThresholdInfo {
  /** The fill threshold used for this scan. */
  threshold:     number;
  /** The margin threshold: filled bubble must exceed runner-up by this much. */
  margin:        number;
  /** Mean fill ratio of the estimated blank bubbles across the sheet. */
  baselineMean:  number;
  /** Std dev of blank bubble fill ratios. */
  baselineStd:   number;
}

// ─── FIX 1: Adaptive threshold ────────────────────────────────────────────────

/**
 * Computes a per-scan fill threshold from the distribution of raw fill ratios.
 *
 * Algorithm:
 *   1. For each question, take the MINIMUM fill ratio across all options.
 *      Statistically this is most likely a blank bubble.
 *   2. Compute mean + std of those minima → this is the "blank bubble baseline".
 *   3. threshold = baselineMean + 3 × baselineStd  (3-sigma rule)
 *   4. margin    = max(0.06, 1.5 × baselineStd)
 *
 * Both values are clamped to safe ranges so extreme images don't break things.
 *
 * @param measurements  Raw BubbleMeasurement[] from omrImageProcessor
 * @returns             ThresholdInfo (threshold, margin, baselineMean, baselineStd)
 */
export function computeAdaptiveThreshold(measurements: BubbleMeasurement[]): ThresholdInfo {
  // Group by question
  const byQuestion: Record<number, number[]> = {};
  for (const m of measurements) {
    if (!byQuestion[m.questionNumber]) byQuestion[m.questionNumber] = [];
    byQuestion[m.questionNumber].push(m.fillRatio);
  }

  // Minimum fill ratio per question = best estimate of a blank bubble
  const minRatios = Object.values(byQuestion).map(ratios => Math.min(...ratios));

  if (minRatios.length === 0) {
    // No data — fall back to config default
    return {
      threshold:    OMR_CONFIG.FILL_THRESHOLD,
      margin:       0.08,
      baselineMean: OMR_CONFIG.FILL_THRESHOLD * 0.4,
      baselineStd:  0.05,
    };
  }

  const baselineMean =
    minRatios.reduce((s, v) => s + v, 0) / minRatios.length;

  const baselineStd = Math.sqrt(
    minRatios.reduce((s, v) => s + (v - baselineMean) ** 2, 0) / minRatios.length
  );

  // 3-sigma above blank baseline, clamped to [0.20, 0.65]
  const threshold = Math.max(0.20, Math.min(0.65, baselineMean + 3.0 * baselineStd));

  // Margin: filled bubble must be at least this much above the runner-up
  const margin = Math.max(0.06, Math.min(0.20, baselineStd * 1.5));

  return { threshold, margin, baselineMean, baselineStd };
}

// ─── FIX 2 + FIX 3: Core classifier ──────────────────────────────────────────

/**
 * Classifies each question by finding which (if any) bubble is filled.
 *
 * Rules:
 *   0 above threshold                  → BLANK
 *   1 above threshold                  → that option (A/B/C/D)
 *   2+ above threshold, clear winner   → winner (erase-and-remark case)
 *   2+ above threshold, too close      → INVALID (double-mark)
 *
 * @param measurements   Array of BubbleMeasurement from omrImageProcessor
 * @param thresholdInfo  Optional — if omitted, computed adaptively from measurements
 */
export function classifyBubbles(
  measurements:  BubbleMeasurement[],
  thresholdInfo?: ThresholdInfo,
): QuestionClassification[] {
  // FIX 2: derive threshold adaptively when not supplied
  const info = thresholdInfo ?? computeAdaptiveThreshold(measurements);
  const { threshold, margin: adaptiveMargin } = info;

  // Group by question
  const grouped: Record<number, BubbleMeasurement[]> = {};
  for (const m of measurements) {
    if (!grouped[m.questionNumber]) grouped[m.questionNumber] = [];
    grouped[m.questionNumber].push(m);
  }

  const results: QuestionClassification[] = [];

  for (const [qStr, bubbles] of Object.entries(grouped)) {
    const qNum = Number(qStr);

    // Build fill map
    const fillRatios = {} as Record<OMROption, number>;
    for (const b of bubbles) fillRatios[b.option] = b.fillRatio;

    // Sort bubbles darkest-first
    const sorted = [...bubbles].sort((a, b) => b.fillRatio - a.fillRatio);
    const best   = sorted[0];
    const second = sorted[1];

    const aboveThreshold = bubbles.filter(b => b.fillRatio >= threshold);

    let selected: OMROption | 'BLANK' | 'INVALID';
    let multipleMarked = false;

    if (aboveThreshold.length === 0) {
      selected = 'BLANK';
    } else if (aboveThreshold.length === 1) {
      selected = aboveThreshold[0].option;
    } else {
      // FIX 3: two or more bubbles above threshold
      // If the best is clearly darker than the runner-up, accept it
      // (student erased one and filled another — common in classroom tests)
      const margin = best.fillRatio - (second?.fillRatio ?? 0);
      if (margin >= adaptiveMargin) {
        selected = best.option; // clear winner — accept it
      } else {
        selected       = 'INVALID';
        multipleMarked = true;
      }
    }

    results.push({ questionNumber: qNum, selected, fillRatios, multipleMarked });
  }

  return results.sort((a, b) => a.questionNumber - b.questionNumber);
}

// ─── Build OMRScanResult ──────────────────────────────────────────────────────

/**
 * Converts classified bubbles + sheet meta into an OMRScanResult that can be
 * passed directly to omrAnswersToFlatKey() → gradeAnswers() in grading.ts.
 */
export function buildOMRScanResult(
  classifications: QuestionClassification[],
  meta:            Partial<OMRSheetMeta>,
): OMRScanResult {
  const answers: Record<number, OMROption | 'BLANK' | 'INVALID'> = {};
  for (const c of classifications) answers[c.questionNumber] = c.selected;

  return {
    studentId:  meta.studentId  ?? '',
    section:    meta.section    ?? '',
    examId:     meta.examId,
    scannedAt:  new Date().toISOString(),
    answers,
  };
}

// ─── Integration helper: OMR → existing grading pipeline ─────────────────────

/**
 * One-stop conversion:
 *   BubbleMeasurement[] + meta  →  FlatAnswerKey (Record<string, string>)
 *
 * Call this, then pass the result directly to gradeAnswers() from grading.ts
 * with examType = 'bubble_omr'.
 */
export function omrToFlatAnswerKey(
  measurements: BubbleMeasurement[],
  meta:         Partial<OMRSheetMeta>,
  thresholdInfo?: ThresholdInfo,
): Record<string, string> {
  const classifications = classifyBubbles(measurements, thresholdInfo);
  const scanResult      = buildOMRScanResult(classifications, meta);
  return omrAnswersToFlatKey(scanResult.answers);
}

// ─── FIX 4: Stats summary (now includes doubleMarked) ────────────────────────

export function summariseClassifications(classifications: QuestionClassification[]): {
  answered:     number;
  blank:        number;
  invalid:      number;
  doubleMarked: number;
  total:        number;
} {
  let answered = 0, blank = 0, invalid = 0, doubleMarked = 0;
  for (const c of classifications) {
    if      (c.selected === 'BLANK')   blank++;
    else if (c.selected === 'INVALID') { invalid++; if (c.multipleMarked) doubleMarked++; }
    else answered++;
  }
  return { answered, blank, invalid, doubleMarked, total: classifications.length };
}