// src/utils/ocrParser.ts
// ─────────────────────────────────────────────────────────────────────────────
// CHANGES FROM PREVIOUS VERSION:
//
//  ✅ FIX 5: parseMultipleChoiceAnswer() — smarter pattern prevents picking up
//     printed choice labels (e.g. "A. CSS" or "B. HTML") as the student's answer.
//
//  How the bug manifested:
//    If Tesseract read "A. CSS B. HTML C. Python D. Java" from the printed choices,
//    the old pattern `/\b([A-Da-d])\b/` matched the FIRST A-D letter found = "A",
//    even though the student actually wrote "B".
//
//  The fix:
//    1. First try to match an ISOLATED letter (not followed by ". word") using the
//       new RE_STANDALONE pattern. This matches a handwritten answer.
//    2. Only fall back to the old greedy match if no standalone letter is found.
//
//  All other functions are unchanged.
// ─────────────────────────────────────────────────────────────────────────────

import type { ExamType, FlatAnswerKey } from '../types/exam';

// ─── Public types ─────────────────────────────────────────────────────────────

export interface RawOcrPayload {
  answers:     Record<string, string>;
  studentName: string | null;
  confidence:  number;
  notes:       string;
}

export interface ParsedScanPayload {
  studentAnswers:      FlatAnswerKey;
  detectedStudentName: string | null;
  ocrConfidence:       number;
  ocrNotes:            string;
  flaggedQuestions:    string[];
}

// ─── Per-type answer parsers ──────────────────────────────────────────────────

function parseBubbleAnswer(raw: string, questionNumber: string, flagged: string[]): string {
  const v = raw.trim().toUpperCase();
  if (['A', 'B', 'C', 'D'].includes(v)) return v;
  flagged.push(questionNumber);
  return '';
}

// ── FIX 5 ─────────────────────────────────────────────────────────────────────
//
// ROOT CAUSE:
//   The old implementation took the FIRST A-D letter found in the raw OCR string.
//   When Tesseract read the printed choices "A. CSS  B. HTML  C. Python  D. Java",
//   it returned "A" — the first letter — even though the student wrote "B".
//
// FIX:
//   We now try two patterns in order of specificity:
//
//   1. RE_STANDALONE — matches a letter that is NOT followed by ". word" (choice format)
//      "B" → matches ✓
//      "A. CSS" → does NOT match (A is followed by ". C")
//      This catches the student's isolated handwritten letter.
//
//   2. RE_ANY — fallback, same as the old pattern (matches the first A-D letter found)
//      Only used when Strategy 1 finds nothing.
//
//   After the match, we also run a final preference step: if the raw OCR contains
//   multiple A-D letters, we prefer the one that appears at the END of the string
//   (student answers often appear after the question number, at the end of the line)
//   over the one at the START (which is more likely to be a printed choice label).

function parseMultipleChoiceAnswer(raw: string, questionNumber: string, flagged: string[]): string {
  const str = raw.trim();

  // Strategy 1: Find a STANDALONE letter — not followed by ". word" (= printed choice)
  // Pattern breakdown:
  //   \b         — word boundary (letter must stand alone as a token)
  //   ([A-Da-d]) — the letter A-D
  //   \b         — word boundary
  //   (?!        — negative lookahead: fail if the letter is followed by...
  //     \s*[.)\-]\s*[A-Za-z]  — a period/paren then text (= printed choice label format)
  //   )
  const RE_STANDALONE = /\b([A-Da-d])\b(?!\s*[.)\-]\s*[A-Za-z])/g;
  const standaloneMatches: string[] = [];
  let m: RegExpExecArray | null;
  while ((m = RE_STANDALONE.exec(str)) !== null) {
    standaloneMatches.push(m[1].toUpperCase());
  }

  if (standaloneMatches.length > 0) {
    // If multiple standalone letters found, prefer the LAST one.
    // Rationale: "1. B" → "B" at the end is the answer; printed choices "A. B. C. D."
    // would have already been filtered by RE_STANDALONE, so the last match is most reliable.
    return standaloneMatches[standaloneMatches.length - 1];
  }

  // Strategy 2: Fallback — extract the last A-D letter in the string.
  // The LAST occurrence is more likely to be the student's answer than the first
  // (which is more likely to be the first printed choice label "A.").
  const allLetters = Array.from(str.matchAll(/\b([A-Da-d])\b/gi)).map(x => x[1].toUpperCase());
  if (allLetters.length > 0) {
    return allLetters[allLetters.length - 1];
  }

  // Nothing found — flag this question
  flagged.push(questionNumber);
  return '';
}

function parseIdentificationAnswer(raw: string): string {
  return raw
    .trim()
    .replace(/\s+/g, ' ')
    .replace(/^[^a-zA-Z0-9]+/, '')
    .replace(/[^a-zA-Z0-9]+$/, '');
}

function parseEnumerationAnswer(raw: string): string {
  const trimmed = raw.trim();

  const lines = trimmed.split('\n').map(l => l.trim()).filter(Boolean);
  if (lines.length > 1) {
    const items = lines
      .map(line => line.replace(/^\d+[\.\)]\s*/, '').trim())
      .filter(Boolean);
    if (items.length > 0) return items.join(';');
  }

  return parseIdentificationAnswer(raw);
}

function parseTrueFalseAnswer(raw: string, questionNumber: string, flagged: string[]): string {
  const v = raw.trim().toLowerCase();
  if (v === 't' || v === 'true')  return 'T';
  if (v === 'f' || v === 'false') return 'F';
  flagged.push(questionNumber);
  return '';
}

// ─── Main parser ──────────────────────────────────────────────────────────────

// FIX 6: per-question type dispatcher for mixed-mode scans
function parseAnswerForType(
  rawText: string,
  type:    ExamType,
  qKey:    string,
  flagged: string[],
): string {
  switch (type) {
    case 'bubble_omr':      return parseBubbleAnswer(rawText, qKey, flagged);
    case 'multiple_choice': return parseMultipleChoiceAnswer(rawText, qKey, flagged);
    case 'identification':  return parseIdentificationAnswer(rawText);
    case 'enumeration':     return parseEnumerationAnswer(rawText);
    case 'true_or_false':   return parseTrueFalseAnswer(rawText, qKey, flagged);
    default: {
      const _exhaustive: never = type;
      console.warn('[ocrParser] Unknown exam type:', _exhaustive);
      flagged.push(qKey);
      return rawText;
    }
  }
}

// FIX 6: added optional questionTypeMap for mixed-mode per-question routing.
// When present, each question uses its own type instead of the global examType.
// This prevents MC parsing from stripping T/F ("False"→""), identification
// ("CSS"→""), and enumeration answers that don't look like A/B/C/D.
export function parseOcrPayload(
  raw:              RawOcrPayload,
  examType:         ExamType,
  questionCount:    number,
  questionTypeMap?: Record<string, ExamType>,
): ParsedScanPayload {
  const studentAnswers: FlatAnswerKey = {};
  const flaggedQuestions: string[]    = [];

  for (let i = 1; i <= questionCount; i++) {
    const qKey    = String(i);
    const rawText = (raw.answers[qKey] ?? '').trim();

    if (rawText === '') {
      studentAnswers[qKey] = '';
      flaggedQuestions.push(qKey);
      continue;
    }

    // Use per-question type in mixed mode, fall back to global examType
    const typeForQuestion: ExamType = questionTypeMap?.[qKey] ?? examType;
    studentAnswers[qKey] = parseAnswerForType(rawText, typeForQuestion, qKey, flaggedQuestions);
  }

  const baseNotes = raw.notes?.trim() ?? '';
  const flagNotes =
    flaggedQuestions.length > 0
      ? `Flagged questions (low confidence or unparseable): ${flaggedQuestions.join(', ')}.`
      : '';
  const ocrNotes = [baseNotes, flagNotes].filter(Boolean).join(' ');

  return {
    studentAnswers,
    detectedStudentName: raw.studentName?.trim() || null,
    ocrConfidence:       raw.confidence ?? 0,
    ocrNotes,
    flaggedQuestions,
  };
}

// ─── Student name extractor ───────────────────────────────────────────────────

export function extractStudentName(rawText: string): string | null {
  if (!rawText) return null;

  const nameMatch = rawText.match(/name[:\s]+([A-Za-z\s,.''-]{3,60})/i);
  if (nameMatch) {
    return nameMatch[1].trim().replace(/\s+/g, ' ');
  }

  const lines = rawText.split('\n').map(l => l.trim()).filter(Boolean);
  for (const line of lines.slice(0, 3)) {
    if (/^[A-Za-z\s,.''-]{3,60}$/.test(line)) {
      return line;
    }
  }

  return null;
}

// ─── Confidence classifier ────────────────────────────────────────────────────

export type OcrConfidenceLevel = 'high' | 'medium' | 'low';

export function classifyConfidence(confidence: number): OcrConfidenceLevel {
  if (confidence >= 0.85) return 'high';
  if (confidence >= 0.60) return 'medium';
  return 'low';
}