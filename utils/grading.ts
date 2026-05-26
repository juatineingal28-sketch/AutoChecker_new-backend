// src/utils/grading.ts
// Strict deterministic grading — no AI, no fuzzy logic.
//
// Supported exam types: bubble_omr | multiple_choice | identification | enumeration | true_or_false
//
// CHANGES FROM PREVIOUS VERSION:
//  ✅ examType is now REQUIRED in gradeAnswers() — no heuristic fallback
//  ✅ detectNormType() removed — type is always known from the template
//  ✅ Uses FlatAnswerKey (Record<string, string>) from exam.ts
//  ✅ Enumeration: student answer matched against any semicolon-separated accepted value
//  ✅ NEW: requireUppercase param on gradeAnswers(), gradeAnswersMixed(), buildScanResult()
//          When true, any student answer containing a lowercase letter is immediately
//          marked wrong with uppercaseViolation:true in the breakdown — applies to
//          multiple_choice, identification, enumeration, true_or_false.
//          bubble_omr is EXCLUDED — OMR detection produces uppercase A–D automatically.
//  ✅ isUppercaseViolation() imported from exam.ts — single source of truth, no duplication.

import type {
  ExamType,
  FlatAnswerKey,
  GradeBreakdown,
  GradeResult,
  NewScanResult,
} from '../types/exam';
import { isUppercaseViolation } from '../types/exam';

// Re-export for consumers that still import from grading.ts
export type { GradeBreakdown, GradeResult };

// ─── Internal Normalisation Strategy ──────────────────────────────────────────

type NormType = 'mcq' | 'identification' | 'enumeration' | 'true_or_false';

/** Maps every ExamType (or string alias) to a NormType.
 *  ✅ Handles unknown/legacy type strings gracefully instead of silently breaking.
 *  ✅ Accepts 'true_false' as an alias for 'true_or_false' (used by auto-detect).
 */
function normTypeFor(examType: ExamType | string): NormType {
  switch (examType) {
    case 'bubble_omr':
    case 'multiple_choice':
      return 'mcq';
    case 'identification':
      return 'identification';
    case 'enumeration':
      return 'enumeration';
    case 'true_or_false':
    case 'true_false': // alias — guard against legacy values
      return 'true_or_false';
    default:
      // Unknown type — fall back to identification so written answers are
      // compared case-insensitively rather than silently scoring as wrong.
      return 'identification';
  }
}

// ─── Normalizers ───────────────────────────────────────────────────────────────

function normalizeMCQ(val: string): string {
  const v = val.trim().toUpperCase();
  return ['A', 'B', 'C', 'D'].includes(v) ? v : '';
}

function normalizeIdentification(val: string): string {
  return val
    .trim()
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .replace(/[.,!?;:'"]/g, '');
}

/**
 * Compact key: strips ALL whitespace and punctuation.
 * Secondary match so "My SQL"=="MySQL", "Node.js"=="Nodejs",
 * "Postgres SQL"=="PostgreSQL", "Mongo DB"=="MongoDB", etc.
 */
function normalizeCompact(val: string): string {
  return val
    .trim()
    .toLowerCase()
    .replace(/[\s.,!?;:'"\-_/\\]/g, '');
}

/**
 * Normalizes one side of an enumeration answer.
 * Splits on semicolons (accepted answer separator), sorts, and joins.
 * This lets "Red;Crimson" match "Crimson;Red".
 */
function normalizeEnumerationSide(val: string): string {
  return val
    .trim()
    .toLowerCase()
    .split(/;+/)
    .map(s => s.trim())
    .filter(Boolean)
    .sort()
    .join(';');
}

/**
 * Normalizes True/False answers.
 * Accepts T, F, True, False (case-insensitive) → canonical 'T' or 'F'.
 */
function normalizeTrueFalse(val: string): string {
  const v = val.trim().toLowerCase();
  if (v === 't' || v === 'true')  return 'T';
  if (v === 'f' || v === 'false') return 'F';
  return '';
}

function normalizeByType(val: string, type: NormType): string {
  switch (type) {
    case 'mcq':            return normalizeMCQ(val);
    case 'identification': return normalizeIdentification(val);
    case 'enumeration':    return normalizeEnumerationSide(val);
    case 'true_or_false':  return normalizeTrueFalse(val);
  }
}

// ─── Enumeration matching ──────────────────────────────────────────────────────

/**
 * For enumeration, the answer key stores semicolon-separated accepted values
 * (e.g. "Red;Crimson"). A student answer is correct if it matches ANY one
 * of those accepted values after normalisation.
 */
function matchEnumeration(studentRaw: string, correctRaw: string): boolean {
  const studentNorm    = normalizeIdentification(studentRaw);
  const studentCompact = normalizeCompact(studentRaw);
  const acceptedValues = correctRaw
    .split(';')
    .map(s => s.trim())
    .filter(Boolean);
  return acceptedValues.some(accepted => {
    // Primary: normalised match (keeps spaces, strips punctuation)
    if (normalizeIdentification(accepted) === studentNorm) return true;
    // Secondary: compact match — "My SQL" == "MySQL", "Node.js" == "Nodejs"
    if (normalizeCompact(accepted) === studentCompact && studentCompact !== '') return true;
    return false;
  });
}

// ─── Core Grading ──────────────────────────────────────────────────────────────

/**
 * Grades student answers against the answer key.
 *
 * examType is REQUIRED. The exam type determines how answers are normalised
 * and compared. There is no fallback heuristic — the caller must always
 * supply the correct type from the template.
 *
 * @param requireUppercase  When true, student answers containing any lowercase
 *   letter are marked wrong with uppercaseViolation:true in the breakdown.
 *   Does NOT affect bubble_omr. Defaults to false for backward compatibility.
 */
export function gradeAnswers(
  studentAnswers:  Record<string, string>,
  answerKey:       FlatAnswerKey,
  examType:        ExamType,
  requireUppercase = false,
): GradeResult {
  const breakdown: GradeBreakdown[] = [];
  let score      = 0;
  let unanswered = 0;

  const total    = Object.keys(answerKey).length;
  const normType = normTypeFor(examType);

  for (const q of Object.keys(answerKey)) {
    const correctRaw = (answerKey[q] ?? '').trim();
    const studentRaw = (studentAnswers[q] ?? '').trim();

    const isUnanswered = studentRaw === '';
    if (isUnanswered) unanswered++;

    let isCorrect          = false;
    let uppercaseViolation = false;
    let studentNorm        = '';
    let correctNorm        = '';

    if (!isUnanswered) {
      // ── ALL CAPS check (before content matching) ───────────────────────────
      // Evaluated first so a lowercase answer is immediately wrong, regardless
      // of whether its content would otherwise match.
      if (isUppercaseViolation(examType, studentRaw, requireUppercase)) {
        uppercaseViolation = true;
        isCorrect          = false;
        // Still compute display norms so the breakdown is readable.
        studentNorm = examType === 'enumeration'
          ? normalizeEnumerationSide(studentRaw)
          : normalizeByType(studentRaw, normType);
        correctNorm = examType === 'enumeration'
          ? normalizeEnumerationSide(correctRaw)
          : normalizeByType(correctRaw, normType);
      } else {
        // ── Normal content matching ──────────────────────────────────────────
        if (examType === 'enumeration') {
          // Enumeration: match student's answer against any accepted value
          isCorrect   = matchEnumeration(studentRaw, correctRaw);
          studentNorm = normalizeIdentification(studentRaw);
          correctNorm = normalizeEnumerationSide(correctRaw);
        } else {
          studentNorm = normalizeByType(studentRaw, normType);
          correctNorm = normalizeByType(correctRaw, normType);
          // Primary match: standard normalisation
          isCorrect = correctNorm !== '' && studentNorm === correctNorm;
          // Compact fallback for identification: "My SQL"=="MySQL", "Node.js"=="Nodejs"
          if (!isCorrect && normType === 'identification') {
            const sc = normalizeCompact(studentRaw);
            const cc = normalizeCompact(correctRaw);
            isCorrect = sc !== '' && cc !== '' && sc === cc;
          }
        }
      }
    }

    if (isCorrect) score++;

    const entry: GradeBreakdown = {
      questionNumber:    q,
      correct:           isCorrect,
      studentAnswer:     studentRaw || '—',
      correctAnswer:     correctRaw,
      normalizedStudent: studentNorm || '—',
      normalizedCorrect: correctNorm,
    };
    if (uppercaseViolation) entry.uppercaseViolation = true;
    breakdown.push(entry);
  }

  const wrong      = Math.max(total - score - unanswered, 0);
  const percentage = total > 0 ? Math.round((score / total) * 100) : 0;

  return {
    score,
    total,
    wrong,
    unanswered,
    percentage,
    passed: percentage >= 75,
    breakdown,
  };
}

// ─── Mixed-mode grading ────────────────────────────────────────────────────────

/**
 * Grades a mixed-type exam where each question may have a different ExamType.
 *
 * @param studentAnswers  The student's answers (question number → answer string)
 * @param answerKey       The correct answers (question number → answer string)
 * @param questionTypeMap Per-question exam type map (question number → ExamType)
 * @param fallbackType    Used for any question not in questionTypeMap
 * @param requireUppercase When true, lowercase answers are marked wrong with
 *   uppercaseViolation:true. Does NOT affect bubble_omr questions. Defaults to false.
 */
export function gradeAnswersMixed(
  studentAnswers:  Record<string, string>,
  answerKey:       FlatAnswerKey,
  questionTypeMap: Record<string, ExamType | string>, // ✅ accepts auto-detected string maps too
  fallbackType:    ExamType = 'multiple_choice',
  requireUppercase = false,
): GradeResult {
  const breakdown: GradeBreakdown[] = [];
  let score      = 0;
  let unanswered = 0;

  const total = Object.keys(answerKey).length;

  for (const q of Object.keys(answerKey)) {
    const examType   = questionTypeMap[q] ?? fallbackType;
    const normType   = normTypeFor(examType);
    const correctRaw = (answerKey[q] ?? '').trim();
    const studentRaw = (studentAnswers[q] ?? '').trim();

    const isUnanswered = studentRaw === '';
    if (isUnanswered) unanswered++;

    let isCorrect          = false;
    let uppercaseViolation = false;
    let studentNorm        = '';
    let correctNorm        = '';

    if (!isUnanswered) {
      // ── ALL CAPS check ──────────────────────────────────────────────────────
      if (isUppercaseViolation(examType, studentRaw, requireUppercase)) {
        uppercaseViolation = true;
        isCorrect          = false;
        studentNorm = examType === 'enumeration'
          ? normalizeEnumerationSide(studentRaw)
          : normalizeByType(studentRaw, normType);
        correctNorm = examType === 'enumeration'
          ? normalizeEnumerationSide(correctRaw)
          : normalizeByType(correctRaw, normType);
      } else {
        // ── Normal content matching ──────────────────────────────────────────
        if (examType === 'enumeration') {
          isCorrect   = matchEnumeration(studentRaw, correctRaw);
          studentNorm = normalizeIdentification(studentRaw);
          correctNorm = normalizeEnumerationSide(correctRaw);
        } else {
          studentNorm = normalizeByType(studentRaw, normType);
          correctNorm = normalizeByType(correctRaw, normType);
          // Primary match: standard normalisation
          isCorrect = correctNorm !== '' && studentNorm === correctNorm;
          // Compact fallback for identification: "My SQL"=="MySQL", "Node.js"=="Nodejs"
          if (!isCorrect && normType === 'identification') {
            const sc = normalizeCompact(studentRaw);
            const cc = normalizeCompact(correctRaw);
            isCorrect = sc !== '' && cc !== '' && sc === cc;
          }
        }
      }
    }

    if (isCorrect) score++;

    const entry: GradeBreakdown = {
      questionNumber:    q,
      correct:           isCorrect,
      studentAnswer:     studentRaw || '—',
      correctAnswer:     correctRaw,
      normalizedStudent: studentNorm || '—',
      normalizedCorrect: correctNorm,
    };
    if (uppercaseViolation) entry.uppercaseViolation = true;
    breakdown.push(entry);
  }

  const wrong      = Math.max(total - score - unanswered, 0);
  const percentage = total > 0 ? Math.round((score / total) * 100) : 0;

  return {
    score,
    total,
    wrong,
    unanswered,
    percentage,
    passed: percentage >= 75,
    breakdown,
  };
}

// ─── Answer Key Validation ─────────────────────────────────────────────────────

export function validateAnswerKey(
  key:      FlatAnswerKey,
  examType: ExamType,
): { valid: boolean; error?: string } {
  const { getTemplate } = require('../types/exam');
  const template = getTemplate(examType);
  const entries  = Object.entries(key);

  if (entries.length === 0) {
    return { valid: false, error: 'Answer key is empty.' };
  }

  for (const [q, a] of entries) {
    if (!a?.trim()) {
      return { valid: false, error: `Question ${q} has no answer.` };
    }
    const err = template.validateAnswer(a.trim(), Number(q));
    if (err) return { valid: false, error: err };
  }

  return { valid: true };
}

// ─── Answer Key Text Parser ────────────────────────────────────────────────────

/**
 * Parses a plain-text answer key into a FlatAnswerKey.
 * Accepts lines like:
 *   "1. A"  "1) B"  "1: C"  "1 D"
 */
export function parseAnswerKeyText(text: string): FlatAnswerKey {
  const key: FlatAnswerKey = {};

  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    const match = trimmed.match(/^(\d+)[.):\s]\s*(.+)/);
    if (match) {
      key[match[1]] = match[2].trim();
    }
  }

  return key;
}

// ─── Result Builder ────────────────────────────────────────────────────────────

/**
 * Builds a NewScanResult (no id / scannedAt) from raw scan data.
 * Embeds gradeResult so downstream consumers (ResultScreen, etc.) never
 * need to re-grade.
 *
 * Pass questionTypeMap for mixed-type exams so each question is graded
 * with the correct normalisation rules for its section type.
 *
 * @param requireUppercase  Teacher-controlled ALL CAPS toggle. When true,
 *   student answers with any lowercase letter are marked wrong and stored
 *   with uppercaseViolation:true in the grade breakdown. Defaults to false.
 *   The value is persisted on the NewScanResult so ReviewScreen and future
 *   re-grades respect the original teacher setting.
 */
export function buildScanResult(
  studentName:         string,
  sectionId:           string | null,
  examType:            ExamType,
  studentAnswers:      FlatAnswerKey,
  answerKey:           FlatAnswerKey,
  ocrConfidence:       number,
  ocrNotes:            string,
  detectedStudentName: string | null = null,
  questionTypeMap?:    Record<string, ExamType>,
  requireUppercase     = false,
): NewScanResult {
  const gradeResult = questionTypeMap && Object.keys(questionTypeMap).length > 0
    ? gradeAnswersMixed(studentAnswers, answerKey, questionTypeMap, examType, requireUppercase)
    : gradeAnswers(studentAnswers, answerKey, examType, requireUppercase);

  return {
    studentName:         studentName.trim(),
    sectionId,
    examType,
    studentAnswers,
    answerKey,
    score:               gradeResult.score,
    total:               gradeResult.total,
    percentage:          gradeResult.percentage,
    passed:              gradeResult.passed,
    ocrConfidence,
    ocrNotes,
    gradeResult,
    detectedStudentName,
    requireUppercase,
    // FIX: store questionTypeMap in result so ReviewScreen can re-grade
    // each question with the correct type when the user edits an answer.
    ...(questionTypeMap && Object.keys(questionTypeMap).length > 0 ? { questionTypeMap } : {}),
  };
}