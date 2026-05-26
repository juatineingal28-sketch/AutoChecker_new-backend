// src/utils/parseAnswerKey.ts
// ─────────────────────────────────────────────────────────────────────────────
// Teacher-Friendly Answer Key Parser
//
// Accepts human-readable answer key formats (section headers, numbered lines,
// unnumbered lines, comma-grouped enumeration) AND the legacy strict format
// (1.A, 2.B, 3.C) for full backward compatibility.
//
// OUTPUT: FlatAnswerKey — Record<string, string> — consumed by grading.ts
//
// SECTION DETECTION (case-insensitive):
//   Multiple Choice | Bubble OMR | Identification | Enumeration |
//   True or False   | True/False
//
// PARSER PRIORITY:
//   1. Teacher-friendly format (section headers present)
//   2. Legacy strict format fallback (1.A, 2.B, 3.C)
// ─────────────────────────────────────────────────────────────────────────────

import type { ExamType, FlatAnswerKey } from '../types/exam';

// ─── Public Types ─────────────────────────────────────────────────────────────

export type SectionType = ExamType;

export interface ParseError {
  code:
    | 'EMPTY_SECTION'
    | 'INVALID_LETTER'
    | 'DUPLICATE_NUMBER'
    | 'UNSUPPORTED_SYMBOL'
    | 'MIXED_SECTION'
    | 'EMPTY_INPUT'
    | 'UNKNOWN';
  message: string;
  section?: SectionType;
  lineNumber?: number;
  raw?: string;
}

export interface SectionBlock {
  sectionType: SectionType;
  /** Global item start index (1-based) for this section in the flat key */
  startIndex:  number;
  answers:     string[];
}

export interface ParsedAnswerKey {
  sections: SectionBlock[];
  flat:     FlatAnswerKey;
}

export interface ParseResult {
  success:  boolean;
  data?:    ParsedAnswerKey;
  errors:   ParseError[];
  warnings: string[];
}

// ─── Section Header Registry ──────────────────────────────────────────────────

interface SectionDef {
  type:     SectionType;
  patterns: RegExp[];
}

const SECTION_DEFS: SectionDef[] = [
  {
    type:     'bubble_omr',
    patterns: [/bubble\s*omr/i, /omr\s*bubble/i, /bubble/i],
  },
  {
    type:     'multiple_choice',
    patterns: [/multiple\s*choice/i, /mult\.?\s*choice/i, /m\.?c\.?/i],
  },
  {
    type:     'identification',
    patterns: [/identification/i, /identify/i],
  },
  {
    type:     'enumeration',
    patterns: [/enumeration/i, /enumerate/i],
  },
  {
    type:     'true_or_false',
    patterns: [/true\s*(or|\/)\s*false/i, /t\s*[\/or]+\s*f\b/i],
  },
];

/** Returns the SectionType if the line is a section header, else null. */
function detectSectionHeader(line: string): SectionType | null {
  // Strip trailing colon, punctuation, whitespace
  let cleaned = line.trim().replace(/[:.]+$/, '').trim();

  // Strip trailing "ANSWERKEY", "ANSWERKEYS", "ANSWER KEY", "ANSWER KEYS"
  cleaned = cleaned.replace(/\s*answer\s*keys?$/i, '').trim();

  // Skip if the line looks like a numbered answer (e.g. "1. True")
  if (/^\d+[.):\s]/.test(cleaned)) return null;

  // Skip if too long to be a header (likely a sentence or an answer)
  if (cleaned.length > 60) return null;

  for (const def of SECTION_DEFS) {
    for (const pattern of def.patterns) {
      if (pattern.test(cleaned)) return def.type;
    }
  }
  return null;
}

/**
 * Strips DOCX list formatting from a line.
 * Handles: "- B", "* B", "• B", "– B", "— B"
 * Returns the clean content after the bullet/dash.
 */
function stripBulletPrefix(line: string): string {
  return line.replace(/^[\t ]*[-–—*•]\s+/, '').trim();
}

// ─── Per-Section Answer Normalizers ──────────────────────────────────────────

const MCQ_VALID = new Set(['A', 'B', 'C', 'D']);

function normalizeMCQ(
  raw:     string,
  lineNum: number,
  errors:  ParseError[],
  section: SectionType,
): string | null {
  const v = raw.trim().toUpperCase();
  if (MCQ_VALID.has(v)) return v;

  // Try to extract a single valid letter (handles "A." or "(B)")
  const match = v.match(/\b([ABCD])\b/);
  if (match) return match[1];

  errors.push({
    code:       'INVALID_LETTER',
    message:    `Invalid answer "${raw}" — only A, B, C, D are allowed.`,
    section,
    lineNumber: lineNum,
    raw,
  });
  return null;
}

function normalizeIdentification(raw: string): string {
  return raw.trim().replace(/\s+/g, ' ');
}

function normalizeEnumeration(raw: string): string {
  // Preserve semicolon-separated synonyms: "HTML;HyperText Markup Language"
  return raw
    .split(';')
    .map(s => s.trim())
    .filter(Boolean)
    .join(';');
}

const TF_MAP: Record<string, string> = {
  true:  'True', t: 'True',
  false: 'False', f: 'False',
};

function normalizeTrueFalse(
  raw:     string,
  lineNum: number,
  errors:  ParseError[],
  section: SectionType,
): string | null {
  const v = raw.trim().toLowerCase();
  if (TF_MAP[v]) return TF_MAP[v];

  errors.push({
    code:       'INVALID_LETTER',
    message:    `Invalid True/False answer "${raw}" — use True, False, T, or F.`,
    section,
    lineNumber: lineNum,
    raw,
  });
  return null;
}

function normalizeAnswer(
  raw:     string,
  type:    SectionType,
  lineNum: number,
  errors:  ParseError[],
): string | null {
  switch (type) {
    case 'bubble_omr':
    case 'multiple_choice':
      return normalizeMCQ(raw, lineNum, errors, type);
    case 'identification':
      return normalizeIdentification(raw) || null;
    case 'enumeration':
      return normalizeEnumeration(raw) || null;
    case 'true_or_false':
      return normalizeTrueFalse(raw, lineNum, errors, type);
    default:
      return null;
  }
}

// ─── Line Parser (single section block) ──────────────────────────────────────

interface RawLine {
  content: string;
  lineNum: number;
}

/**
 * Strips a leading number from a line and returns the answer part.
 * Handles: "1. B", "1) B", "1: B", "1 B"
 * Returns null if the line has no leading number.
 */
function stripLeadingNumber(line: string): { num: number; answer: string } | null {
  const match = line.match(/^(\d+)\s*[.):\s]\s*(.+)/);
  if (!match) return null;
  return { num: parseInt(match[1], 10), answer: match[2].trim() };
}

/**
 * Parses lines belonging to one section into an ordered answers array.
 * Supports:
 *   - Numbered:   "1. B", "2. CSS"
 *   - Unnumbered: "B", "CSS" (auto-numbered from 1)
 *   - Comma-grouped (enumeration only): "HTML, CSS, JavaScript"
 */
function parseSectionLines(
  lines:    RawLine[],
  type:     SectionType,
  errors:   ParseError[],
  warnings: string[],
): string[] {
  const results: string[] = [];
  const seenNumbers = new Set<number>();

  // For enumeration: also support comma-grouped lines
  const expandedLines: RawLine[] = [];

  if (type === 'enumeration') {
    for (const line of lines) {
      const stripped = stripLeadingNumber(line.content);

      if (stripped) {
        // Could be "1. HTML, CSS" (numbered + comma list) or just "1. HTML"
        // If the answer part contains commas and no semicolons → expand
        const ans = stripped.answer;
        if (ans.includes(',') && !ans.includes(';')) {
          const items = ans.split(',').map(s => s.trim()).filter(Boolean);
          if (items.length > 1) {
            // Expand into multiple unnumbered lines following this number
            for (const item of items) {
              expandedLines.push({ content: item, lineNum: line.lineNum });
            }
            continue;
          }
        }
      } else {
        // Unnumbered: might be comma-grouped "HTML, CSS, JavaScript"
        const content = line.content.trim();
        if (content.includes(',') && !content.includes(';')) {
          const items = content.split(',').map(s => s.trim()).filter(Boolean);
          if (items.length > 1) {
            for (const item of items) {
              expandedLines.push({ content: item, lineNum: line.lineNum });
            }
            continue;
          }
        }
      }

      expandedLines.push(line);
    }
  } else {
    expandedLines.push(...lines);
  }

  // Now process expanded lines
  let autoCounter = 1;
  let hasNumbered   = false;
  let hasUnnumbered = false;

  for (const line of expandedLines) {
    const stripped = stripLeadingNumber(line.content);

    if (stripped) {
      hasNumbered = true;

      if (hasUnnumbered) {
        warnings.push(
          `Line ${line.lineNum}: Mixed numbered/unnumbered lines detected. ` +
          `Treat all as numbered from this point.`,
        );
      }

      const { num, answer } = stripped;

      // Duplicate number check
      if (seenNumbers.has(num)) {
        errors.push({
          code:       'DUPLICATE_NUMBER',
          message:    `Duplicate question number ${num} in section.`,
          section:    type,
          lineNumber: line.lineNum,
          raw:        line.content,
        });
        continue;
      }

      seenNumbers.add(num);
      autoCounter = num + 1;

      const normalized = normalizeAnswer(answer, type, line.lineNum, errors);
      if (normalized !== null) {
        // Ensure array is big enough (handles non-sequential numbers)
        while (results.length < num - 1) results.push('');
        results[num - 1] = normalized;
      }
    } else {
      // Unnumbered line — auto-number
      hasUnnumbered = true;

      const content = line.content.trim();
      const normalized = normalizeAnswer(content, type, line.lineNum, errors);

      if (normalized !== null) {
        results.push(normalized);
        seenNumbers.add(autoCounter);
        autoCounter++;
      }
    }
  }

  return results;
}

// ─── Teacher-Friendly Format Parser ──────────────────────────────────────────

/**
 * Detects whether the text contains at least one known section header.
 */
function hasTeacherFormat(lines: string[]): boolean {
  return lines.some(l => detectSectionHeader(l) !== null);
}

/**
 * Splits raw text into section blocks and parses each block.
 */
function parseTeacherFormat(
  rawLines:  string[],
  errors:    ParseError[],
  warnings:  string[],
): SectionBlock[] {
  const blocks:   SectionBlock[] = [];
  let currentType: SectionType | null = null;
  let currentRaw:  RawLine[] = [];
  let globalIndex  = 1;
  let lineNum      = 0;

  const flushBlock = () => {
    if (!currentType) return;

    const nonEmpty = currentRaw.filter(l => l.content.trim() !== '');

    if (nonEmpty.length === 0) {
      errors.push({
        code:    'EMPTY_SECTION',
        message: `Section "${currentType}" has no answers.`,
        section: currentType,
      });
    } else {
      const answers = parseSectionLines(nonEmpty, currentType, errors, warnings);
      if (answers.length > 0) {
        blocks.push({ sectionType: currentType, startIndex: globalIndex, answers });
        globalIndex += answers.length;
      }
    }

    currentRaw = [];
  };

  for (const rawLine of rawLines) {
    lineNum++;
    const trimmed = rawLine.trim();

    // Skip blank lines (but record them for section flushing)
    if (!trimmed) continue;

    const headerType = detectSectionHeader(trimmed);

    if (headerType !== null) {
      flushBlock();
      currentType = headerType;
    } else if (currentType !== null) {
      const content = stripBulletPrefix(trimmed); if (content) currentRaw.push({ content, lineNum });
    } else {
      // Content before any header — warn and skip
      warnings.push(`Line ${lineNum}: Content "${trimmed}" appears before any section header — skipped.`);
    }
  }

  flushBlock(); // flush last section

  return blocks;
}

// ─── Legacy Strict Format Parser ─────────────────────────────────────────────

/**
 * Parses the legacy format: "1.A, 2.B, 3.C" or "1.A 2.B 3.C"
 * Also handles multiline: each line being "1.A" style.
 */
function parseLegacyFormat(
  text:     string,
  errors:   ParseError[],
  warnings: string[],
): FlatAnswerKey {
  const flat: FlatAnswerKey = {};

  warnings.push('No section headers detected — using legacy strict format parser.');

  // Normalize: split on commas or newlines
  const tokens = text
    .replace(/\n/g, ',')
    .split(',')
    .map(t => t.trim())
    .filter(Boolean);

  const seenNumbers = new Set<string>();

  for (const token of tokens) {
    // Matches: "1.A", "1. A", "1)A", "1: A", "1 A" (number + separator + answer)
    const match = token.match(/^(\d+)\s*[.):\s]\s*(.+)/);
    if (!match) {
      errors.push({
        code:    'UNSUPPORTED_SYMBOL',
        message: `Cannot parse token "${token}" in legacy format.`,
        raw:     token,
      });
      continue;
    }

    const [, num, answer] = match;
    const cleanAnswer = answer.trim();

    if (seenNumbers.has(num)) {
      errors.push({
        code:    'DUPLICATE_NUMBER',
        message: `Duplicate question number ${num} in answer key.`,
        raw:     token,
      });
      continue;
    }

    seenNumbers.add(num);
    flat[num] = cleanAnswer;
  }

  return flat;
}

// ─── FlatAnswerKey Builder ────────────────────────────────────────────────────

function buildFlatFromSections(sections: SectionBlock[]): FlatAnswerKey {
  const flat: FlatAnswerKey = {};
  let globalIndex = 1;

  for (const section of sections) {
    for (const answer of section.answers) {
      flat[String(globalIndex)] = answer;
      globalIndex++;
    }
  }

  return flat;
}

// ─── Main Entry Point ─────────────────────────────────────────────────────────

/**
 * Parses a teacher-supplied answer key text into a FlatAnswerKey.
 *
 * Accepts:
 *   - Teacher-friendly format with section headers (Multiple Choice, Identification, etc.)
 *   - Legacy strict format: 1.A, 2.B, 3.C
 *
 * @param text  Raw text from pasted input, .txt, or extracted .docx content
 * @returns     ParseResult with FlatAnswerKey and any errors/warnings
 */
export function parseAnswerKey(text: string): ParseResult {
  const errors:   ParseError[] = [];
  const warnings: string[]     = [];

  // ── Guard: empty input ────────────────────────────────────────────────────
  const trimmedText = (text ?? '').trim();
  if (!trimmedText) {
    return {
      success: false,
      errors: [{
        code:    'EMPTY_INPUT',
        message: 'Answer key text is empty.',
      }],
      warnings,
    };
  }

  const rawLines = trimmedText.split('\n');

  // ── Route: teacher format vs legacy ──────────────────────────────────────
  if (hasTeacherFormat(rawLines)) {
    // Teacher-friendly path
    const sections = parseTeacherFormat(rawLines, errors, warnings);

    if (sections.length === 0) {
      errors.push({
        code:    'EMPTY_INPUT',
        message: 'No valid answers could be parsed from any section.',
      });
    }

    if (errors.some(e =>
      e.code === 'EMPTY_SECTION' ||
      e.code === 'INVALID_LETTER' ||
      e.code === 'DUPLICATE_NUMBER',
    )) {
      return { success: false, errors, warnings };
    }

    const flat = buildFlatFromSections(sections);

    if (Object.keys(flat).length === 0) {
      return {
        success: false,
        errors: [...errors, { code: 'EMPTY_INPUT', message: 'Answer key produced no entries.' }],
        warnings,
      };
    }

    return {
      success: true,
      data:    { sections, flat },
      errors,
      warnings,
    };
  }

  // Legacy fallback
  const flat = parseLegacyFormat(trimmedText, errors, warnings);

  const hasBlockingErrors = errors.some(e =>
    e.code === 'DUPLICATE_NUMBER' ||
    e.code === 'EMPTY_INPUT',
  );

  if (hasBlockingErrors || Object.keys(flat).length === 0) {
    if (Object.keys(flat).length === 0) {
      errors.push({ code: 'EMPTY_INPUT', message: 'No valid answers found in answer key.' });
    }
    return { success: false, errors, warnings };
  }

  return {
    success: true,
    data:    { sections: [], flat },
    errors,
    warnings,
  };
}

// ─── Convenience: parse + extract flat key ────────────────────────────────────

/**
 * Parses text and returns only the FlatAnswerKey.
 * Throws if parsing fails. Use `parseAnswerKey()` for full error handling.
 */
export function parseFlatAnswerKey(text: string): FlatAnswerKey {
  const result = parseAnswerKey(text);
  if (!result.success || !result.data) {
    const msg = result.errors.map(e => e.message).join('; ');
    throw new Error(`[parseAnswerKey] Parse failed: ${msg}`);
  }
  return result.data.flat;
}

// ─── Backward-compat re-export (used by grading.ts) ─────────────────────────

/**
 * Drop-in replacement for the old `parseAnswerKeyText` in grading.ts.
 * Wraps the new parser with silent fallback — returns empty key on failure.
 */
export function parseAnswerKeyText(text: string): FlatAnswerKey {
  try {
    const result = parseAnswerKey(text);
    return result.data?.flat ?? {};
  } catch {
    return {};
  }
}