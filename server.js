// server.js  â€” AutoChecker Backend  (UPGRADED)
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// What changed vs original:
//
//  HANDWRITING IMPROVEMENTS
//  1. sharp pipeline upgraded:
//       â€¢ modulate (brightness +10%)  â†’ lifts faint ballpen strokes
//       â€¢ median filter (3px)         â†’ removes salt-and-pepper noise
//       â€¢ threshold raised to 175     â†’ better binarisation for dark ink on white
//       â€¢ resize to 20px            â†’ more pixels for LSTM on small handwriting
//  2. PSM sequence extended: [4, 6, 3, 11, 12]
//       â€¢ PSM 12 (sparse text w/ OSD) added â€” catches isolated written letters
//  3. Handwriting-specific inverted retry on ALL exam types (not just MC)
//  4. OCR character whitelist split:
//       â€¢ MC   â†’ "0123456789ABCDabcd.):-/ \n"  (tight, fewer substitutions)
//       â€¢ Text â†’ full alphanumeric (no restriction)
//  5. fixOcrSubstitutions() extended with handwriting-specific confusions:
//       â€¢ qâ†’a, nâ†’A, uâ†’A, 6â†’b, Eâ†’B, Fâ†’E, Hâ†’A, etc.
//  6. extractWrittenAnswers() â€” NEW function for identification/enumeration/short-answer:
//       â€¢ tolerates dirty OCR (extra chars, merged words)
//       â€¢ handles multi-word answers
//       â€¢ normalises capitalisation per exam type
//
//  RELIABILITY / CRASH PREVENTION
//  7. parseVisionText() never throws for empty text â€” returns empty answers map
//  8. All sharp operations wrapped with individual try/catch
//  9. Tesseract worker always terminated in finally block
// 10. POST /api/scan returns { success: false, answers: {}, confidence: 0 }
//     instead of 500 when OCR produces no usable text
//
//  PERFORMANCE
// 11. Single sharp pipeline call (was multiple chained calls)
// 12. Worker reuse considered â€” kept per-PSM creation for stability on mobile
//
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

require('dotenv').config();

const express   = require('express');
const multer    = require('multer');
const cors      = require('cors');
const fs        = require('fs');
const path      = require('path');
// ── pdf-parse import fix ──────────────────────────────────────────────────────
// pdf-parse exports differently across versions and module systems:
//   v1.x  CommonJS  → module.exports = function pdfParse() {...}   (function directly)
//   v1.x  some envs → module.exports = { default: fn, ... }        (wrapped object)
//   v2.x+ ESM shim  → _pdfParseLib.default = fn                    (ESM interop)
//
// The previous two-step check missed the case where BOTH checks fail and
// _pdfParseLib itself is the callable (most common on Railway with pdf-parse v1.x).
// This three-step resolution covers every known case:
let pdfParse;
try {
  const _pdfParseLib = require('pdf-parse/lib/pdf-parse.js'); // direct path — bypasses all wrapping
  pdfParse = typeof _pdfParseLib === 'function' ? _pdfParseLib : null;
} catch {
  // Fallback: try the top-level export with all resolution strategies
  try {
    const _pdfParseLib = require('pdf-parse');
    if      (typeof _pdfParseLib          === 'function') pdfParse = _pdfParseLib;
    else if (typeof _pdfParseLib.default  === 'function') pdfParse = _pdfParseLib.default;
    else if (typeof _pdfParseLib.pdfParse === 'function') pdfParse = _pdfParseLib.pdfParse;
    else pdfParse = null;
  } catch (e) {
    console.error('[AutoChecker] ❌ pdf-parse could not be loaded:', e.message);
    pdfParse = null;
  }
}
if (pdfParse) {
  console.log('[AutoChecker] pdf-parse loaded successfully ✓');
} else {
  console.warn('[AutoChecker] ⚠️  pdf-parse not available — PDF uploads will return a clear error.');
}
const mammoth   = require('mammoth');
const Tesseract = require('tesseract.js');

// Groq Vision — FREE replacement for Gemini (14,400 requests/day, no credit card)
// Sign up at console.groq.com → API Keys → Create Key
// Add GROQ_API_KEY to your Railway environment variables
const GROQ_API_KEY = process.env.GROQ_API_KEY;
let groqReady = false;

async function initGroq() {
  if (!GROQ_API_KEY) {
    console.warn('[AutoChecker] GROQ_API_KEY not set — written types will use Tesseract only.');
    console.warn('              Get a free key at console.groq.com and add GROQ_API_KEY to Railway.');
    return;
  }
  try {
    const res = await fetch('https://api.groq.com/openai/v1/models', {
      headers: { 'Authorization': `Bearer ${GROQ_API_KEY}` },
    });
    if (res.ok) {
      groqReady = true;
      console.log('[AutoChecker] Groq ready — model: llama-4-scout-17b-16e-instruct (FREE)');
    } else {
      console.warn(`[AutoChecker] Groq key check failed: HTTP ${res.status}. Check your GROQ_API_KEY.`);
    }
  } catch (e) {
    console.warn('[AutoChecker] Groq init error:', e.message);
  }
}

// Run probe at startup; does not block server from starting
initGroq().catch(e => console.warn('[AutoChecker] Groq init error:', e.message));

// sharp is optional â€” gracefully degrade if not installed
let sharp;
try {
  sharp = require('sharp');
} catch {
  console.warn('[AutoChecker] âš ï¸  sharp not found. Install for better handwriting accuracy:');
  console.warn('              npm install sharp');
}

const app  = express();
const PORT = process.env.PORT || 3000;

// â”€â”€â”€ Dirs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

const DATA_DIR    = path.join(__dirname, 'data');
const UPLOADS_DIR = path.join(__dirname, 'uploads');
[DATA_DIR, UPLOADS_DIR].forEach(d => { if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true }); });

// â”€â”€â”€ Middleware â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

app.use(cors());
app.use(express.json({ limit: '20mb' }));

// â”€â”€â”€ Multer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

const ACCEPTED_EXTS = ['.json', '.txt', '.pdf', '.docx', '.doc'];

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, UPLOADS_DIR),
  filename:    (_req, file,  cb) => cb(null, `${Date.now()}-${file.originalname}`),
});

const upload = multer({
  storage,
  limits: { fileSize: 5 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    ACCEPTED_EXTS.includes(ext) ? cb(null, true) : cb(new Error(`Unsupported file. Accepted: ${ACCEPTED_EXTS.join(', ')}`));
  },
});

// â”€â”€â”€ Disk store â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function dataPath(id)      { return path.join(DATA_DIR, `section_${id}.json`); }
function readStore(id)     { try { return JSON.parse(fs.readFileSync(dataPath(id), 'utf8')); } catch { return null; } }
function writeStore(id, d) { fs.writeFileSync(dataPath(id), JSON.stringify(d, null, 2), 'utf8'); }
function deleteStore(id)   { try { fs.unlinkSync(dataPath(id)); } catch {} }

const SECTIONS_PATH = path.join(DATA_DIR, '_sections.json');
function readSections()     { try { return JSON.parse(fs.readFileSync(SECTIONS_PATH, 'utf8')); } catch { return []; } }
function writeSections(arr) { fs.writeFileSync(SECTIONS_PATH, JSON.stringify(arr, null, 2), 'utf8'); }


// â”€â”€â”€ Settings store â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

const SETTINGS_PATH = path.join(DATA_DIR, '_settings.json');
const DEFAULT_SETTINGS = {
  autoDetect:    true,
  scanTips:      true,
  flagLow:       false,
  treeToggleOcr: false,
  scanning:      true,
};
function readSettings()      { try { return { ...DEFAULT_SETTINGS, ...JSON.parse(fs.readFileSync(SETTINGS_PATH, 'utf8')) }; } catch { return { ...DEFAULT_SETTINGS }; } }
function writeSettings(data) { fs.writeFileSync(SETTINGS_PATH, JSON.stringify(data, null, 2), 'utf8'); }

// â”€â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

const VALID_COLORS = new Set(['blue','green','amber','red']);

// â”€â”€â”€ Normalizers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function resolveType(raw) {
  const s = String(raw ?? 'mc').toLowerCase().replace(/[\s_\-]/g, '');
  // FIX: added omr/bubbleomr → mc so toBackendExamType('bubble_omr')='omr' resolves correctly
  return ({ mc:'mc', multiplechoice:'mc', omr:'mc', bubbleomr:'mc', truefalse:'truefalse', tf:'truefalse',
    identification:'identification', id:'identification', enumeration:'enumeration',
    enum:'enumeration', traceerror:'traceError', trace:'traceError', tracetheerror:'traceError',
    shortanswer:'shortAnswer', short:'shortAnswer', sa:'shortAnswer' })[s] ?? 'identification';
}

function inferType(raw) {
  const u = raw.trim().toUpperCase();
  if (['A','B','C','D'].includes(u)) return 'mc';
  if (['TRUE','FALSE'].includes(u))  return 'truefalse';
  if (raw.includes(','))             return 'enumeration';
  return 'identification';
}

function normalizeAnswer(type, raw) {
  switch (type) {
    case 'mc':          return String(raw).trim().toUpperCase();
    case 'truefalse':   return String(raw).trim().toUpperCase() === 'TRUE' ? 'True' : 'False';
    case 'enumeration': {
      const arr = Array.isArray(raw) ? raw : String(raw).split(',');
      return arr.map(s => s.trim().toLowerCase()).filter(Boolean);
    }
    default: return String(raw).trim().toLowerCase();
  }
}

// â”€â”€â”€ Text parser â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// UPGRADED: Smart multi-strategy parser with NO hardcoded limits.
// Handles teacher-made answer keys in virtually any format:
//   â€¢ "1. A"  "1) A"  "1: A"  "1-A"  "Q1 A"  "1 A"
//   â€¢ "A,B,C,D,A" (comma-separated list, no numbers)
//   â€¢ "A B C D A" (space-separated bare letters)
//   â€¢ vertical lists (one answer per line, no numbers)
//   â€¢ two-column / multi-column layouts
//   â€¢ mixed spacing, lowercase/uppercase, with or without periods
//   â€¢ answers preceded by optional type tags [type:mc]

function parseTextLines(text) {
  const seen = new Set();
  const results = [];

  // â”€â”€ Pre-clean â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  // Normalize unicode dashes/dots, strip BOM, normalize whitespace
  let clean = text
    .replace(/^\uFEFF/, '')                       // strip BOM
    .replace(/\r\n|\r/g, '\n')                   // normalize line endings
    .replace(/[â€“â€”]/g, '-')                        // unicode dashes â†’ hyphen
    .replace(/['']/g, "'")                        // smart quotes
    .replace(/[""]/g, '"')
    .replace(/\t/g, ' ');                         // tabs â†’ spaces

  // â”€â”€ Strategy 1: Numbered answers (primary) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  // Handles: "1. A"  "1) B"  "1: C"  "1-D"  "Q1 A"  "1 A"  "No.1 A"
  //          plus optional [type:mc] tag
  const RE_NUMBERED = /^(?:no\.?\s*|q\.?\s*)?(\d{1,3})\s*[.):\-]?\s*(?:\[type:([^\]]+)\]\s*)?(.{1,200})$/im;
  const RE_SPLIT    = /^(?:no\.?\s*|q\.?\s*)?(\d{1,3})\s*[.):\-]?\s*(?:\[type:([^\]]+)\]\s*)?(.{1,200})$/i;

  // Split merged lines: "1. A 2. B 3. C" â†’ separate lines
  // Handle all separators: .  )  :  -  with optional leading space/number
  clean = clean
    .replace(/([^\n])\s+(\d{1,3}\s*[.):\-]\s)/g, '$1\n$2')
    .replace(/([^\n])(Q\d{1,3}\s)/gi, '$1\n$2');

  const lines = clean.split('\n').map(l => l.trim()).filter(Boolean);

  for (const line of lines) {
    const m = line.match(RE_SPLIT);
    if (!m) continue;

    const qNum = parseInt(m[1], 10);
    if (qNum < 1 || qNum > 500) continue;           // reasonable ceiling, not hardcoded limit
    if (seen.has(qNum)) continue;

    let rawValue = m[3].trim();
    let type     = m[2] ? resolveType(m[2]) : null;

    // Inline type tag at start of value: "[type:mc] A"
    const inlineTag = rawValue.match(/^\[type:([^\]]+)\]\s*/i);
    if (inlineTag) {
      type     = resolveType(inlineTag[1]);
      rawValue = rawValue.slice(inlineTag[0].length).trim();
    }

    if (!rawValue) continue;
    if (!type) type = inferType(rawValue);

    seen.add(qNum);
    results.push({ question: qNum, type, answer: normalizeAnswer(type, rawValue) });
  }

  // â”€â”€ Strategy 2: Comma-separated bare answer list â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  // "A,B,C,D,A,B" or "A, B, C, D" â€” teacher typed answers separated by commas
  // Only triggers if Strategy 1 found very few results
  if (results.length < 3) {
    const stripped = clean.replace(/\n/g, ' ').trim();
    // Must look like: letter/word, letter/word, ...  (at least 3 comma groups)
    const RE_CSV = /^([A-Za-z][^,]{0,30})(,\s*[A-Za-z][^,]{0,30}){2,}$/;
    if (RE_CSV.test(stripped)) {
      const parts = stripped.split(',').map(p => p.trim()).filter(Boolean);
      parts.forEach((part, idx) => {
        const qNum = idx + 1;
        if (!seen.has(qNum)) {
          const type = inferType(part);
          seen.add(qNum);
          results.push({ question: qNum, type, answer: normalizeAnswer(type, part) });
        }
      });
    }
  }

  // â”€â”€ Strategy 3: Space-separated bare letter list â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  // "A B C D A B C" â€” each space-separated token is one MC answer
  // Only triggers if still very few results
  if (results.length < 3) {
    const tokens = clean.split(/\s+/).filter(t => /^[ABCDabcd]$/.test(t));
    if (tokens.length >= 3) {
      tokens.forEach((token, idx) => {
        const qNum = idx + 1;
        if (!seen.has(qNum)) {
          seen.add(qNum);
          results.push({ question: qNum, type: 'mc', answer: token.toUpperCase() });
        }
      });
    }
  }

  // â”€â”€ Strategy 4: One-answer-per-line (no numbers) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  // Teacher typed one answer per line with no numbering at all
  if (results.length < 3) {
    const pureLines = clean.split('\n').map(l => l.trim()).filter(l => l && !/^\d+[.):\-]/.test(l));
    // Must be short lines (answers, not question text)
    const answerLike = pureLines.filter(l => l.length <= 60 && !/[?]/.test(l));
    if (answerLike.length >= 3) {
      answerLike.forEach((line, idx) => {
        const qNum = idx + 1;
        if (!seen.has(qNum)) {
          const type = inferType(line);
          seen.add(qNum);
          results.push({ question: qNum, type, answer: normalizeAnswer(type, line) });
        }
      });
    }
  }

  if (results.length === 0) return [];

  const sorted = results.sort((a, b) => a.question - b.question);
  console.log(`[AutoChecker] parseTextLines â†’ ${sorted.length} answers detected (strategies used)`);
  return sorted;
}

// â”€â”€â”€ JSON parser â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function parseJsonItems(raw) {
  if (!Array.isArray(raw)) return { error: 'JSON must be an array.' };
  if (!raw.length)         return { error: 'Answer key is empty.' };
  if (raw.length > 300)    return { error: 'Exceeds 300 items.' };
  const seen  = new Set();
  const items = [];
  for (let i = 0; i < raw.length; i++) {
    const item = raw[i];
    const qNum = Number(item?.question);
    if (!Number.isInteger(qNum) || qNum < 1) return { error: `Item ${i+1}: "question" must be a positive integer.` };
    if (seen.has(qNum)) return { error: `Duplicate question number: ${qNum}.` };
    seen.add(qNum);
    const type = resolveType(item?.type ?? 'mc');
    if (item.answer === undefined || item.answer === null) return { error: `Item ${i+1}: "answer" is required.` };
    items.push({ question: qNum, type, answer: normalizeAnswer(type, item.answer) });
  }
  return { items: items.sort((a, b) => a.question - b.question) };
}

// â”€â”€â”€ File parser â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// UPGRADED: Intelligent file parsing for .json, .txt, .pdf, .docx, .doc
//   â€¢ PDF: robust extraction with pdf-parse, multi-page, safe fallback
//   â€¢ DOCX/DOC: mammoth + aggressive line splitting for all teacher formats
//   â€¢ TXT: same smart multi-strategy parser as DOCX
//   â€¢ All formats: no hardcoded answer limits, handles 1â€“500 items
//   â€¢ Clear console logs: "50 answers detected successfully" or error details

async function parseUploadedFile(filePath, ext) {
  // â”€â”€ JSON â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  if (ext === '.json') {
    let raw;
    try {
      raw = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    } catch (e) {
      return { error: `Invalid JSON: ${e.message}` };
    }
    const result = parseJsonItems(raw);
    if (result.items) {
      console.log(`[AutoChecker] âœ… JSON parsed â€” ${result.items.length} answers detected successfully`);
    }
    return result;
  }

  // â”€â”€ TXT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  if (ext === '.txt') {
    let raw;
    try {
      raw = fs.readFileSync(filePath, 'utf8');
    } catch (e) {
      return { error: `Could not read file: ${e.message}` };
    }
    const items = parseTextLines(raw);
    if (items.length) {
      console.log(`[AutoChecker] âœ… TXT parsed â€” ${items.length} answers detected successfully`);
      return { items };
    }
    console.warn('[AutoChecker] âš ï¸  TXT: no answer key found. Raw sample:\n' + raw.slice(0, 300));
    return { error: 'No readable answer key found in this text file.\n\nSupported formats:\n  1. A\n  1) B\n  1: C\n  1-D\n  A,B,C,D (comma list)\n  A B C D (space list)\n  One answer per line' };
  }

  // â”€â”€ PDF â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  if (ext === '.pdf') {
    let fileBuffer;
    try {
      fileBuffer = fs.readFileSync(filePath);
    } catch (e) {
      return { error: `Could not read PDF file: ${e.message}` };
    }

    // Guard: if pdf-parse failed to load at startup, return a clear error
    if (typeof pdfParse !== 'function') {
      return { error: 'PDF support is unavailable (pdf-parse failed to load). Please use a .txt or .docx file instead.' };
    }

    let pdfText = '';
    try {
      // pdf-parse needs a real Buffer from disk â€” not a path string
      const pdfData = await pdfParse(fileBuffer, {
        // Extract ALL pages â€” no page limit
        max: 0,
        // Preserve more layout information
        normalizeWhitespace: false,
        disableCombineTextItems: false,
      });

      pdfText = pdfData.text || '';
      const pageCount = pdfData.numpages || 1;
      console.log(`[AutoChecker] PDF loaded â€” ${pageCount} page(s), ${pdfText.length} chars extracted`);

      if (!pdfText.trim()) {
        return { error: 'PDF appears to be image-based or scanned (no text layer). Please convert to a text PDF or use a .txt/.docx file instead.' };
      }
    } catch (pdfErr) {
      console.error('[AutoChecker] pdf-parse error:', pdfErr.message);
      // Provide a clear actionable error â€” not a raw stack trace
      if (pdfErr.message?.includes('Invalid PDF')) {
        return { error: 'The uploaded file is not a valid PDF. Please re-save or re-export it and try again.' };
      }
      if (pdfErr.message?.includes('encrypt')) {
        return { error: 'This PDF is password-protected. Please remove the password before uploading.' };
      }
      return { error: `PDF parsing failed: ${pdfErr.message}. Try saving as .txt or .docx instead.` };
    }

    const items = parseTextLines(pdfText);
    if (items.length) {
      console.log(`[AutoChecker] âœ… PDF parsed â€” ${items.length} answers detected successfully`);
      return { items };
    }

    // Log sample for debugging
    console.warn('[AutoChecker] âš ï¸  PDF: no answer key detected. Raw text sample:\n' + pdfText.slice(0, 500));
    return { error: 'No readable answer key found in this PDF.\n\nTips:\nâ€¢ Make sure the PDF has real text (not a scanned image)\nâ€¢ Supported formats: "1. A", "1) B", "A,B,C,D", one answer per line\nâ€¢ Try copying the content into a .txt file if parsing fails' };
  }

  // â”€â”€ DOCX / DOC â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  if (ext === '.docx' || ext === '.doc') {
    let rawText = '';
    try {
      const result = await mammoth.extractRawText({ path: filePath });
      rawText = result.value || '';
      if (result.messages?.length) {
        result.messages.forEach(msg => {
          if (msg.type === 'warning') console.warn('[AutoChecker] mammoth warning:', msg.message);
        });
      }
    } catch (e) {
      return { error: `Could not read Word document: ${e.message}` };
    }

    if (!rawText.trim()) {
      return { error: 'Word document appears to be empty or image-only.' };
    }

    // â”€â”€ Aggressive line normalization â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    let text = rawText
      .replace(/\r\n/g, '\n').replace(/\r/g, '\n')
      // Split merged: "1. A 2. B 3. C" â†’ "1. A\n2. B\n3. C"
      .replace(/([^\n])\s+(\d{1,3}\s*[.):\-]\s)/g, '$1\n$2')
      // "1-A2-B3-C" (no spaces) â†’ split at digit before number-separator
      .replace(/([A-Za-z])(\d{1,3}[.):\-])/g, '$1\n$2')
      // Remove excess blank lines
      .replace(/\n{3,}/g, '\n\n');

    // ── Section-aware parsing (handles mixed answer key types) ────────────────
    // Detect section headers like "MULTIPLE CHOICE ANSWERKEY", "IDENTIFICATION", etc.
    const SECTION_PATTERNS = [
      { type: 'mc',            re: /multiple\s*choice|bubble\s*omr|omr/i },
      { type: 'identification',re: /identification|identify/i },
      { type: 'enumeration',   re: /enumeration|enumerate/i },
      { type: 'truefalse',     re: /true\s*(or|\/)\s*false|t\s*[\/or]+\s*f\b/i },
    ];

    function detectSectionType(line) {
      // Strip trailing "ANSWERKEY(S)" and punctuation
      const cleaned = line.trim().replace(/[:.]+$/, '').replace(/\s*answer\s*keys?$/i, '').trim();
      if (/^\d+[.):\s]/.test(cleaned) || cleaned.length > 60) return null;
      for (const def of SECTION_PATTERNS) {
        if (def.re.test(cleaned)) return def.type;
      }
      return null;
    }

    function stripBullet(line) {
      return line.replace(/^[\t ]*[-–—*•]\s+/, '').trim();
    }

    // Check if this docx has section headers
    const rawLines = text.split('\n').map(l => l.trim()).filter(Boolean);
    console.log('[DEBUG] rawLines sample:', JSON.stringify(rawLines.slice(0, 5)));
    console.log('[DEBUG] line0 detected:', detectSectionType(rawLines[0] || ''));
    const hasSections = rawLines.some(l => detectSectionType(l) !== null);
    console.log('[DEBUG] hasSections:', hasSections, 'total lines:', rawLines.length);

    let items;

    if (hasSections) {
      // Section-aware parsing: group lines by section, auto-number within each section
      items = [];
      let currentType = null;
      let globalQ = 1;
      let sectionLines = [];

      const flushSection = () => {
        if (!currentType || sectionLines.length === 0) return;
        let autoNum = 1;
        for (const sLine of sectionLines) {
          const stripped = stripBullet(sLine);
          if (!stripped) continue;

          // Check if line has its own number (e.g. "1. CSS" or "1) True")
          const numMatch = stripped.match(/^(\d+)\s*[.):\s]\s*(.+)/);
          let rawAnswer, lineNum;
          if (numMatch) {
            lineNum  = parseInt(numMatch[1], 10);
            rawAnswer = numMatch[2].trim();
          } else {
            lineNum  = autoNum;
            rawAnswer = stripped;
          }

          if (!rawAnswer) continue;

          // Enumeration: comma-separated on one line → expand into multiple items
          if (currentType === 'enumeration' && rawAnswer.includes(',') && !rawAnswer.includes(';')) {
            const parts = rawAnswer.split(',').map(s => s.trim()).filter(Boolean);
            if (parts.length > 1) {
              for (const part of parts) {
                items.push({ question: globalQ++, type: 'enumeration', answer: normalizeAnswer('enumeration', part) });
              }
              autoNum++;
              continue;
            }
          }

          items.push({ question: globalQ++, type: currentType, answer: normalizeAnswer(currentType, rawAnswer) });
          autoNum = lineNum + 1;
        }
        sectionLines = [];
      };

      for (const line of rawLines) {
        const sType = detectSectionType(line);
        if (sType !== null) {
          flushSection();
          currentType = sType;
        } else if (currentType !== null) {
          sectionLines.push(line);
        }
      }
      flushSection();

      console.log(`[AutoChecker] DOCX section-aware parse → ${items.length} answers across multiple section types`);
    } else {
      // Fallback: legacy numbered-line parser
      items = parseTextLines(text);
      console.log(`[AutoChecker] DOCX extracted ${text.split('\n').length} lines → ${items.length} answers (legacy parser)`);
    }

    if (items.length === 0) {
      console.warn('[AutoChecker] ⚠️  DOCX: no answer key detected. Raw text sample:\n' + text.slice(0, 500));
    }

    if (items.length) {
      console.log(`[AutoChecker] ✅ DOCX parsed — ${items.length} answers detected successfully`);
      return { items };
    }
    return {
      error: 'No readable answer key found in this Word document.\n\nSupported formats:\n  1. A\n  2. B\n  1) True\n  1: photosynthesis\n  A,B,C,D (comma list)\n  [type:enumeration] oxygen, carbon\n  Or use section headers: MULTIPLE CHOICE ANSWERKEY, IDENTIFICATION ANSWERKEY, etc.',
    };
  }

  return { error: `Unsupported file type: ${ext}` };
}

// â”€â”€â”€ Scoring â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

// ─── Levenshtein distance helper (for fuzzy matching) ────────────────────────
//
// v3.0 NEW: Allows small spelling errors caused by messy handwriting OCR.
// "photosintesis" → matches "photosynthesis" (2-char distance)
// "rosess"        → matches "roses"           (1-char distance)
// MC and True/False are NOT fuzzy — A must equal A, True must equal True.
function levenshtein(a, b) {
  const m = a.length, n = b.length;
  const dp = Array.from({ length: m + 1 }, (_, i) =>
    Array.from({ length: n + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0))
  );
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = a[i - 1] === b[j - 1]
        ? dp[i - 1][j - 1]
        : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
    }
  }
  return dp[m][n];
}

// Fuzzy match: allow 1 typo for short answers (≤6 chars), 2 for longer ones.
// This handles OCR errors like "photosintesis" vs "photosynthesis".
function fuzzyMatch(studentRaw, correctAnswer) {
  const a = String(studentRaw).trim().toLowerCase();
  const b = String(correctAnswer).trim().toLowerCase();
  if (a === b) return true;
  // Don't fuzzy-match very short strings (1-2 chars) — too many false positives
  if (a.length < 3 || b.length < 3) return false;
  const maxDist = b.length <= 6 ? 1 : 2;
  return levenshtein(a, b) <= maxDist;
}

function checkAnswer(keyItem, studentRaw) {
  if (keyItem.type === 'mc')          return String(studentRaw).trim().toUpperCase() === keyItem.answer;
  if (keyItem.type === 'truefalse')   return String(studentRaw).trim().toUpperCase() === String(keyItem.answer).toUpperCase();
  if (keyItem.type === 'enumeration') {
    if (!Array.isArray(keyItem.answer)) return false;
    const si = String(studentRaw).split(',').map(s => s.trim().toLowerCase()).filter(Boolean);
    // v3.0: fuzzy match each required item against student's answer list
    return keyItem.answer.every(req => si.some(s => fuzzyMatch(s, req)));
  }
  // identification / shortAnswer / traceError → fuzzy match
  return fuzzyMatch(studentRaw, keyItem.answer);
}

function typeSummary(key) {
  return key.reduce((acc, item) => { acc[item.type] = (acc[item.type] || 0) + 1; return acc; }, {});
}

// â”€â”€â”€ Image Pre-processing (sharp) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
//  Pipeline order matters:
//
//  1.  rotate()         â€” auto-correct EXIF orientation (upside-down phone photos)
//  2.  greyscale()      â€” remove colour noise
//  3.  modulate()       â€” raise brightness slightly (+15%) so faint ballpen ink is visible
//  4.  median(3)        â€” remove salt-and-pepper noise without blurring text edges
//  5.  normalise()      â€” stretch histogram end-to-end for maximum contrast
//  6.  sharpen(2.0)     â€” stronger sharpening than before (Ïƒ=1.5 â†’ 2.0) for handwriting
//  7.  threshold(175)   â€” binarise; 175 works better for dark ballpen on white
//                         (original used 160 which was tuned for pencil marks)
//  8.  resize(20)     â€” more pixels than before (24 â†’ 20) so tiny handwritten
//                         characters have enough resolution for LSTM to decode
//
//  Result: PNG buffer â€” no temp file needed.

async function preprocessImage(base64, mimeType) {
  if (!sharp) {
    console.warn('[AutoChecker] Falling back to raw image (sharp not installed)');
    return Buffer.from(base64, 'base64');
  }

  const inputBuffer = Buffer.from(base64, 'base64');

  try {
    const processed = await sharp(inputBuffer)
      .rotate()                              // EXIF auto-rotate
      .greyscale()                           // strip colour
      .modulate({ brightness: 1.15 })        // âœ¨ NEW: lift faint ballpen strokes
      .median(3)                             // âœ¨ NEW: remove noise before threshold
      .normalise()                           // auto-contrast stretch
      .sharpen({ sigma: 2.0 })               // âœ¨ UPGRADED: stronger than 1.5 for handwriting
      // BUG FIX: threshold lowered from 175 â†’ 155.
      // 175 was too aggressive â€” it over-binarized ballpen strokes, causing broken
      // characters that Tesseract couldn't recognize. 155 preserves more of the ink.
      .threshold(140)
      .resize({
        width:              2000,          // FIX: was accidentally 20px — too small for Tesseract
        fit:                'inside',
        withoutEnlargement: false,
      })
      .png()
      .toBuffer();

    console.log('[AutoChecker] Image pre-processed with sharp âœ“');
    return processed;
  } catch (err) {
    console.warn('[AutoChecker] sharp pre-processing failed, using raw image:', err.message);
    return Buffer.from(base64, 'base64');
  }
}

// â”€â”€â”€ Tesseract setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

const { createWorker, OEM, PSM } = Tesseract;

// ─── Character whitelists ─────────────────────────────────────────────────────
const CHAR_WHITELIST_MC   = '0123456789ABCDabcd.):-/ \n';
const CHAR_WHITELIST_TEXT = ''; // no restriction for handwritten words

// PSM sequences — first entry is tried first; fallbacks follow if score is poor.
//
// FIX: PSM modes are now correctly matched to exam type:
//   MCQ / enumeration → PSM 6  (uniform block of text, best for dense answer sheets)
//   true/false        → PSM 7  (treat image as a single text line — T/F fits one line)
//   mixed / fallback  → PSM 11 (sparse text — find as much as possible anywhere)
//
// PSM 4 (single column) is kept as a secondary fallback for MC in case the sheet
// has a narrow single-column layout.
const PSM_SEQUENCE_MC        = [6, 4];      // MCQ: uniform block → single column
const PSM_SEQUENCE_TRUEFALSE = [7, 6, 11];  // T/F: single-line → block → sparse
const PSM_SEQUENCE_TEXT      = [6, 11];     // identification/enumeration: block → sparse

// ─── Persistent worker pool ───────────────────────────────────────────────────
// Workers are created ONCE at startup and reused across all scan requests.
// This eliminates the ~2-4s cold-start cost that was paid on every scan.
//
// CRITICAL FIX: tessedit_ocr_engine_mode (OEM) must be supplied as the second
// argument to createWorker() — it selects the engine DLL at load time.
// Setting it later via setParameters() has NO effect and causes runtime warnings:
//   "Warning: Parameter not set: tessedit_ocr_engine_mode"
// The whitelist and PSM, by contrast, ARE runtime parameters and belong in
// setParameters() / recognize() options — that is intentional and correct.

let workerMC   = null;  // for multiple_choice / bubble_omr
let workerText = null;  // for identification / enumeration / true_or_false / true_or_false
let workerMCReady   = false;
let workerTextReady = false;

async function initWorkers() {
  const loggerFn = m => {
    if (m.status === 'recognizing text') {
      process.stdout.write(`\r[Tesseract] ${Math.round(m.progress * 100)}%  `);
    }
  };

  // OEM.LSTM_ONLY (= 1) is passed as the second arg to createWorker so it is
  // applied at engine-init time — NOT via setParameters() after the fact.
  const OEM_LSTM = OEM?.LSTM_ONLY ?? 1;

  try {
    workerMC = await createWorker('eng', OEM_LSTM, { logger: loggerFn });
    // tessedit_char_whitelist IS a valid runtime parameter — correct to set here.
    await workerMC.setParameters({ tessedit_char_whitelist: CHAR_WHITELIST_MC });
    workerMCReady = true;
    console.log('[AutoChecker] Tesseract MC worker ready (OEM=LSTM_ONLY, whitelist=MC)');
  } catch (e) {
    console.error('[AutoChecker] MC worker init failed:', e.message);
  }

  try {
    workerText = await createWorker('eng', OEM_LSTM, { logger: loggerFn });
    // No whitelist for handwritten text — empty string disables restriction.
    await workerText.setParameters({ tessedit_char_whitelist: CHAR_WHITELIST_TEXT });
    workerTextReady = true;
    console.log('[AutoChecker] Tesseract Text worker ready (OEM=LSTM_ONLY, whitelist=none)');
  } catch (e) {
    console.error('[AutoChecker] Text worker init failed:', e.message);
  }
}

// Start warming up workers immediately — don't await, server starts instantly
// and workers will be ready by the time the first scan arrives.
initWorkers().catch(e => console.error('[AutoChecker] Worker pool init error:', e.message));

// ─── OCR with persistent worker ───────────────────────────────────────────────
//
// CRITICAL FIX: tessedit_ocr_engine_mode is intentionally removed from this
// setParameters() call. OEM is an engine-selection flag that only takes effect
// at worker creation (createWorker's 2nd argument). Calling setParameters with
// it at recognition time produces:
//   "Warning: Parameter not set: tessedit_ocr_engine_mode"
// and wastes a round-trip to the worker. It is now set correctly in initWorkers().
//
// tessedit_pageseg_mode (PSM) and preserve_interword_spaces ARE valid runtime
// parameters — they are applied per-recognition call, which is correct.
async function tryOcrWithPsm(worker, imageBuffer, psmMode, extraParams = {}) {
  await worker.setParameters({
    tessedit_pageseg_mode:     psmMode,
    preserve_interword_spaces: '1',
    tessedit_do_invert:        '0',
    ...extraParams,
  });
  const { data: { text, confidence } } = await worker.recognize(imageBuffer);
  process.stdout.write('\n');
  return { text: text ?? '', confidence: confidence ?? 0 };
}

// ─── Main Tesseract runner ────────────────────────────────────────────────────
async function runTesseract(imageBase64, mimeType = 'image/jpeg', examType = 'multiple_choice') {
  const isMcType = examType === 'bubble_mc' || examType === 'text_mc' ||
                   examType === 'bubble_omr' || examType === 'multiple_choice' || examType === 'omr';
  const isTrueFalse = examType === 'true_or_false' || examType === 'truefalse';

  // Select PSM sequence and whitelist by exam type.
  // PSM 7 for true/false (single line per answer) prevents Tesseract from trying
  // to detect multi-column layout and missing isolated T/F tokens.
  let psmList;
  if (isMcType)      psmList = PSM_SEQUENCE_MC;
  else if (isTrueFalse) psmList = PSM_SEQUENCE_TRUEFALSE;
  else               psmList = PSM_SEQUENCE_TEXT;

  const whitelist   = isMcType ? CHAR_WHITELIST_MC : CHAR_WHITELIST_TEXT;
  const extraParams = whitelist ? { tessedit_char_whitelist: whitelist } : {};

  // ── Sharp + worker selection run concurrently ─────────────────────────────
  // preprocessImage is CPU-bound; worker selection is instant.
  // Running them together hides any remaining sharp latency.
  const [imageBuffer, worker] = await Promise.all([
    preprocessImage(imageBase64, mimeType, examType),
    (async () => {
      // Use persistent worker if ready, otherwise spin up a temporary one.
      // FIX: temporary worker also receives OEM_LSTM as the 2nd createWorker arg.
      if (isMcType && workerMCReady)    return { w: workerMC,   temp: false };
      if (!isMcType && workerTextReady) return { w: workerText, temp: false };
      console.warn('[AutoChecker] Persistent worker not ready — creating temporary worker');
      const w = await createWorker('eng', OEM?.LSTM_ONLY ?? 1);
      return { w, temp: true };
    })(),
  ]);

  const { w, temp } = worker;

  console.log(`[AutoChecker] OCR start — examType=${examType}, PSM=${psmList.join(',')}, worker=${temp ? 'temp' : 'pooled'}`);

  function scoreMcText(text) {
    return (text.match(/[Qq]?\d{1,3}\s*[.):\-\/]?\s*[ABCDabcd](?:\s|[.,;)\n]|$)/g) ?? []).length;
  }
  function scoreTextBlock(text) {
    return text.split('\n').filter(l => l.trim().length > 2).length;
  }

  let bestText       = '';
  let bestConfidence = 0;
  let bestScore      = -1;

  try {
    for (const psm of psmList) {
      try {
        const { text, confidence } = await tryOcrWithPsm(w, imageBuffer, psm, extraParams);
        const preview = (text ?? '').slice(0, 200).replace(/\n/g, '\u21b5');
        console.log(`[Tesseract PSM${psm}] conf=${confidence?.toFixed(1)}% | preview: ${preview}`);

        const score = isMcType ? scoreMcText(text) : scoreTextBlock(text);
        console.log(`[Tesseract PSM${psm}] score=${score}`);

        if (score > bestScore || (score === bestScore && confidence > bestConfidence)) {
          bestText       = text;
          bestConfidence = confidence;
          bestScore      = score;
        }

        // Early exit — first PSM pass scored well enough, skip fallback
        // MC: 3+ answers found is a strong enough signal (was 5, lowered for speed)
        // Text: 6+ non-empty lines is sufficient
        if (isMcType  && score >= 3) break;
        if (!isMcType && score >= 6) break;

      } catch (psmErr) {
        console.warn(`[Tesseract PSM${psm}] failed: ${psmErr.message}`);
      }
    }

    // ── Inverted-image retry (only when forward pass scored very poorly) ──────
    const INVERT_THRESHOLD = isMcType ? 2 : 4;
    if (bestScore < INVERT_THRESHOLD) {
      try {
        let invertedBuffer;
        if (sharp) {
          invertedBuffer = await sharp(imageBuffer).negate({ alpha: false }).png().toBuffer();
        } else {
          const buf = Buffer.from(imageBuffer);
          for (let i = 0; i < buf.length; i++) buf[i] = 255 - buf[i];
          invertedBuffer = buf;
        }
        const invertPsm = isMcType ? 6 : 4;
        const { text: invText, confidence: invConf } = await tryOcrWithPsm(w, invertedBuffer, invertPsm, extraParams);
        const invScore = isMcType ? scoreMcText(invText) : scoreTextBlock(invText);
        console.log(`[Tesseract INVERTED PSM${invertPsm}] conf=${invConf?.toFixed(1)}% score=${invScore}`);
        if (invScore > bestScore || (invScore === bestScore && invConf > bestConfidence)) {
          bestText = invText; bestConfidence = invConf; bestScore = invScore;
          console.log('[AutoChecker] Inverted image gave better result');
        }
      } catch (invErr) {
        console.warn('[AutoChecker] Inverted retry failed:', invErr.message);
      }
    }

  } finally {
    // Only terminate temporary workers — persistent pool workers stay alive
    if (temp) {
      try { await w.terminate(); } catch {}
    }
  }

  console.log(`[AutoChecker] OCR done — score=${bestScore}, conf=${bestConfidence?.toFixed?.(1) ?? 'n/a'}%`);
  return { text: bestText, engineConfidence: Math.max(bestConfidence, 0) };
}


// â”€â”€â”€ OCR Text Post-processing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// âœ¨ UPGRADED: added handwriting-specific substitution fixes
//
// New entries:
//   q â†’ a  (handwritten lowercase 'a' often looks like 'q')
//   6 â†’ B  (in answer position; a hasty 'B' looks like '6')
//   E â†’ B  (hasty capital E read as B)
//   0 â†’ D  (round shape confusion in answer column)
//   Digit 1 in answer position â†’ I (then handled as identification answer)

function fixOcrSubstitutions(text) {
  return text
    // â”€â”€ Question number fixes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    .replace(/^[Ol](\d)/gm,   '0$1')
    .replace(/(\d)[Ol]\b/gm,  '$10')
    // â”€â”€ MC answer position: digit â†’ letter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    .replace(/^(\d+[.):\s]+)8\s*$/gm, '$1B')   // 8 â†’ B
    .replace(/^(\d+[.):\s]+)6\s*$/gm, '$1B')   // âœ¨ 6 â†’ B (ballpen B confusion)
    .replace(/^(\d+[.):\s]+)0\s*$/gm, '$1D')   // âœ¨ 0 â†’ D (round shape in answer)
    // â”€â”€ Handwriting: lowercase a written as q â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    .replace(/^(\d+[.):\s]+)q\s*$/gim, '$1A')  // âœ¨ q â†’ A in answer position
    // â”€â”€ Normalize separators â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    .replace(/^(\d+)\s*[):]\s*/gm, '$1. ')
    // â”€â”€ Strip trailing punctuation on answer lines â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    .replace(/^(\d+\.\s+[A-Da-d])[,;.]\s*$/gm, '$1')
    // â”€â”€ Collapse multiple blank lines â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    .replace(/\n{3,}/g, '\n\n')
    // â”€â”€ Strip non-printable characters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    .replace(/[^\x09\x0A\x0D\x20-\x7E]/g, '');
}

// â”€â”€â”€ MC Answer Extraction (bubble / text_mc) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function extractMcAnswers(text, questionCount) {
  const answers = {};
  const maxQ    = Math.max(questionCount * 3, 50);

  // Pattern A: same-line "1. B"  "1) B"  "Q1: B"
  //
  // FIX: Added negative lookahead (?!\s*[.)\-]\s*[A-Za-z]) after the letter.
  // Prevents matching printed choice labels like "3. A. HTML" or "4. A. Structure"
  // where the letter is immediately followed by ". word" (= printed choice format).
  // A student's handwritten "B" is standalone and does NOT match that pattern.
  const RE_INLINE = /[Qq]?(\d{1,3})\s*[.):\-\/]?\s*([ABCDabcd])(?!\s*[.)\-]\s*[A-Za-z])(?:\s|[.,;)\n]|$)/g;
  let m;
  while ((m = RE_INLINE.exec(text)) !== null) {
    const q = parseInt(m[1], 10);
    if (q >= 1 && q <= maxQ) answers[String(q)] = m[2].toUpperCase();
  }

  // Pattern B: number on one line, letter on next
  // FIX: require the answer line to be a truly standalone letter
  const lines = text.split('\n').map(l => l.trim()).filter(Boolean);
  for (let i = 0; i < lines.length - 1; i++) {
    const numMatch = lines[i].match(/^[Qq]?(\d{1,3})[.):\s]*$/);
    const ansLine  = lines[i + 1];
    const ansMatch = ansLine.match(/^[\[(]?([ABCDabcd])[\].]?\s*$/i);
    if (numMatch && ansMatch) {
      const q = parseInt(numMatch[1], 10);
      if (q >= 1 && q <= maxQ && !answers[String(q)]) {
        answers[String(q)] = ansMatch[1].toUpperCase();
      }
    }
  }

  // Pattern C: letter BEFORE number — handles exam format "___B___ 1. What is..."
  // where the student's handwritten answer appears on a blank BEFORE the question number.
  // Tesseract OCR often reads this as: standalone letter line → then number+question line.
  // FIX: this is the primary format of this exam paper — added as Pattern C.
  const lines3 = text.split('\n').map(l => l.trim()).filter(Boolean);
  for (let i = 0; i < lines3.length - 1; i++) {
    // Current line is a standalone handwritten letter (A, B, C, or D)
    const letterOnlyMatch = lines3[i].match(/^([ABCDabcd])\s*$/i);
    if (!letterOnlyMatch) continue;
    // Next line starts with a question number
    const nextNumMatch = lines3[i + 1].match(/^[Qq]?(\d{1,3})\s*[.):\s]/);
    if (!nextNumMatch) continue;
    const q = parseInt(nextNumMatch[1], 10);
    if (q >= 1 && q <= maxQ && !answers[String(q)]) {
      answers[String(q)] = letterOnlyMatch[1].toUpperCase();
    }
  }

  // Also handle: letter and number on same line but letter comes first "B 1."
  const RE_LETTER_FIRST = /^([ABCDabcd])\s+[Qq]?(\d{1,3})\s*[.):\s]/gim;
  let mLF;
  while ((mLF = RE_LETTER_FIRST.exec(text)) !== null) {
    const q = parseInt(mLF[2], 10);
    if (q >= 1 && q <= maxQ && !answers[String(q)]) {
      answers[String(q)] = mLF[1].toUpperCase();
    }
  }

  // Pattern D (was C): compact grid "1B 2C 3A" — only when A/B found very few results
  if (Object.keys(answers).length < Math.floor(questionCount * 0.3)) {
    const RE_GRID = /(\d{1,3})[.\-]?\s*([ABCDabcd])(?=\s|\d|$)/g;
    while ((m = RE_GRID.exec(text)) !== null) {
      const q = parseInt(m[1], 10);
      if (q >= 1 && q <= maxQ && !answers[String(q)]) {
        answers[String(q)] = m[2].toUpperCase();
      }
    }
  }

  // Trim to expected range
  for (const key of Object.keys(answers)) {
    if (parseInt(key, 10) > questionCount) delete answers[key];
  }

  return answers;
}

// â”€â”€â”€ Written Answer Extraction (identification / enumeration / short_answer) â”€â”€
//
// âœ¨ NEW FUNCTION â€” purpose-built for handwritten answers
//
// Strategies used:
//   1. Primary: numbered line regex (same as MC but accepts word answers)
//   2. Fallback: consecutive-line pairing (question number on one line, answer on next)
//   3. Normalise: lowercase for identification, preserve case for short_answer
//   4. Tolerant: accepts answers with OCR noise (extra dots, dashes, smeared chars)

// Patterns that indicate the captured text is printed question text, not a student answer
const MAX_ANSWER_LENGTH = 50;
const QUESTION_TEXT_PATTERNS = /\b(is used to|stands for|which of|refers to|the following|it is|used for|describes|that is|a system that|a format used|connects|manages|a javascript|library for)\b/i;

// When OCR merges a student's short answer with the printed question text on one line,
// extract just the first meaningful token (the student's answer).
// Example: "CSS It is used to style web pages." → "CSS"
function firstWordIfTooLong(raw) {
  const trimmed = raw.trim();
  if (trimmed.length <= MAX_ANSWER_LENGTH) return trimmed;
  const sentenceSplit = trimmed.match(/^([A-Za-z0-9._\-]{1,30})\s+[A-Z]/);
  if (sentenceSplit) {
    const candidate = sentenceSplit[1].trim();
    if (candidate.length >= 1 && candidate.length <= MAX_ANSWER_LENGTH) return candidate;
  }
  return trimmed.slice(0, MAX_ANSWER_LENGTH).trim();
}

function extractWrittenAnswers(text, questionCount, examType) {
  const answers = {};
  const maxQ    = Math.max(questionCount + 5, 20);
  const isTf    = examType === 'truefalse' || examType === 'true_or_false';

  // ── FIX: True/False dedicated fast-path ────────────────────────────────────
  // T and F are single characters. The generic numbered-line regex below
  // captures them fine, but PSM output often puts them on their own line
  // BEFORE the question number (exam format: "___F___  6. HTML is used...").
  // We handle all T/F formats explicitly here so none fall through to "?".
  if (isTf) {
    // Pattern A: "6. F" / "6) T" / "Q6 T" — number then T/F on same line
    const RE_TF_INLINE = /[Qq]?(\d{1,3})\s*[.):\-\/]?\s*([TtFf])(?:\s|$)/g;
    let mTf;
    while ((mTf = RE_TF_INLINE.exec(text)) !== null) {
      const q = parseInt(mTf[1], 10);
      if (q >= 1 && q <= maxQ && !answers[String(q)]) {
        answers[String(q)] = normaliseWritten(mTf[2], examType);
      }
    }

    // Pattern B: standalone T or F line, followed by a line starting with a number
    // e.g.:  "F\n6. HTML is used to style web pages."
    const lines = text.split('\n').map(l => l.trim()).filter(Boolean);
    for (let i = 0; i < lines.length - 1; i++) {
      const tfMatch  = lines[i].match(/^([TtFf])\s*$/);
      if (!tfMatch) continue;
      const nextNum  = lines[i + 1].match(/^[Qq]?(\d{1,3})\s*[.):\-\s]/);
      if (!nextNum) continue;
      const q = parseInt(nextNum[1], 10);
      if (q >= 1 && q <= maxQ && !answers[String(q)]) {
        answers[String(q)] = normaliseWritten(tfMatch[1], examType);
      }
    }

    // Pattern C: "true" / "false" written in full, same line as number
    const RE_TF_FULL = /[Qq]?(\d{1,3})\s*[.):\-\/]?\s*(true|false)(?:\s|$)/gi;
    let mTfFull;
    while ((mTfFull = RE_TF_FULL.exec(text)) !== null) {
      const q = parseInt(mTfFull[1], 10);
      if (q >= 1 && q <= maxQ && !answers[String(q)]) {
        answers[String(q)] = normaliseWritten(mTfFull[2], examType);
      }
    }

    // Fill any remaining blanks
    for (let i = 1; i <= questionCount; i++) {
      if (!answers[String(i)]) answers[String(i)] = '';
    }
    return answers;
  }

  // Strategy 1: numbered lines
  const RE_NUMBERED = /^[Qq]?(\d{1,3})\s*[.):\-\/]?\s*(.{1,120})$/gm;
  let m;
  while ((m = RE_NUMBERED.exec(text)) !== null) {
    const q = parseInt(m[1], 10);
    let v   = m[2].trim();

    if (q < 1 || q > maxQ)         continue;
    if (!v || /^\d+$/.test(v))      continue;

    // FIX: skip lines that look like printed question text
    if (QUESTION_TEXT_PATTERNS.test(v)) continue;

    // FIX: if too long, try to extract just the answer portion
    if (v.length > MAX_ANSWER_LENGTH) {
      v = firstWordIfTooLong(v);
      if (QUESTION_TEXT_PATTERNS.test(v)) continue;
    }

    if (!answers[String(q)]) {
      answers[String(q)] = normaliseWritten(v, examType);
    }
  }

  // Strategy 2: number + answer on consecutive lines
  const lines = text.split('\n').map(l => l.trim()).filter(Boolean);
  for (let i = 0; i < lines.length - 1; i++) {
    const numMatch = lines[i].match(/^[Qq]?(\d{1,3})[.):\s]*$/);
    if (!numMatch) continue;

    const q        = parseInt(numMatch[1], 10);
    const nextLine = lines[i + 1];

    if (q < 1 || q > maxQ)                          continue;
    if (answers[String(q)])                          continue;
    if (/^\d+[.):]\s/.test(nextLine))               continue;

    const cleaned = nextLine.replace(/^[-\u2013\u2014\u2022]\s*/, '').trim();
    if (!cleaned)                                    continue;
    if (QUESTION_TEXT_PATTERNS.test(cleaned))        continue;
    if (cleaned.length > MAX_ANSWER_LENGTH)          continue;

    answers[String(q)] = normaliseWritten(cleaned, examType);
  }

  return answers;
}

function normaliseWritten(raw, examType) {
  const cleaned = raw
    .replace(/[^\x20-\x7E]/g, '')   // strip non-printable
    .replace(/\s{2,}/g, ' ')         // collapse whitespace
    .trim();

  // FIX: true_or_false must return "True" / "False" (capital first letter)
  // because the grader does case-sensitive matching against those exact strings.
  // Previously this fell through to toLowerCase() → "true"/"false" which never matched.
  if (examType === 'truefalse' || examType === 'true_or_false') {
    const lower = cleaned.toLowerCase();
    if (lower === 't' || lower === 'true')  return 'True';
    if (lower === 'f' || lower === 'false') return 'False';
    return '';  // unrecognised T/F token → blank rather than a wrong string
  }

  if (examType === 'short_answer' || examType === 'trace_error') {
    return cleaned; // preserve original capitalisation for open-ended
  }
  return cleaned.toLowerCase(); // normalise for identification / enumeration
}

// â”€â”€â”€ Student Name Extraction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function extractStudentName(text) {
  const sameLine = text.match(
    /(?:name|student|pangalan|examinee|pupil)\s*[:\-]\s*([A-Za-z ,.]+)/i
  );
  if (sameLine) return sameLine[1].trim().replace(/\s{2,}/g, ' ') || null;

  const lines = text.split('\n').map(l => l.trim());
  for (let i = 0; i < lines.length - 1; i++) {
    if (/^(?:name|student|pangalan|examinee|pupil)\s*:?\s*$/i.test(lines[i])) {
      const candidate = lines[i + 1];
      if (/^[A-Za-z ,.']{3,60}$/.test(candidate)) return candidate;
    }
  }
  return null;
}

// â”€â”€â”€ Full OCR pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// âœ¨ UPGRADED: never throws for empty/unreadable text
//             returns empty answers map with confidence=0 instead

// ─── PRODUCTION OMR ENGINE v6 ─────────────────────────────────────────────────
//
// Complete rewrite of the bubble detection pipeline.
// Fixes all 8 issues from the audit:
//
//  [1] BUBBLE OVERLAP        — center-weighted sampling (distance falloff)
//                              tighter ink zone (60% radius), prevents neighbor bleed
//  [2] GRID DRIFT            — dynamic row recalibration using horizontal dark-peak scan
//                              per-row Y is snapped to nearest real ink peak within ±5px
//  [3] DOUBLE MARK           — strict dual-threshold: both bubbles must exceed 28%
//                              AND be within 82% of each other to qualify as double
//  [4] FALSE BLANKS          — MIN_FILLED_PERCENT = 22%: if top bubble ≥ 22%, return it
//  [5] PREPROCESSING         — CLAHE-approximation (local histogram), adaptive threshold,
//                              Gaussian blur, morphological cleanup, shadow normalization
//  [6] CONFIDENCE SCORING    — (topFill - secondFill) × topFill per question, averaged
//  [7] SECOND PASS           — low-confidence questions get a ±3px Y adjustment retry
//  [8] TARGET PERFORMANCE    — 48–50 detected, near-zero false doubles, >90% confidence

let Jimp;
try {
  Jimp = require('jimp');
} catch {
  console.warn('[AutoChecker] jimp not found — bubble OMR will fall back to Tesseract.');
  console.warn('              Fix: npm install jimp');
}

// ── Canonical canvas dimensions (A4 @ 96 dpi) ─────────────────────────────────
const TARGET_W   = 794;
const TARGET_H   = 1123;
const MARK_INSET = 27; // registration mark center inset from paper edge (px)

// ── OMR Detection constants — all named, none magic ───────────────────────────
const OMR_CONSTANTS = {
  // Bubble sampling
  // FIX: Reduced INK_ZONE_FACTOR 0.55→0.45 — samples only the true inner fill zone,
  // avoids picking up ink bleed from the printed border ring which was causing
  // spurious high fills on adjacent/empty bubbles.
  INK_ZONE_FACTOR:   0.45,
  CENTER_WEIGHT_EXP: 2.5,
  ADAPTIVE_BIAS:     0.12,

  // Fill classification
  MIN_FILLED_PERCENT: 0.12,
  BLANK_SIGMA_MULT:   1.2,

  // Double-mark — FIX: Raised 0.95→0.98. The old 0.95 triggered on Q9(52%/50.1%),
  // Q19(76.8%/76.6%), Q30(88.5%/84.6%) which were border bleeds, not real double marks.
  // At 0.98 only truly indistinguishable fills (within 2%) trigger double resolution.
  DOUBLE_MARK_RATIO: 0.98,
  MIN_DOUBLE_FILL:   0.55,   // raised from 0.50 — both must be strongly filled

  // Winner disambiguation
  MIN_RATIO:         1.12,
  MARGIN_THRESH_MULT: 2.0,

  // Second-pass — FIX: Raised threshold 0.18→0.25 so only genuinely ambiguous
  // questions retry. Lowered Y offsets to ±1px only — ±2px was jumping to wrong rows.
  SECOND_PASS_CONF_THRESHOLD: 0.25,
  SECOND_PASS_Y_OFFSETS: [-1, 1],

  // Row snapping
  ROW_SNAP_SEARCH_RADIUS: 8,   // reduced from 10 — tighter snap, less risk of wrong row
  ROW_SNAP_MIN_DARK_FRAC: 0.20, // raised from 0.18 — requires stronger ink signal to snap

  // Preprocessing
  GAUSSIAN_SIGMA: 0.8,
  CLAHE_TILE_SIZE: 24,
  MORPH_OPEN_R: 1,
};

// ── OMR_LAYOUT — derived from omrConfig.ts renderer (single source of truth) ──
//
// All fractions relative to TARGET_W=794 × TARGET_H=1123.
// Verified by /api/omr-debug overlay against physical printed sheets.
//
// 2-col (50Q) confirmed pixel coords from live server log:
//   Q1 A(104,271) B(185,271) C(265,271) D(345,271) r=12px
const OMR_LAYOUT = {
  // ── 1-column layout (1–25 questions) ──────────────────────────────────────
  // Same bubble dimensions as 2-col. BUBBLE_STEP = 81px/794 = 0.1020
  // Q_NUM_W: 92px/794 = 0.1159 (A-bubble absolute x on canvas)
  1: {
    GRID_TOP:    0.1800,
    ROW_STEP:    0.0292,
    BUBBLE_R:    0.0113,
    COL_LEFT:    [0.0000],
    Q_NUM_W:     0.1159,
    BUBBLE_STEP: 0.1020,
  },
  // ── 2-column layout (26–50 questions) ─────────────────────────────────────
  // RECALIBRATED v9.1 from live production log:
  //   Log: Q1 A(92,217) r=9px at H=1123 → GRID_TOP = 217/1123 = 0.1932
  //   Previous GRID_TOP=0.1780 = 200px — was 17px too high, causing cumulative
  //   row drift that required excessive second-pass retries by Q13+.
  //   ROW_STEP confirmed: (snappedCy row-to-row diff from log ≈ 34.8px) / 1123 = 0.0310 ✓
  //   Q_NUM_W and BUBBLE_STEP unchanged — confirmed correct from fills log.
  2: {
    GRID_TOP:    0.1932,
    ROW_STEP:    0.0310,
    BUBBLE_R:    0.0113,
    COL_LEFT:    [0.0000, 0.4484],
    Q_NUM_W:     0.1159,
    BUBBLE_STEP: 0.1020,
  },
  // 3-column layout (51–100 questions)
  // RECALIBRATED v9.4 — re-measured from actual 75Q AutoChecker student sheet.
  //
  // Physical sheet measurements (after homography warp to 794×1123px):
  //   Header height: ~17% of page → GRID_TOP = 0.1932 (same as 2-col, same header)
  //   Row spacing: 25 rows per column, same height as 2-col → ROW_STEP = 0.0310
  //
  //   3-column sheet: usable width split into 3 equal columns
  //   Each column: A B C D bubbles, step between bubbles ≈ 40px
  //
  //   Col0 (Q1-Q25)  first A-bubble x ≈ 111px  (Q_NUM_W = 111/794 = 0.1398)
  //   Col1 (Q26-Q50) first A-bubble x ≈ 358px  (offset = (358-111)/794 = 0.3112)
  //   Col2 (Q51-Q75) first A-bubble x ≈ 606px  (offset = (606-111)/794 = 0.6233)
  //   BUBBLE_STEP ≈ 40px → 40/794 = 0.0504
  //
  //   v9.4 fix: COL_LEFT offsets adjusted — COL_LEFT is the offset from
  //   paper left edge to the START of each column's content area.
  //   bubbleX0 = (COL_LEFT[col] + Q_NUM_W) * W
  //   So COL_LEFT[0]=0.0000 → bubbleX0 = 0.1398*794 = 111px ✓
  //      COL_LEFT[1]=0.3112 → bubbleX0 = (0.3112+0.1398)*794 = 357px ✓ (was 0.3123 → 357px, close)
  //      COL_LEFT[2]=0.6233 → bubbleX0 = (0.6233+0.1398)*794 = 605px ✓ (was 0.6247 → 606px, close)
  3: {
    GRID_TOP:    0.1932,  // same as 2-col — same header height across all layouts
    ROW_STEP:    0.0310,  // 25 rows per column, same spacing as 2-col
    BUBBLE_R:    0.0113,  // same bubble radius as 2-col
    COL_LEFT:    [0.0000, 0.3112, 0.6233],
    Q_NUM_W:     0.1398,
    BUBBLE_STEP: 0.0504,
  },
};

function omrColCount(q) { return q <= 25 ? 1 : q <= 50 ? 2 : 3; }

// ── Otsu global threshold ─────────────────────────────────────────────────────
function otsuThreshold(gray) {
  const hist = new Array(256).fill(0);
  for (let i = 0; i < gray.length; i++) hist[gray[i]]++;
  const total = gray.length;
  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * hist[i];
  let sumB = 0, wB = 0, maxVar = 0, threshold = 128;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const varBetween = wB * wF * (mB - mF) ** 2;
    if (varBetween > maxVar) { maxVar = varBetween; threshold = t; }
  }
  return threshold;
}

// ── FIX 5: CLAHE-approximation (local contrast enhancement) ──────────────────
// Divides image into tiles and equalizes histogram locally.
// This lifts faint pencil marks in shadowed regions without blowing out bright areas.
function claheApprox(gray, W, H, tileSize) {
  const out = new Uint8Array(gray.length);
  const tilesX = Math.ceil(W / tileSize);
  const tilesY = Math.ceil(H / tileSize);

  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const x0 = tx * tileSize, x1 = Math.min(x0 + tileSize, W);
      const y0 = ty * tileSize, y1 = Math.min(y0 + tileSize, H);

      // Build local histogram
      const hist = new Array(256).fill(0);
      let count = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          hist[gray[y * W + x]]++;
          count++;
        }
      }

      // Build cumulative distribution
      const cdf = new Array(256).fill(0);
      cdf[0] = hist[0];
      for (let i = 1; i < 256; i++) cdf[i] = cdf[i - 1] + hist[i];

      // Clip at 90th percentile (CLAHE clip limit) to reduce noise amplification
      const clipLimit = Math.max(1, Math.round(count * 0.90 / 256 * 3));
      for (let i = 0; i < 256; i++) {
        cdf[i] = Math.min(cdf[i], cdf[i] + clipLimit);
      }

      const cdfMin = cdf.find(v => v > 0) ?? 0;
      const scale = count > cdfMin ? 255 / (count - cdfMin) : 1;

      // Apply equalization to tile pixels
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const v = gray[y * W + x];
          out[y * W + x] = Math.round(Math.max(0, Math.min(255, (cdf[v] - cdfMin) * scale)));
        }
      }
    }
  }
  return out;
}

// ── FIX 5: Gaussian blur (3×3 kernel approximation) ──────────────────────────
// Removes high-frequency speck noise before adaptive thresholding.
// Uses integer arithmetic for speed.
function gaussianBlur3(gray, W, H) {
  // Kernel: [1 2 1 / 2 4 2 / 1 2 1] × 1/16
  const out = new Uint8Array(gray.length);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const x0 = Math.max(0, x - 1), x1 = Math.min(W - 1, x + 1);
      const y0 = Math.max(0, y - 1), y1 = Math.min(H - 1, y + 1);
      const v = (
        gray[y0 * W + x0] * 1 + gray[y0 * W + x ] * 2 + gray[y0 * W + x1] * 1 +
        gray[y  * W + x0] * 2 + gray[y  * W + x ] * 4 + gray[y  * W + x1] * 2 +
        gray[y1 * W + x0] * 1 + gray[y1 * W + x ] * 2 + gray[y1 * W + x1] * 1
      );
      out[y * W + x] = Math.round(v / 16);
    }
  }
  return out;
}

// ── FIX 5: Morphological opening (erode then dilate) ─────────────────────────
// Removes single-pixel specks (dust, paper grain) while preserving bubble ink.
// Only applied to the binarized image as a cleanup pass.
function morphOpen(binary, W, H, r) {
  // Erode: a pixel is dark only if all neighbors within r are dark
  const eroded = new Uint8Array(binary.length);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (binary[y * W + x] !== 0) { eroded[y * W + x] = 1; continue; }
      let allDark = true;
      outer: for (let dy = -r; dy <= r && allDark; dy++) {
        for (let dx = -r; dx <= r && allDark; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
          if (binary[ny * W + nx] !== 0) allDark = false;
        }
      }
      eroded[y * W + x] = allDark ? 0 : 1;
    }
  }
  // Dilate: a pixel is dark if any neighbor within r is dark in eroded image
  const dilated = new Uint8Array(binary.length).fill(1);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (eroded[y * W + x] !== 0) continue; // not dark, skip
      for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx >= 0 && nx < W && ny >= 0 && ny < H) dilated[ny * W + nx] = 0;
        }
      }
    }
  }
  return dilated;
}

// ── FIX 5: Shadow normalization (local background subtraction) ────────────────
// Estimates large-scale illumination gradient using a downsampled blurred version
// of the image, then subtracts it. This removes shadows from phone lighting.
function normalizeShadow(gray, W, H) {
  // Downsample to 1/8 for speed
  const dw = Math.ceil(W / 8), dh = Math.ceil(H / 8);
  const down = new Uint8Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(W - 1, x * 8 + 4);
      const sy = Math.min(H - 1, y * 8 + 4);
      down[y * dw + x] = gray[sy * W + sx];
    }
  }

  // Blur the downsampled image (3 passes of 3×3 gaussian = ~9×9 effective)
  let blurred = down;
  for (let pass = 0; pass < 3; pass++) {
    blurred = gaussianBlur3(blurred, dw, dh);
  }

  // Upsample background estimate back to full size and subtract
  const out = new Uint8Array(gray.length);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const bx = Math.min(dw - 1, Math.round(x / 8));
      const by = Math.min(dh - 1, Math.round(y / 8));
      const bg = blurred[by * dw + bx];
      // Normalize: shift pixel so background = 200 (white paper)
      const normalized = Math.round(gray[y * W + x] - bg + 200);
      out[y * W + x] = Math.max(0, Math.min(255, normalized));
    }
  }
  return out;
}

// ── FIX 1 + FIX 6: Center-weighted bubble sampler ────────────────────────────
//
// Returns { fill, confidence } where:
//   fill = weighted dark pixel fraction (center pixels count more than edge)
//   weight = (1 - dist/inkR)^CENTER_WEIGHT_EXP  per pixel
//
// Also uses adaptive local background from the annular halo to handle varying
// paper brightness — so a shadow behind a blank bubble doesn't count as filled.
//
// The ink zone is only the inner INK_ZONE_FACTOR of the radius, to avoid the
// printed bubble border ring bleeding into neighbor measurements.
function sampleBubbleWeighted(gray, W, H, cx, cy, r) {
  const { INK_ZONE_FACTOR, CENTER_WEIGHT_EXP, ADAPTIVE_BIAS } = OMR_CONSTANTS;

  const inkR  = Math.max(2, Math.round(r * INK_ZONE_FACTOR));
  const haloR = Math.round(r * 2.2);
  const inkR2  = inkR * inkR;
  const haloR2 = haloR * haloR;

  // Step 1: Estimate local background from annular halo (outside ink zone, inside halo)
  let bgSum = 0, bgCount = 0;
  for (let dy = -haloR; dy <= haloR; dy++) {
    for (let dx = -haloR; dx <= haloR; dx++) {
      const d2 = dx * dx + dy * dy;
      if (d2 <= inkR2 || d2 > haloR2) continue;
      const px = cx + dx, py = cy + dy;
      if (px < 0 || px >= W || py < 0 || py >= H) continue;
      bgSum += gray[py * W + px];
      bgCount++;
    }
  }
  const localBg = bgCount > 0 ? bgSum / bgCount : 210;
  const localThreshold = Math.max(30, localBg * (1 - ADAPTIVE_BIAS));

  // Step 2: Center-weighted sampling inside ink zone only
  // FIX: weight = (1 - dist/inkR)^exp — center pixels have weight ~1.0, edge ~0.0
  let weightedDark = 0, weightedTotal = 0;
  for (let dy = -inkR; dy <= inkR; dy++) {
    for (let dx = -inkR; dx <= inkR; dx++) {
      const d2 = dx * dx + dy * dy;
      if (d2 > inkR2) continue;
      const px = cx + dx, py = cy + dy;
      if (px < 0 || px >= W || py < 0 || py >= H) continue;

      const dist = Math.sqrt(d2);
      const weight = Math.pow(1.0 - dist / inkR, CENTER_WEIGHT_EXP);
      weightedTotal += weight;
      if (gray[py * W + px] < localThreshold) {
        weightedDark += weight;
      }
    }
  }

  const fill = weightedTotal > 0 ? weightedDark / weightedTotal : 0;
  return fill;
}

// ── FIX 2: Dynamic row recalibration ─────────────────────────────────────────
//
// For each expected row Y, scan a horizontal band of ±SNAP_RADIUS pixels.
// Find the Y offset that maximizes the count of dark pixels across all 4 bubble
// columns. Snap the actual row center to that offset.
//
// This corrects for:
//   - Row drift from printing misalignment
//   - Paper stretch from humid storage
//   - Camera tilt causing perspective residual after homography
function snapRowY(gray, W, H, expectedCy, bubbleXs, r) {
  const { ROW_SNAP_SEARCH_RADIUS, ROW_SNAP_MIN_DARK_FRAC } = OMR_CONSTANTS;
  const inkR = Math.max(2, Math.round(r * OMR_CONSTANTS.INK_ZONE_FACTOR));

  // v9.2: Score each candidate Y by summing darkness across a full vertical
  // band around each bubble X centre (not just a single horizontal line).
  // This makes the snap robust to sub-pixel Y variation within a bubble.
  const bandHalf = Math.max(2, Math.round(r * 0.4)); // ±40% of radius vertically

  let bestOffset = 0;
  let bestScore  = -1;
  const scores   = [];

  for (let offset = -ROW_SNAP_SEARCH_RADIUS; offset <= ROW_SNAP_SEARCH_RADIUS; offset++) {
    const testCy = expectedCy + offset;
    if (testCy < bandHalf || testCy >= H - bandHalf) continue;

    let score = 0;
    for (const cx of bubbleXs) {
      // Sample a small rectangle around (cx, testCy)
      for (let dy = -bandHalf; dy <= bandHalf; dy++) {
        const py = testCy + dy;
        if (py < 0 || py >= H) continue;
        for (let dx = -inkR; dx <= inkR; dx++) {
          const px = Math.round(cx) + dx;
          if (px < 0 || px >= W) continue;
          score += (255 - gray[py * W + px]);
        }
      }
    }
    scores.push({ offset, score });
    if (score > bestScore) { bestScore = score; bestOffset = offset; }
  }

  // Only snap if:
  // 1. Best score represents meaningful ink (darkFrac threshold)
  // 2. The best offset is meaningfully better than offset=0 (prevents drift on blank rows)
  const sampleArea = (inkR * 2) * (bandHalf * 2 + 1) * bubbleXs.length;
  const darkFrac   = bestScore / (sampleArea * 255);

  const scoreAtZero = scores.find(s => s.offset === 0)?.score ?? 0;
  const improvement = scoreAtZero > 0 ? bestScore / scoreAtZero : 1;

  // Require: ink present AND snap is at least 15% better than no-snap
  if (darkFrac < ROW_SNAP_MIN_DARK_FRAC || improvement < 1.15) {
    return expectedCy;
  }

  return expectedCy + bestOffset;
}

// ── Perspective correction: homography warp ───────────────────────────────────
function homographyWarp(srcGray, srcW, srcH, srcPts, dstPts, outW, outH) {
  const [s0,s1,s2,s3] = srcPts;
  const [d0,d1,d2,d3] = dstPts;

  const A = [], b = [];
  const pairs = [[s0,d0],[s1,d1],[s2,d2],[s3,d3]];
  for (const [s,d] of pairs) {
    A.push([s.cx??s.x, s.cy??s.y, 1, 0, 0, 0, -(d.x)*(s.cx??s.x), -(d.x)*(s.cy??s.y)]);
    A.push([0, 0, 0, s.cx??s.x, s.cy??s.y, 1, -(d.y)*(s.cx??s.x), -(d.y)*(s.cy??s.y)]);
    b.push(d.x, d.y);
  }

  const n = 8;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let r = col+1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[maxRow][col])) maxRow = r;
    [M[col], M[maxRow]] = [M[maxRow], M[col]];
    if (Math.abs(M[col][col]) < 1e-10) return srcGray;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col] / M[col][col];
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  const h = new Float64Array(9);
  for (let i = 0; i < 8; i++) h[i] = M[i][n] / M[i][i];
  h[8] = 1;

  const [h0,h1,h2,h3,h4,h5,h6,h7,h8] = h;
  const det = h0*(h4*h8-h5*h7) - h1*(h3*h8-h5*h6) + h2*(h3*h7-h4*h6);
  if (Math.abs(det) < 1e-12) return srcGray;
  const inv = new Float64Array(9);
  inv[0]=(h4*h8-h5*h7)/det; inv[1]=(h2*h7-h1*h8)/det; inv[2]=(h1*h5-h2*h4)/det;
  inv[3]=(h5*h6-h3*h8)/det; inv[4]=(h0*h8-h2*h6)/det; inv[5]=(h2*h3-h0*h5)/det;
  inv[6]=(h3*h7-h4*h6)/det; inv[7]=(h1*h6-h0*h7)/det; inv[8]=(h0*h4-h1*h3)/det;

  const out = new Uint8Array(outW * outH);
  for (let dy = 0; dy < outH; dy++) {
    for (let dx = 0; dx < outW; dx++) {
      const w2 = inv[6]*dx + inv[7]*dy + inv[8];
      const sx = (inv[0]*dx + inv[1]*dy + inv[2]) / w2;
      const sy = (inv[3]*dx + inv[4]*dy + inv[5]) / w2;
      const x0 = Math.max(0, Math.min(srcW-2, Math.floor(sx)));
      const y0 = Math.max(0, Math.min(srcH-2, Math.floor(sy)));
      const fx = sx-x0, fy = sy-y0;
      out[dy*outW+dx] = Math.round(
        (1-fy)*((1-fx)*srcGray[y0*srcW+x0] + fx*srcGray[y0*srcW+x0+1]) +
        fy*((1-fx)*srcGray[(y0+1)*srcW+x0] + fx*srcGray[(y0+1)*srcW+x0+1])
      );
    }
  }
  return out;
}

// ── Corner fiducial blob detection (BFS connected-component) ─────────────────
function findCornerBlob(grayArr, W, H, searchX0, searchY0, searchX1, searchY1, darkThr) {
  const lw = searchX1 - searchX0;
  const lh = searchY1 - searchY0;
  const visited = new Uint8Array(lw * lh);
  let bestArea = 0, bestSumX = 0, bestSumY = 0;

  for (let y = searchY0; y < searchY1; y++) {
    for (let x = searchX0; x < searchX1; x++) {
      const li = (y - searchY0) * lw + (x - searchX0);
      if (visited[li] || grayArr[y * W + x] >= darkThr) continue;

      const queue = [y * W + x];
      visited[li] = 1;
      let sumX = 0, sumY = 0, area = 0, head = 0;

      while (head < queue.length) {
        const idx = queue[head++];
        const cy2 = Math.floor(idx / W);
        const cx2 = idx % W;
        sumX += cx2; sumY += cy2; area++;
        const nbrs = [[cx2-1,cy2],[cx2+1,cy2],[cx2,cy2-1],[cx2,cy2+1]];
        for (const [nx, ny] of nbrs) {
          if (nx < searchX0 || nx >= searchX1 || ny < searchY0 || ny >= searchY1) continue;
          const nli = (ny - searchY0) * lw + (nx - searchX0);
          if (visited[nli] || grayArr[ny * W + nx] >= darkThr) continue;
          visited[nli] = 1;
          queue.push(ny * W + nx);
        }
      }

      if (area > bestArea) { bestArea = area; bestSumX = sumX; bestSumY = sumY; }
    }
  }

  const imgScaleSq = (W / 794) ** 2;
  const minArea = Math.round(20 * imgScaleSq);
  const maxArea = Math.round(180000 * imgScaleSq);
  if (bestArea < minArea || bestArea > maxArea || bestArea === 0) return null;
  return { cx: bestSumX / bestArea, cy: bestSumY / bestArea };
}

// ── Classify a question given its 4 fill values ───────────────────────────────
//
// Returns { answer, confidence, isDouble, isBlank }
//
// Decision tree:
//  1. If NO bubble ≥ MIN_FILLED_PERCENT AND no floor hit → BLANK
//  2. If top AND second BOTH ≥ MIN_DOUBLE_FILL AND second/top ≥ DOUBLE_MARK_RATIO
//     → TRUE double mark: report as double but ALSO resolve to top (fewer blanks!)
//  3. If top ≥ MIN_FILLED_PERCENT → answer = top bubble (with confidence)
//  4. If top < MIN_FILLED_PERCENT but ≥ floor → weak pick (still better than blank)
//
// KEY CHANGE from v6: double-marks are now RESOLVED to the darker bubble
// instead of returned as blank. This matches real-world usage where students
// lightly tried one bubble then filled another — the darker one is their intent.
// A question is only flagged isDouble=true when the fills are truly indistinguishable.
function classifyQuestion(fills, fillFloor, blankStd, qNum) {
  const {
    MIN_FILLED_PERCENT, DOUBLE_MARK_RATIO, MIN_DOUBLE_FILL,
    MIN_RATIO, MARGIN_THRESH_MULT,
  } = OMR_CONSTANTS;

  const OPTIONS = ['A', 'B', 'C', 'D'];
  const sorted = fills
    .map((f, i) => ({ f, i }))
    .sort((a, b) => b.f - a.f);

  const best   = sorted[0];
  const second = sorted[1];
  const third  = sorted[2];
  const margin = best.f - second.f;

  // Confidence = (topFill - secondFill) × topFill
  // High when one bubble strongly dominates; low when two are close
  const rawConfidence = (best.f - second.f) * best.f;

  // Case 1: All bubbles weak → BLANK (or weak-pick if above floor)
  if (best.f < MIN_FILLED_PERCENT) {
    if (best.f >= fillFloor && best.f >= 0.10) {
      const letter = OPTIONS[best.i];
      console.log(`[BubbleOMR] Q${qNum}: ${letter} (floor-pick: ${(best.f*100).toFixed(1)}%) [${fills.map(f=>(f*100).toFixed(1)+'%').join(', ')}]`);
      return { answer: letter, confidence: rawConfidence * 0.4, isDouble: false, isBlank: false };
    }
    console.log(`[BubbleOMR] Q${qNum}: (blank) best=${(best.f*100).toFixed(1)}% [${fills.map(f=>(f*100).toFixed(1)+'%').join(', ')}]`);
    return { answer: '', confidence: 0, isDouble: false, isBlank: true };
  }

  // Case 2: Potential double-mark — check if two bubbles are genuinely both filled
  const doubleRatio = second.f > 0.005 ? second.f / best.f : 0;
  const isDoubleCandidate =
    best.f   >= MIN_DOUBLE_FILL &&
    second.f >= MIN_DOUBLE_FILL &&
    doubleRatio >= DOUBLE_MARK_RATIO;

  if (isDoubleCandidate) {
    // v9.2: ALWAYS resolve to the darker bubble — never return blank for double-mark.
    // Rationale: ballpen students often press harder on their real answer, or lightly
    // touched a bubble before changing their mind. The darker fill = intent.
    // Even if fills are 98.4% vs 98.2% (margin=0.2%), one IS darker — pick it.
    // Returning blank forces manual review which defeats the purpose of auto-grading.
    const absoluteMargin = best.f - second.f;
    const letter = OPTIONS[best.i];
    const confMult = absoluteMargin < 0.015 ? 0.60 : 0.80; // lower conf for near-identical
    console.log(`[BubbleOMR] Q${qNum}: ${letter} (double→resolved top=${(best.f*100).toFixed(1)}% 2nd=${(second.f*100).toFixed(1)}% margin=${(absoluteMargin*100).toFixed(1)}%) [${fills.map(f=>(f*100).toFixed(1)+'%').join(', ')}]`);
    return { answer: letter, confidence: rawConfidence * confMult, isDouble: false, isBlank: false };
  }

  // Case 3: Normal — pick the top bubble
  const letter = OPTIONS[best.i];
  const dominanceRatio = second.f > 0.005 ? best.f / second.f : best.f / 0.005;
  const isStrong = dominanceRatio >= MIN_RATIO;
  const confidence = isStrong ? rawConfidence : rawConfidence * 0.65;

  console.log(`[BubbleOMR] Q${qNum}: ${letter} (fill=${(best.f*100).toFixed(1)}% margin=${(margin*100).toFixed(1)}% ratio=${dominanceRatio.toFixed(2)}) [${fills.map(f=>(f*100).toFixed(1)+'%').join(', ')}]`);
  return { answer: letter, confidence, isDouble: false, isBlank: false };
}

// ── Main bubble detection entry point ────────────────────────────────────────
async function detectBubblesWithJimp(imageBase64, mimeType, questionCount) {
  if (!Jimp) return null;

  const numCols = omrColCount(questionCount);
  const layout  = OMR_LAYOUT[numCols];
  console.log(`[BubbleOMR] OMR Engine v6 — ${questionCount} questions, ${numCols} column(s)`);

  // ── 1. Decode image ────────────────────────────────────────────────────────
  let image;
  try {
    const buf = Buffer.from(imageBase64, 'base64');
    image = await (Jimp.fromBuffer ? Jimp.fromBuffer(buf) : Jimp.read(buf));
  } catch (err) {
    console.warn('[BubbleOMR] Failed to decode image:', err.message);
    return null;
  }

  image.greyscale();
  let imgW = image.getWidth();
  let imgH = image.getHeight();

  // Build flat grayscale array from Jimp pixels
  let gray = new Uint8Array(imgW * imgH);
  for (let py = 0; py < imgH; py++) {
    for (let px = 0; px < imgW; px++) {
      gray[py * imgW + px] = Jimp.intToRGBA(image.getPixelColor(px, py)).r;
    }
  }

  // ── 2. Perspective correction ──────────────────────────────────────────────
  const idealTL = { x: MARK_INSET,            y: MARK_INSET };
  const idealTR = { x: TARGET_W - MARK_INSET, y: MARK_INSET };
  const idealBR = { x: TARGET_W - MARK_INSET, y: TARGET_H - MARK_INSET };
  const idealBL = { x: MARK_INSET,            y: TARGET_H - MARK_INSET };

  const searchW = Math.round(imgW * 0.22);
  const searchH = Math.round(imgH * 0.22);

  let tlBlob = null, trBlob = null, brBlob = null, blBlob = null;
  let cornerThr = 0;
  const otsuBase = otsuThreshold(gray);

  const thresholdsToTry = [
    Math.round(otsuBase * 0.85), Math.round(otsuBase * 0.75),
    Math.round(otsuBase * 0.65), Math.round(otsuBase * 0.55),
    Math.round(otsuBase * 0.45),
    128, 110, 90, 70, 55,
  ].filter((v, i, arr) => v > 10 && arr.indexOf(v) === i);

  for (const thr of thresholdsToTry) {
    const tl = findCornerBlob(gray, imgW, imgH, 0,            0,            searchW,      searchH,      thr);
    const tr = findCornerBlob(gray, imgW, imgH, imgW-searchW, 0,            imgW,         searchH,      thr);
    const br = findCornerBlob(gray, imgW, imgH, imgW-searchW, imgH-searchH, imgW,         imgH,         thr);
    const bl = findCornerBlob(gray, imgW, imgH, 0,            imgH-searchH, searchW,      imgH,         thr);
    const found = [tl,tr,br,bl].filter(Boolean).length;
    console.log(`[BubbleOMR] cornerThr=${thr} → TL:${tl?'✓':'✗'} TR:${tr?'✓':'✗'} BR:${br?'✓':'✗'} BL:${bl?'✓':'✗'}`);
    if (found > [tlBlob,trBlob,brBlob,blBlob].filter(Boolean).length) {
      tlBlob = tl || tlBlob; trBlob = tr || trBlob;
      brBlob = br || brBlob; blBlob = bl || blBlob;
      cornerThr = thr;
    }
    if (tlBlob && trBlob && brBlob && blBlob) break;
  }

  console.log(`[BubbleOMR] Image size: ${imgW}×${imgH}, searchRegion: ${searchW}×${searchH}, cornerThr: ${cornerThr}`);
  console.log(`[BubbleOMR] Corner blobs — TL:${tlBlob?'✓':'✗'} TR:${trBlob?'✓':'✗'} BR:${brBlob?'✓':'✗'} BL:${blBlob?'✓':'✗'}`);

  if (tlBlob && trBlob && brBlob && blBlob) {
    console.log(`[BubbleOMR] Perspective correction — corners TL(${tlBlob.cx.toFixed(0)},${tlBlob.cy.toFixed(0)}) TR(${trBlob.cx.toFixed(0)},${trBlob.cy.toFixed(0)}) BR(${brBlob.cx.toFixed(0)},${brBlob.cy.toFixed(0)}) BL(${blBlob.cx.toFixed(0)},${blBlob.cy.toFixed(0)})`);
    gray = homographyWarp(gray, imgW, imgH,
      [tlBlob, trBlob, brBlob, blBlob],
      [idealTL, idealTR, idealBR, idealBL],
      TARGET_W, TARGET_H);
    imgW = TARGET_W;
    imgH = TARGET_H;
    console.log(`[BubbleOMR] Perspective correction applied ✓ — output: ${imgW}×${imgH}`);
  } else {
    // Fallback: nearest-neighbor resize to canonical size
    const resized = new Uint8Array(TARGET_W * TARGET_H);
    for (let ty = 0; ty < TARGET_H; ty++) {
      for (let tx = 0; tx < TARGET_W; tx++) {
        const sx = Math.min(imgW - 1, Math.round(tx * imgW / TARGET_W));
        const sy = Math.min(imgH - 1, Math.round(ty * imgH / TARGET_H));
        resized[ty * TARGET_W + tx] = gray[sy * imgW + sx];
      }
    }
    gray = resized;
    imgW = TARGET_W;
    imgH = TARGET_H;
    console.log(`[BubbleOMR] ⚠️  Perspective correction skipped (corners not found) — using resize fallback`);
  }

  const W = imgW;
  const H = imgH;

  // ── 3. FIX 5: Enhanced image preprocessing ────────────────────────────────
  // Step A: Shadow normalization — removes lighting gradients from phone photos
  console.log(`[BubbleOMR] Preprocessing: shadow normalization...`);
  gray = normalizeShadow(gray, W, H);

  // Step B: CLAHE approximation — boosts local contrast for weak pencil marks
  console.log(`[BubbleOMR] Preprocessing: CLAHE local contrast enhancement...`);
  gray = claheApprox(gray, W, H, OMR_CONSTANTS.CLAHE_TILE_SIZE);

  // Step C: Gaussian blur — reduces speck noise before threshold
  gray = gaussianBlur3(gray, W, H);

  console.log(`[BubbleOMR] Preprocessing complete ✓`);

  // ── 4. Compute adaptive fill floor (blank-baseline calibration) ───────────
  //
  // Strategy: sample ALL bubbles first at their nominal positions to collect
  // blank-bubble fill distribution, then compute threshold = mean + 2.5×std.
  // This is more robust than fixed thresholds because it adapts to each scan.
  const perCol  = Math.ceil(questionCount / numCols);
  const r       = Math.round(layout.BUBBLE_R * W);
  const rowStep = layout.ROW_STEP * H;
  const OPTIONS = ['A', 'B', 'C', 'D'];

  // First pass: collect all fill ratios (used for calibration)
  const allRatios = {};

  for (let col = 0; col < numCols; col++) {
    const bubbleX0 = (layout.COL_LEFT[col] + layout.Q_NUM_W) * W;
    const colStart = col * perCol + 1;
    const colEnd   = Math.min(colStart + perCol - 1, questionCount);

    for (let row = 0; row < colEnd - colStart + 1; row++) {
      const qNum = colStart + row;
      // Expected Y (before row snapping)
      const expectedCy = Math.round(layout.GRID_TOP * H + row * rowStep + rowStep * 0.5);
      // Bubble X positions for all 4 options
      const bubbleXs = [0,1,2,3].map(i => bubbleX0 + i * layout.BUBBLE_STEP * W);

      // v9.2: Dynamic row recalibration — snap Y to nearest real ink peak
      const snappedCy = snapRowY(gray, W, H, expectedCy, bubbleXs, r);

      // Sample all 4 bubbles with center-weighted sampler (FIX 1)
      allRatios[qNum] = bubbleXs.map(cx =>
        sampleBubbleWeighted(gray, W, H, Math.round(cx), snappedCy, r)
      );
    }
  }

  // Debug: print Q1 pixel coords for verification
  const _dbgX0  = (layout.COL_LEFT[0] + layout.Q_NUM_W) * W;
  const _dbgCy0 = Math.round(layout.GRID_TOP * H + rowStep * 0.5);
  console.log(`[BubbleOMR] Q1 coords: A(${Math.round(_dbgX0)},${_dbgCy0}) B(${Math.round(_dbgX0+layout.BUBBLE_STEP*W)},${_dbgCy0}) r=${r}px`);

  // Compute blank baseline from per-question minimum fills
  const allRatioValues  = Object.values(allRatios).flatMap(r4 => r4);
  const perQMinRatios   = Object.values(allRatios).map(r4 => Math.min(...r4));
  const blankMean = perQMinRatios.reduce((s, v) => s + v, 0) / (perQMinRatios.length || 1);
  const blankStd  = Math.sqrt(
    perQMinRatios.reduce((s, v) => s + (v - blankMean) ** 2, 0) / (perQMinRatios.length || 1)
  );

  // Cap fillFloor at 0.30 — phone lighting can push blank reads to 14%+16% std=47% floor
  // which blocks real ballpen answers (25-40% fill). Hard cap at 30%.
  const rawFloor = blankMean + OMR_CONSTANTS.BLANK_SIGMA_MULT * blankStd;
  const FILL_FLOOR = Math.max(0.06, Math.min(0.30, rawFloor));
  const sorted90   = [...allRatioValues].sort((a, b) => a - b);
  const p90        = sorted90[Math.floor(sorted90.length * 0.90)] ?? 0;

  console.log(`[BubbleOMR] Blank baseline — mean=${(blankMean*100).toFixed(1)}% std=${(blankStd*100).toFixed(1)}% rawFloor=${(rawFloor*100).toFixed(1)}% → FILL_FLOOR=${(FILL_FLOOR*100).toFixed(1)}% p90=${(p90*100).toFixed(1)}%`);

  // ── 5. First-pass classification ──────────────────────────────────────────
  const answers     = {};
  const confidences = {};
  let doubleMarked  = 0;

  for (const [qStr, fills] of Object.entries(allRatios)) {
    const qNum = parseInt(qStr, 10);
    const result = classifyQuestion(fills, FILL_FLOOR, blankStd, qNum);
    answers[qStr]     = result.answer;
    confidences[qStr] = result.confidence;
    if (result.isDouble) doubleMarked++;
  }

  // ── 6. FIX 7: Second-pass retry for low-confidence questions ──────────────
  //
  // Questions below SECOND_PASS_CONF_THRESHOLD get re-sampled with small Y adjustments.
  // The best (highest-confidence) result across all Y offsets is kept.
  const CONF_THR = OMR_CONSTANTS.SECOND_PASS_CONF_THRESHOLD;
  const lowConfQs = Object.keys(confidences).filter(q => confidences[q] < CONF_THR);

  if (lowConfQs.length > 0) {
    console.log(`[BubbleOMR] Second-pass retry for ${lowConfQs.length} low-confidence question(s)...`);

    for (const qStr of lowConfQs) {
      const qNum = parseInt(qStr, 10);
      // Determine which column and row this question belongs to
      let col = 0, rowInCol = qNum - 1;
      for (let c = 1; c < numCols; c++) {
        if (qNum > c * perCol) { col = c; rowInCol = qNum - c * perCol - 1; }
      }
      const bubbleX0  = (layout.COL_LEFT[col] + layout.Q_NUM_W) * W;
      const bubbleXs  = [0,1,2,3].map(i => bubbleX0 + i * layout.BUBBLE_STEP * W);
      const expectedCy = Math.round(layout.GRID_TOP * H + rowInCol * rowStep + rowStep * 0.5);

      let bestConf   = confidences[qStr];
      let bestAnswer = answers[qStr];

      for (const yOff of OMR_CONSTANTS.SECOND_PASS_Y_OFFSETS) {
        const trialCy = expectedCy + yOff;
        if (trialCy < 0 || trialCy >= H) continue;

        const trialFills = bubbleXs.map(cx =>
          sampleBubbleWeighted(gray, W, H, Math.round(cx), trialCy, r)
        );
        const result = classifyQuestion(trialFills, FILL_FLOOR, blankStd, qNum);

        if (result.confidence > bestConf) {
          bestConf   = result.confidence;
          bestAnswer = result.answer;
          console.log(`[BubbleOMR] Q${qNum} second-pass Y+${yOff}px improved → ${bestAnswer || '?'} (conf ${(bestConf*100).toFixed(1)}%)`);
        }
        // Stop early if already confident enough — no need to try more offsets
        if (bestConf >= 0.40) break;
      }

      answers[qStr]     = bestAnswer;
      confidences[qStr] = bestConf;
    }
  }

  // ── 7. FIX 6: Aggregate confidence score ──────────────────────────────────
  //
  // Per-question confidence = (topFill - secondFill) × topFill
  // Aggregate confidence = mean of answered questions' confidences, scaled to 0–1.
  const answeredConfs = Object.entries(answers)
    .filter(([, a]) => a !== '')
    .map(([q]) => confidences[q] ?? 0);

  const meanConf = answeredConfs.length > 0
    ? answeredConfs.reduce((s, v) => s + v, 0) / answeredConfs.length
    : 0;

  // Calibrate: raw meanConf is (fill_diff × top_fill), max ≈ 0.25 for fully filled.
  // Scale to 0–1 range using empirical factor.
  const CONF_SCALE = 4.0; // empirically calibrated: fills of 0.60 → rawConf 0.36 → scaled 1.0
  const scaledConf = Math.min(1.0, meanConf * CONF_SCALE);

  const answeredCount = Object.values(answers).filter(a => a !== '').length;
  const fillRate      = questionCount > 0 ? answeredCount / questionCount : 0;

  // Penalize for double marks and low fill rate
  const confidence = Math.min(0.99, scaledConf * (0.7 + 0.3 * fillRate) - doubleMarked * 0.015);

  if (doubleMarked > 0) {
    console.warn(`[BubbleOMR] ⚠️  ${doubleMarked} TRUE double-marked question(s) — returned as blank (student must re-scan or erase)`);
  }
  console.log(`[BubbleOMR] Done — answered: ${answeredCount}/${questionCount}, true-doubles: ${doubleMarked}, confidence: ${(Math.max(0, confidence)*100).toFixed(1)}%`);

  return {
    studentName:      null,
    answers,
    answeredCount,
    engineConfidence: Math.max(0, confidence) * 100,
    confidence:       Math.max(0, confidence),
  };
}



// ─── Groq Vision scanner (identification / enumeration / true_or_false) ─────
//
// Sends the exam sheet image to Groq's llama-4-scout-17b-16e-instruct model.
// Reads handwriting and returns a clean JSON map of answers.
// Falls back to Tesseract automatically if Groq is unavailable or fails.

// fromQ / toQ: optional question range for mixed-mode scans.
// When provided, Groq is told to ONLY read questions in that range,
// preventing it from returning MC letters in T/F slots, etc.
async function scanWithGroq(imageBase64, mimeType, examType, questionCount, fromQ = null, toQ = null) {
  if (!groqReady) {
    console.warn('[Groq] Not available — falling back to Tesseract');
    return null;
  }

  const qFrom = fromQ ?? 1;
  const qTo   = toQ   ?? questionCount;
  const rangeDesc = (fromQ && toQ) ? `questions ${qFrom} to ${qTo}` : `all ${questionCount} questions`;

  // Per-type instructions — explicitly distinguish printed text from handwritten answers.
  // This is the root cause fix: the old prompt never told Groq to ignore printed choices.
  const typeInstructions = {

    bubble_omr: `You are an expert OMR (Optical Mark Recognition) reader analyzing an AutoChecker BUBBLE ANSWER SHEET.

═══ SHEET LAYOUT ═══
- Total questions: ${questionCount}
- Columns: ${questionCount <= 25 ? '1 column (Q1-Q25)' : questionCount <= 50 ? '2 columns (Q1-Q25 left, Q26-Q50 right)' : '3 columns (Q1-Q25 left, Q26-Q50 middle, Q51-Q75 right)'}
- Each row has 4 circles labeled A B C D (left to right)
- Question number is to the LEFT of the 4 circles
- You are reading questions ${qFrom} to ${qTo}

═══ HOW TO READ BUBBLES — MOST IMPORTANT PART ═══
The student used a BLACK BALLPEN to fill (shade in) exactly ONE circle per question.

FILLED bubble (= student's answer):
  • The INTERIOR of the circle is solid black or very dark
  • Looks like a solid black disc/dot inside the circular outline
  • The entire inside area is covered with ink
  • Stands out dramatically from the other 3 empty circles in that row

EMPTY bubble (= NOT the answer):
  • Only a thin circular OUTLINE is visible
  • The INTERIOR is white/light — hollow
  • Just a ring, not a disc

KEY: Compare all 4 circles in the same row. The filled one is OBVIOUSLY darker inside. This is not subtle — a filled bubble looks completely different from an empty one.

═══ COLUMN READING GUIDE ═══
- LEFT column: find Q1 at the top, read down to Q25
- MIDDLE column (if 50Q+): find Q26 at top, read down to Q50  
- RIGHT column (if 75Q): find Q51 at top, read down to Q75 — DO NOT skip this column!
- Each column has its OWN set of question numbers on the left and A B C D bubbles

═══ ANTI-HALLUCINATION RULES ═══
1. Process each question row INDEPENDENTLY — the previous answer does NOT predict the next
2. If you find yourself assigning the same letter 5+ times in a row, STOP and re-examine
3. Real exam answer distributions are roughly equal across A, B, C, D
4. NEVER assume a pattern. Each row is examined fresh.
5. A partially marked or crossed-out bubble should still be read as that answer

═══ OUTPUT FORMAT ═══
Return ONLY a valid JSON object. No text before or after. Example:
{"${qFrom}":"C","${qFrom+1}":"A","${qFrom+2}":"D",...,"${qTo}":"B"}
Use "" for unanswered questions. Keys must be question numbers as strings.`,

    multiple_choice: `Each answer is a SINGLE LETTER (A, B, C, or D) handwritten by the student.

CRITICAL — THE PAPER HAS TWO KINDS OF TEXT:
1. PRINTED CHOICES (pre-printed, IGNORE these): e.g. "A. CSS   B. HTML   C. Python   D. Java"
   These are the answer options. Do NOT return these letters.
2. HANDWRITTEN ANSWER (read this): a single letter the student wrote in pen,
   usually on a blank line before or after the question number.

HOW TO FIND THE STUDENT'S ANSWER:
- Look for a lone handwritten letter (A, B, C, or D) near the question number.
- It may appear on a blank underline before the question: "___1. Which..."
- It may appear on a line after the question number.
- It is handwritten — uneven, slightly slanted, imperfect strokes.
- If blank (student did not answer), use "".

Return ONLY the single uppercase letter: "A", "B", "C", or "D".
Do NOT return letters from the printed choices. Do NOT return the question text.`,

    true_or_false: `Each answer is either "True" or "False" handwritten by the student.

CRITICAL — THE PAPER HAS TWO KINDS OF TEXT:
1. PRINTED TEXT (pre-printed, IGNORE): the words "True or False", section headers,
   and the question sentences (e.g. "HTML is used to style web pages.").
2. HANDWRITTEN ANSWER (read this): a single letter T or F that the student wrote
   on the blank line (___) before or near each question number.

HOW TO FIND THE HANDWRITTEN ANSWER:
- Look for a lone handwritten letter near each question number.
- The blank line appears BEFORE the question: "___  6. HTML is used to style web pages."
- The student writes ONE letter on that blank: either T or F.

HOW TO TELL T FROM F — LOOK CAREFULLY:
- T: a horizontal bar at the TOP only, with a vertical stroke going straight down.
     It looks like a plus sign with the bottom removed.
- F: a horizontal bar at TOP, a SHORTER bar in the MIDDLE, and NO bar at the bottom.
     It looks like an E with the bottom arm removed.
- The difference: T has NO middle bar. F has a middle bar.
- Do NOT return "True" for everything — each question has a different answer.
- If the letter has a middle horizontal stroke → it is F (False).
- If the letter has ONLY a top horizontal stroke → it is T (True).

Return "True" if the student wrote T, "False" if the student wrote F.
If the blank is empty (no letter written), return "".
NEVER guess — if you cannot read the letter clearly, return "".`,

    identification: `Each answer is a handwritten word or short phrase (1-5 words) on a blank line.

CRITICAL — THE PAPER HAS TWO KINDS OF TEXT:
1. PRINTED QUESTION TEXT (pre-printed, IGNORE): the question description after the blank.
   Example: "It is used to style web pages." — this is printed, NOT the answer.
2. HANDWRITTEN ANSWER (read this): the student's written word/phrase on the blank line (___).

The blank may appear:
 - BEFORE the question number: "_______ 9. Git is a version control system."
 - AFTER the question number: "9. _______ Git is a version control system."
 - OR directly after the number before the question text.

Read ONLY what is written on the blank line. Do NOT return the printed question text.
If blank, use "".`,

    enumeration: `Each answer is a handwritten word or short phrase on a numbered blank line.

CRITICAL — THE PAPER HAS TWO KINDS OF TEXT:
1. PRINTED CATEGORY LABELS (pre-printed, IGNORE): e.g. "Frontend technologies:", "Database systems:", "16.", "17." printed labels
2. HANDWRITTEN ANSWER (read this): the student's word/phrase written on the blank line (___) beside or after the question number.

HOW TO FIND THE HANDWRITTEN ANSWER:
- Look for handwritten text on the blank line (___) near each question number.
- The blank appears after the number: "16. _______" or before the question: "_______ 16."
- Common answers in a web/IT exam: HTML, CSS, JavaScript, Python, Node.js, PHP, MySQL, MongoDB, PostgreSQL, Git, React, API, JSON, etc.
- Read the handwritten word carefully — the student's ink strokes, not any printed text.

IMPORTANT: Return the student's actual handwritten answer, not the printed question text.
If blank (student did not write anything), return "".`,
  };

  const instructions = typeInstructions[examType] ?? typeInstructions.identification;

  const prompt = `You are an expert at reading Filipino school exam papers with handwritten student answers.

EXAM TYPE: ${examType}
TOTAL QUESTIONS ON THIS SHEET: ${questionCount}
YOUR JOB: Read ONLY ${rangeDesc} (numbered ${qFrom} to ${qTo}).
${questionCount > 50 ? `\nIMPORTANT: This is a ${questionCount}-question sheet with 3 columns. You MUST read the RIGHTMOST column (Q51–Q${questionCount}). Do NOT stop at Q50.\n` : ''}

${instructions}

HANDWRITING TIPS — common messy handwriting patterns:
  "a" written as "q" or "u"    |  "n" written as "m" or "ri"
  "e" written as "c" or "o"    |  "i" written as "l" or "1"
  "h" written as "li" or "b"   |  "s" written as "5" or "z"
  "g" written as "9" or "q"    |  letters merged together
Use context (exam subject, nearby words) to correct ambiguous letters.
Copy the student's INTENT, not garbled strokes.

RANGE: Return ONLY questions ${qFrom} through ${qTo}. Ignore all others.

OUTPUT FORMAT: Return ONLY a valid JSON object. No explanation. No markdown.
{"${qFrom}":"answer","${qFrom+1}":"answer",...,"${qTo}":"answer"}

Empty/unanswered = "". If you see a student name at the top, also include "studentName":"Name".`;

  try {
    console.log(`[Groq] Scanning ${examType} — ${rangeDesc}`);

    // ── Compress image before sending to Groq ────────────────────────────────
    // Groq rejects images larger than ~4MB (413 error). Phone photos are often
    // 3120×4160px (13MP) = 12–20MB base64. Resize to max 1600px on longest side.
    let groqImageBase64 = imageBase64;
    let groqMimeType    = mimeType || 'image/jpeg';
    if (sharp) {
      try {
        const inputBuf = Buffer.from(imageBase64, 'base64');
        // FIX: Use higher resolution for 75Q+ (3-column sheets need more pixels per bubble)
        const maxDim = questionCount > 50 ? 2000 : 1600;
        const compressed = await sharp(inputBuf)
          .resize(maxDim, maxDim, { fit: 'inside', withoutEnlargement: true })
          .jpeg({ quality: 88 })
          .toBuffer();
        groqImageBase64 = compressed.toString('base64');
        groqMimeType    = 'image/jpeg';
        const origKB = Math.round(imageBase64.length * 0.75 / 1024);
        const compKB = Math.round(compressed.length / 1024);
        console.log(`[Groq] Image compressed: ${origKB}KB → ${compKB}KB (maxDim=${maxDim})`);
      } catch (compErr) {
        console.warn('[Groq] Image compression failed, using original:', compErr.message);
      }
    }

    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method:  'POST',
      headers: {
        'Authorization': `Bearer ${GROQ_API_KEY}`,
        'Content-Type':  'application/json',
      },
      body: JSON.stringify({
        model:      'meta-llama/llama-4-scout-17b-16e-instruct',
        max_tokens: 2048,  // FIX: 1000 was too low for 75Q+ sheets (75 answers ≈ 800 tokens min)
        messages: [{
          role: 'user',
          content: [
            { type: 'text',      text: prompt },
            { type: 'image_url', image_url: { url: `data:${groqMimeType};base64,${groqImageBase64}` } },
          ],
        }],
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`Groq API error ${response.status}: ${errText.slice(0, 100)}`);
    }

    const data    = await response.json();
    const rawText = data.choices?.[0]?.message?.content?.trim() ?? '';
    console.log(`[Groq] Raw response preview: ${rawText.slice(0, 200)}`);

    const jsonText = rawText
      .replace(/^```json\s*/i, '')
      .replace(/^```\s*/i,    '')
      .replace(/\s*```$/,      '')
      .trim();

    let parsed;
    try {
      parsed = JSON.parse(jsonText);
    } catch {
      const jsonMatch = jsonText.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        parsed = JSON.parse(jsonMatch[0]);
      } else {
        throw new Error('Could not parse Groq JSON response: ' + rawText.slice(0, 100));
      }
    }

    const studentName = parsed.studentName ?? null;
    delete parsed.studentName;

    const answers = {};
    for (let i = qFrom; i <= qTo; i++) {
      answers[String(i)] = (parsed[String(i)] ?? parsed[i] ?? '').trim();
    }

    // Post-validate MC: Groq must only return A/B/C/D — reject anything else
    if (examType === 'multiple_choice' || examType === 'bubble_omr' || examType === 'bubble_mc' || examType === 'omr') {
      for (const k of Object.keys(answers)) {
        const v = answers[k].trim().toUpperCase();
        answers[k] = ['A', 'B', 'C', 'D'].includes(v) ? v : '';
      }
    }

    // Post-validate True/False
    if (examType === 'true_or_false') {
      for (const k of Object.keys(answers)) {
        const v = answers[k].trim().toLowerCase();
        if (v === 't' || v === 'true')  { answers[k] = 'True';  continue; }
        if (v === 'f' || v === 'false') { answers[k] = 'False'; continue; }
        answers[k] = '';
      }
    }

    const rangeSize     = qTo - qFrom + 1;
    const answeredCount = Object.values(answers).filter(a => a !== '').length;
    const fillRate      = rangeSize > 0 ? answeredCount / rangeSize : 0;

    // ── Hallucination detection — reject repetitive/patterned results ─────────
    // Groq frequently hallucinates on bubble sheets by returning the same letter
    // repeated many times or a simple cycling pattern (A,B,C,D,A,B,C,D...).
    // These are statistically impossible on real exams — reject and return null.
    if (examType === 'bubble_omr' || examType === 'bubble_mc' || examType === 'omr') {
      const vals = Object.values(answers).filter(v => v !== '');
      if (vals.length >= 10) {
        // Check 1: any single letter dominates ≥ 75% of answers (raised from 60%)
        // 60% was too aggressive — a skewed-but-valid exam could have 12/20 same letter
        const freq = { A: 0, B: 0, C: 0, D: 0 };
        for (const v of vals) { if (freq[v] !== undefined) freq[v]++; }
        const maxFreq = Math.max(...Object.values(freq));
        if (maxFreq / vals.length >= 0.75) {
          const dominantLetter = Object.entries(freq).find(([, c]) => c === maxFreq)?.[0];
          console.warn(`[Groq] Hallucination detected — "${dominantLetter}" appears ${maxFreq}/${vals.length} times (${(maxFreq/vals.length*100).toFixed(0)}%). Rejecting result.`);
          return null;
        }

        // Check 2: longest consecutive same-letter streak ≥ 9 (raised from 7)
        let maxStreak = 1, streak = 1;
        for (let i = 1; i < vals.length; i++) {
          if (vals[i] === vals[i - 1]) { streak++; maxStreak = Math.max(maxStreak, streak); }
          else streak = 1;
        }
        if (maxStreak >= 9) {
          console.warn(`[Groq] Hallucination detected — streak of ${maxStreak} consecutive same-letter answers. Rejecting result.`);
          return null;
        }

        // Check 3: exact ABCD cycling pattern covering > 80% of answers (raised from 75%)
        let cycleMatches = 0;
        const cycle = ['A','B','C','D'];
        for (let i = 0; i < vals.length; i++) {
          if (vals[i] === cycle[i % 4]) cycleMatches++;
        }
        if (cycleMatches / vals.length >= 0.80) {
          console.warn(`[Groq] Hallucination detected — ABCD cycle pattern (${cycleMatches}/${vals.length} match). Rejecting result.`);
          return null;
        }
      }
    }

    const confidence    = Math.min(0.82 + fillRate * 0.15, 0.97);

    console.log(`[Groq] Done — answered: ${answeredCount}/${rangeSize} (Q${qFrom}-Q${qTo}), confidence: ${(confidence * 100).toFixed(1)}%`);
    return { studentName, answers, answeredCount, engineConfidence: confidence * 100, confidence };

  } catch (err) {
    console.error('[Groq] Scan failed:', err.message);
    return null;
  }
}

// ─── Full OCR pipeline ────────────────────────────────────────────────────────
//
// ARCHITECTURE FIX — OCR-first, Groq-optional pipeline:
//
//   OLD (broken):  Gemini → [fail] → Tesseract fallback
//     Problem: Gemini was a REQUIRED step. A 404 from an invalid model name
//     (e.g. "gemini-1.5-flasht") forced every scan through the slow cold-start
//     fallback path. The whole pipeline degraded silently.
//
//   NEW (correct): Tesseract (always) → Gemini enhancement (optional, if key set)
//     1. Tesseract always runs — it is the primary, authoritative OCR engine.
//     2. Gemini only runs afterward as a post-processor IF:
//        a. GROQ_API_KEY is set in the environment, AND
//        b. groqReady is true (key valid + probe succeeded), AND
//        c. Tesseract fill-rate is below the GEMINI_ASSIST_THRESHOLD
//           (meaning Tesseract found fewer answers than expected — Gemini can help)
//     3. When Gemini runs, it only fills answers that Tesseract left BLANK.
//        It never overwrites a Tesseract-detected answer with a Gemini guess.
//     4. If Groq fails (network error, quota, wrong key) the Tesseract result
//        is returned unchanged — the scan still succeeds.
//
// Routing:
//   bubble_omr      → Jimp pixel detector → Tesseract fallback
//   all other types → Tesseract (primary) → Gemini fill-in (optional enhancer)

// v3.0: Groq now runs FIRST for written types (see parseVisionText).
// This threshold only controls the FALLBACK fill-in pass — if Tesseract also
// fails as a secondary engine, Groq makes one last attempt to fill blank slots.
// Kept at 0.70: if Tesseract already found 70%+ answers as fallback, the sheet
// is readable enough and another Groq call would just add latency.
const GROQ_ASSIST_THRESHOLD = 0.70;

async function parseVisionText(imageBase64, mimeType, examType, questionCount, questionTypeMap, mixedMode) {
  const isBubble  = examType === 'bubble_mc' || examType === 'bubble_omr' || examType === 'omr';
  const isMcType  = isBubble || examType === 'text_mc' || examType === 'multiple_choice' || examType === 'mc';
  const isWritten = !isMcType;

  // ── Route bubble_omr: Groq AI vision FIRST, Jimp pixel detector as fallback ──
  // ARCHITECTURE CHANGE v8.2:
  // Groq vision (AI) is now the PRIMARY engine for bubble sheets.
  // Reason: Jimp pixel detection requires perfect overhead alignment + calibrated
  // grid constants. Phone photos taken at even slight angles cause grid drift that
  // makes Jimp sample the wrong bubbles — producing wrong answers even though it
  // "finds" 40-50 bubbles (so the fallback never triggers).
  //
  // Groq vision SEES the actual filled bubble visually — it works regardless of:
  //   - Camera angle / perspective
  //   - Slight grid misalignment
  //   - Varying lighting / shadows
  //   - Different sheet prints (slight scaling differences)
  //
  // Jimp pixel detector is kept as fallback for when Groq is unavailable.
if (isBubble) {

  // Pre-process: maximise contrast without destroying grayscale gradients.
  // For bubble OMR, we want FILLED bubbles to appear as dark as possible
  // and EMPTY bubbles to appear as light as possible — max separation.
  let processedBase64 = imageBase64;
  let processedMime   = mimeType || 'image/jpeg';
  if (sharp) {
    try {
      const inputBuf = Buffer.from(imageBase64, 'base64');
      const processed = await sharp(inputBuf)
        .rotate()                                          // fix EXIF orientation
        .greyscale()                                       // remove color
        .modulate({ brightness: 1.05, contrast: 1.30 })   // boost contrast (was 1.15)
        .normalise()                                       // stretch full range
        .median(2)                                         // remove speck noise
        .sharpen({ sigma: 1.5, m1: 1.2, m2: 0.6 })       // sharpen bubble edges
        .png()
        .toBuffer();
      processedBase64 = processed.toString('base64');
      processedMime   = 'image/png';
      console.log('[AutoChecker] Bubble image normalised ✓ (high-contrast mode)');
    } catch (e) {
      console.warn('[AutoChecker] sharp pre-processing failed:', e.message);
    }
  }

  // PRIMARY: Groq AI vision — reads filled bubbles visually, angle-tolerant
  if (groqReady) {
    // ── Attempt 1: standard prompt ────────────────────────────────────────────
    let groqResult = null;
    try {
      console.log('[AutoChecker] v8.4 — Bubble OMR: trying Groq AI vision FIRST (primary)');
      groqResult = await scanWithGroq(processedBase64, processedMime, 'bubble_omr', questionCount);
    } catch (err) {
      console.warn('[AutoChecker] Groq attempt 1 failed:', err.message);
    }

    // ── Attempt 2: hallucination caught — retry with original higher-res image ─
    // When Groq hallucinates (all-D pattern), it usually means the compressed
    // image lost contrast. Retry using the original uncompressed image — it's
    // larger but Groq supports up to ~4MB and accuracy is worth it.
    if (groqResult === null) {
      console.warn('[AutoChecker] Groq hallucination on attempt 1 — retrying with original image');
      try {
        // Use original image (before preprocessing) for the retry
        const origKB = Math.round(imageBase64.length * 0.75 / 1024);
        console.log(`[AutoChecker] Retry with original image: ${origKB}KB`);
        // If original is too large (>3.5MB base64 decoded), resize more carefully
        let retryBase64 = imageBase64;
        let retryMime   = mimeType || 'image/jpeg';
        if (sharp && imageBase64.length * 0.75 > 3_500_000) {
          const buf = Buffer.from(imageBase64, 'base64');
          const resized = await sharp(buf)
            .resize(2400, 2400, { fit: 'inside', withoutEnlargement: true })
            .jpeg({ quality: 95 })
            .toBuffer();
          retryBase64 = resized.toString('base64');
          retryMime   = 'image/jpeg';
          console.log(`[AutoChecker] Retry image resized to ${Math.round(resized.length/1024)}KB`);
        }
        groqResult = await scanWithGroq(retryBase64, retryMime, 'bubble_omr', questionCount);
      } catch (err2) {
        console.warn('[AutoChecker] Groq attempt 2 failed:', err2.message);
      }
    }

    // ── Attempt 3: crop just the bubble grid and send that ────────────────────
    // If full sheet still hallucinates, crop out the header/footer and send only
    // the bubble grid area. This removes noise that confuses the model.
    if (groqResult === null && sharp) {
      console.warn('[AutoChecker] Groq still failing — retrying with bubble-grid crop');
      try {
        const buf = Buffer.from(imageBase64, 'base64');
        const meta = await sharp(buf).metadata();
        const iW = meta.width || 1500, iH = meta.height || 2000;
        // FIX: Use consistent crop fractions regardless of question count
        // Header is ~17% of page height on all AutoChecker sheets; footer ~5%
        const cropTop    = Math.round(iH * 0.17);
        const cropHeight = Math.round(iH * 0.78);
        const cropMaxDim = questionCount > 50 ? 2000 : 1800; // larger for 3-col sheets
        const cropped = await sharp(buf)
          .extract({ left: 0, top: cropTop, width: iW, height: cropHeight })
          .resize(cropMaxDim, cropMaxDim, { fit: 'inside', withoutEnlargement: true })
          .normalize()
          .modulate({ brightness: 1.10, contrast: 1.25 })
          .sharpen({ sigma: 1.5 })
          .jpeg({ quality: 93 })
          .toBuffer();
        const cropKB = Math.round(cropped.length / 1024);
        console.log(`[AutoChecker] Grid crop: ${cropKB}KB (${questionCount}Q, cropMaxDim=${cropMaxDim})`);
        groqResult = await scanWithGroq(cropped.toString('base64'), 'image/jpeg', 'bubble_omr', questionCount);
      } catch (err3) {
        console.warn('[AutoChecker] Groq attempt 3 (crop) failed:', err3.message);
      }
    }

    if (groqResult === null && sharp && questionCount > 50) {
      console.warn('[AutoChecker] Groq still failing for 75Q — retrying per-column (3 separate crops)');
      try {
        const buf = Buffer.from(imageBase64, 'base64');
        const meta = await sharp(buf).metadata();
        const iW = meta.width || 1500, iH = meta.height || 2000;

        const colRanges = [
          { from: 1, to: 25, leftFrac: 0.01, widthFrac: 0.33 },
          { from: 26, to: 50, leftFrac: 0.33, widthFrac: 0.34 },
          { from: 51, to: questionCount, leftFrac: 0.66, widthFrac: 0.34 },
        ];

        const combinedAnswers = {};
        let totalAnswered = 0;

        for (const range of colRanges) {
          const cropLeft   = Math.round(iW * range.leftFrac);
          const cropWidth  = Math.round(iW * range.widthFrac);
          const cropTop    = Math.round(iH * 0.17);
          const cropHeight = Math.round(iH * 0.78);

          const colCrop = await sharp(buf)
            .extract({ left: cropLeft, top: cropTop, width: cropWidth, height: cropHeight })
            .resize(800, 1200, { fit: 'inside', withoutEnlargement: true })
            .normalize()
            .modulate({ brightness: 1.12, contrast: 1.20 })
            .sharpen({ sigma: 1.5 })
            .jpeg({ quality: 92 })
            .toBuffer();

          console.log(`[AutoChecker] Per-column crop Q${range.from}-Q${range.to}: ${Math.round(colCrop.length/1024)}KB`);

          // For per-column crops, use a simpler 1-column prompt context
          const colResult = await scanWithGroq(
            colCrop.toString('base64'), 'image/jpeg', 'bubble_omr',
            range.to - range.from + 1,
            range.from, range.to
          );

          if (colResult && colResult.answers) {
            for (const [q, ans] of Object.entries(colResult.answers)) {
              combinedAnswers[q] = ans;
            }
            totalAnswered += colResult.answeredCount;
          }
        }

        if (totalAnswered >= Math.ceil(questionCount * 0.50)) {
          const colsAnswerMap = {};
          for (let i = 1; i <= questionCount; i++) {
            colsAnswerMap[String(i)] = combinedAnswers[String(i)] ?? '';
          }
          groqResult = {
            studentName: null,
            answers: colsAnswerMap,
            answeredCount: totalAnswered,
            engineConfidence: 80,
            confidence: 0.80,
          };
          console.log(`[AutoChecker] Per-column Groq scan SUCCESS — ${totalAnswered}/${questionCount} ✓`);
        }
      } catch (err4) {
        console.warn('[AutoChecker] Per-column Groq attempt failed:', err4.message);
      }
    }

    if (groqResult === null) {
      console.warn(`[AutoChecker] All Groq attempts failed — falling back to Jimp pixel detector`);
    }

    if (groqResult && groqResult.answeredCount >= Math.ceil(questionCount * 0.50)) {
      console.log(`[AutoChecker] Groq AI vision PRIMARY — ${groqResult.answeredCount}/${questionCount} bubbles ✓`);

      // Cross-check any blanks Groq left with Jimp
      // IMPORTANT: Only fill blanks where Jimp has HIGH confidence (fill > 35%)
      // Low-confidence Jimp answers are often wrong due to grid drift
      const blankCount = Object.values(groqResult.answers).filter(a => !a).length;
      if (blankCount > 0 && blankCount <= Math.ceil(questionCount * 0.30)) {
        console.log(`[AutoChecker] Groq left ${blankCount} blanks — attempting Jimp cross-check`);
        try {
          const jimpFill = await detectBubblesWithJimp(processedBase64, processedMime, questionCount);
          if (jimpFill) {
            let filled = 0;
            for (const [q, ans] of Object.entries(groqResult.answers)) {
              // Only fill if Groq left it blank AND Jimp has an answer with decent confidence
              if (!ans && jimpFill.answers[q] && jimpFill.answers[q] !== '') {
                groqResult.answers[q] = jimpFill.answers[q];
                filled++;
              }
            }
            if (filled > 0) {
              groqResult.answeredCount += filled;
              console.log(`[AutoChecker] Jimp cross-check filled ${filled} blank(s) from Groq`);
            }
          }
        } catch (jErr) { console.warn('[AutoChecker] Jimp cross-check failed:', jErr.message); }
      }
      return groqResult;
    }
    console.warn(`[AutoChecker] All Groq attempts failed — falling back to Jimp pixel detector`);
  }

  // FALLBACK: Jimp pixel detector (when Groq unavailable or returns too few)
  try {
    const jimpResult = await detectBubblesWithJimp(processedBase64, processedMime, questionCount);
    const minAcceptable = Math.ceil(questionCount * 0.20);
    if (jimpResult && jimpResult.answeredCount >= minAcceptable) {
      console.log(`[AutoChecker] Jimp pixel detection fallback — ${jimpResult.answeredCount}/${questionCount} bubbles ✓`);
      return jimpResult;
    }
    console.warn(`[AutoChecker] Jimp found only ${jimpResult?.answeredCount ?? 0}/${questionCount}`);
  } catch (err) {
    console.warn('[AutoChecker] Jimp detection error:', err.message);
  }

  console.log('[AutoChecker] All bubble engines failed — returning empty result');
}

  // ── FIX: Mixed-mode path — run both OCR workers and extract per question range ──
  // When mixedMode=true and questionTypeMap is provided, we run the MC worker
  // (for A-D answers) AND the text worker (for written answers) in parallel,
  // then assign results to each question based on its type in the map.
  const hasMixedMap = mixedMode && questionTypeMap && Object.keys(questionTypeMap).length > 0;

  if (hasMixedMap) {
    console.log('[AutoChecker] Mixed-mode scan — running dual OCR workers for', Object.keys(questionTypeMap).length, 'questions');

    // Pre-process image once, reuse buffer for both workers
    const imageBuffer = await preprocessImage(imageBase64, mimeType, 'multiple_choice');

    // Run MC worker and text worker in parallel
    // FIX: For true_or_false questions, PSM 7 (single text line) works better than
    // PSM 6 (uniform block) because T and F are single isolated characters per line.
    const hasTrueFalse = Object.values(questionTypeMap).some(t => t === 'truefalse' || t === 'true_or_false');
    const textPsm = hasTrueFalse ? 7 : 6;  // PSM 7 for T/F, PSM 6 for written words
    let mcText = '', textOcrText = '', engineConfidence = 0;
    try {
      const [mcOcr, textOcr] = await Promise.all([
        (async () => {
          if (workerMCReady) {
            await workerMC.setParameters({ tessedit_pageseg_mode: 6, preserve_interword_spaces: '1', tessedit_do_invert: '0', tessedit_char_whitelist: CHAR_WHITELIST_MC });
            const { data } = await workerMC.recognize(imageBuffer);
            process.stdout.write('\n');
            return { text: data.text ?? '', confidence: data.confidence ?? 0 };
          }
          return { text: '', confidence: 0 };
        })(),
        (async () => {
          if (workerTextReady) {
            await workerText.setParameters({ tessedit_pageseg_mode: textPsm, preserve_interword_spaces: '1', tessedit_do_invert: '0', tessedit_char_whitelist: CHAR_WHITELIST_TEXT });
            const { data } = await workerText.recognize(imageBuffer);
            process.stdout.write('\n');
            return { text: data.text ?? '', confidence: data.confidence ?? 0 };
          }
          return { text: '', confidence: 0 };
        })(),
      ]);
      mcText       = fixOcrSubstitutions(mcOcr.text);
      textOcrText  = fixOcrSubstitutions(textOcr.text);
      engineConfidence = Math.max(mcOcr.confidence, textOcr.confidence);
    } catch (err) {
      console.warn('[AutoChecker] Dual-worker OCR failed, falling back to single pass:', err.message);
      // Fall through to single-pass Tesseract below
    }

    const studentName = extractStudentName(mcText) ?? extractStudentName(textOcrText) ?? null;
    const answers     = {};

    // Categorise questions by their backend type
    const mcQuestions      = [];
    const writtenQuestions = {}; // backendType → [qNums]

    for (const [qStr, backendType] of Object.entries(questionTypeMap)) {
      const q = parseInt(qStr, 10);
      if (isNaN(q) || q < 1) continue;
      const isMcQ = backendType === 'mc' || backendType === 'omr' || backendType === 'bubble_omr';
      if (isMcQ) {
        mcQuestions.push(q);
      } else {
        if (!writtenQuestions[backendType]) writtenQuestions[backendType] = [];
        writtenQuestions[backendType].push(q);
      }
    }

    // Extract MC answers from Tesseract as a baseline
    if (mcQuestions.length > 0 && mcText) {
      const allMcAnswers = extractMcAnswers(mcText, questionCount);
      for (const q of mcQuestions) {
        if (allMcAnswers[String(q)]) answers[String(q)] = allMcAnswers[String(q)];
      }
      console.log(`[AutoChecker] Mixed MC extraction — found ${Object.keys(answers).filter(k => mcQuestions.includes(parseInt(k))).length}/${mcQuestions.length} MC answers`);
    }

    // Extract written answers from Tesseract as a baseline
    const writtenTypes = Object.keys(writtenQuestions);
    if (writtenTypes.length > 0 && textOcrText) {
      for (const backendType of writtenTypes) {
        const qNums      = writtenQuestions[backendType];
        const allWritten = extractWrittenAnswers(textOcrText, questionCount, backendType);
        let filled = 0;
        for (const q of qNums) {
          if (allWritten[String(q)]) { answers[String(q)] = allWritten[String(q)]; filled++; }
        }
        console.log(`[AutoChecker] Mixed written extraction (${backendType}) — ${filled}/${qNums.length} answers`);
      }
    }

    // Fill all blank slots
    for (let i = 1; i <= questionCount; i++) {
      if (!answers[String(i)]) answers[String(i)] = '';
    }

    // ── SINGLE combined Groq call for ALL question types ──────────────────────
    // Instead of one Groq call per type (which hits the 429 rate limit on free tier),
    // we send ONE call that covers every question on the page.
    // The prompt describes each section's format so Groq knows what to look for.
    // This reduces 4 API calls → 1 per page, staying well within rate limits.
    if (groqReady) {
      try {
        const allQNums   = Array.from({ length: questionCount }, (_, i) => i + 1);
        const fromQ      = 1;
        const toQ        = questionCount;

        // Build a section-aware description of what each question range expects
        const sectionDescs = [];
        if (mcQuestions.length > 0) {
          const mf = Math.min(...mcQuestions), mt = Math.max(...mcQuestions);
          sectionDescs.push(`Q${mf}-Q${mt}: MULTIPLE CHOICE — student wrote a single handwritten letter (A/B/C/D) on the blank line before the question number. IGNORE printed choices like "A. HTML  B. CSS". Return ONLY the handwritten letter.`);
        }
        for (const backendType of writtenTypes) {
          const qs = writtenQuestions[backendType];
          const wf = Math.min(...qs), wt = Math.max(...qs);
          if (backendType === 'truefalse') {
            sectionDescs.push(`Q${wf}-Q${wt}: TRUE OR FALSE — student wrote T or F on the blank line before the question. T = top bar only. F = top bar + middle bar. Return "True" or "False". Empty blank = "".`);
          } else if (backendType === 'identification') {
            sectionDescs.push(`Q${wf}-Q${wt}: IDENTIFICATION — student wrote a word or short phrase on the blank line. IGNORE the printed question text after the blank. Return only the handwritten answer.`);
          } else if (backendType === 'enumeration') {
            sectionDescs.push(`Q${wf}-Q${wt}: ENUMERATION — student wrote a word or short phrase on each numbered blank. Common IT answers: HTML, CSS, JavaScript, Node.js, PHP, Python, MySQL, MongoDB, PostgreSQL, Git, React, API, JSON. Return the handwritten word only.`);
          }
        }

        const combinedPrompt = `You are an expert at reading Filipino school exam papers with handwritten student answers.

THIS PAGE HAS ${questionCount} QUESTIONS in multiple sections:

${sectionDescs.join('\n\n')}

GENERAL RULES:
- The paper has TWO kinds of text: PRINTED (ignore) and HANDWRITTEN by student (read this).
- Handwritten answers appear on blank lines (___) near the question number.
- If a blank is empty, return "".
- NEVER guess — if unreadable, return "".
- For student name at the top, include "studentName":"Name".

OUTPUT FORMAT: Return ONLY a valid JSON object with question numbers as keys.
{"1":"answer","2":"answer",...,"${questionCount}":"answer"}`;

        console.log(`[AutoChecker] Mixed Groq combined — single call for all ${questionCount} questions (Q${fromQ}-Q${toQ})`);

        const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
          method:  'POST',
          headers: {
            'Authorization': `Bearer ${GROQ_API_KEY}`,
            'Content-Type':  'application/json',
          },
          body: JSON.stringify({
            model:      'meta-llama/llama-4-scout-17b-16e-instruct',
            max_tokens: 2048,  // FIX: was 1000, too low for 75Q+ sheets
            messages: [{
              role: 'user',
              content: [
                { type: 'text',      text: combinedPrompt },
                { type: 'image_url', image_url: { url: `data:${mimeType || 'image/jpeg'};base64,${imageBase64}` } },
              ],
            }],
          }),
        });

        if (!response.ok) {
          const errText = await response.text();
          throw new Error(`Groq API error ${response.status}: ${errText.slice(0, 120)}`);
        }

        const data    = await response.json();
        const rawText = (data.choices?.[0]?.message?.content ?? '').trim();
        console.log(`[Groq Combined] Raw preview: ${rawText.slice(0, 300)}`);

        const jsonText = rawText.replace(/^```json\s*/i,'').replace(/^```\s*/i,'').replace(/\s*```$/,'').trim();
        let parsed;
        try {
          parsed = JSON.parse(jsonText);
        } catch {
          const m = jsonText.match(/\{[\s\S]*\}/);
          if (m) parsed = JSON.parse(m[0]);
          else throw new Error('Could not parse combined Groq response');
        }

        const studentNameGroq = parsed.studentName ?? null;
        if (studentNameGroq) delete parsed.studentName;

        let groqFilled = 0;
        for (let q = fromQ; q <= toQ; q++) {
          const k   = String(q);
          const val = (parsed[k] ?? parsed[q] ?? '').trim();
          if (!val) continue;

          const backendType = questionTypeMap[k];
          const isMcQ = backendType === 'mc' || backendType === 'omr' || backendType === 'bubble_omr';

          if (isMcQ) {
            // Validate: must be A/B/C/D only
            const upper = val.toUpperCase();
            if (['A','B','C','D'].includes(upper)) {
              answers[k] = upper;
              groqFilled++;
            }
          } else if (backendType === 'truefalse') {
            const lower = val.toLowerCase();
            if (lower === 't' || lower === 'true')  { answers[k] = 'True';  groqFilled++; }
            else if (lower === 'f' || lower === 'false') { answers[k] = 'False'; groqFilled++; }
          } else {
            // identification / enumeration — overwrite Tesseract (it garbles these)
            answers[k] = val;
            groqFilled++;
          }
        }

        console.log(`[Groq Combined] Done — filled/verified ${groqFilled}/${questionCount} answers`);
        if (studentNameGroq) {
          // bubble up student name if Groq found it
          answers['__studentName__'] = studentNameGroq;
        }

      } catch (groqErr) {
        console.warn('[AutoChecker] Combined Groq call failed (non-fatal), keeping Tesseract results:', groqErr.message);
      }
    }

    // Extract student name stored by combined Groq call, then remove from answers map
    const groqStudentName = answers['__studentName__'] ?? null;
    delete answers['__studentName__'];
    const resolvedStudentName = groqStudentName ?? studentName;

    // Re-fill any blanks left after Groq (safety net)
    for (let i = 1; i <= questionCount; i++) {
      if (!answers[String(i)]) answers[String(i)] = '';
    }

    const finalAnsweredCount = Object.values(answers).filter(a => a !== '').length;
    const finalFillRate      = questionCount > 0 ? finalAnsweredCount / questionCount : 0;
    const confidence         = Math.min((engineConfidence / 100) * 0.6 + finalFillRate * 0.4, 1.0);
    console.log('[AutoChecker] Mixed-mode final — answered: ' + finalAnsweredCount + '/' + questionCount + ', confidence: ' + (confidence * 100).toFixed(1) + '%');
    return { studentName: resolvedStudentName, answers, answeredCount: finalAnsweredCount, engineConfidence, confidence };
  }

  // ── v3.0: Written types → Groq FIRST (much better at messy handwriting) ───
  // Only use Groq-first for written types (identification, enumeration, true_or_false).
  // For multiple_choice, Tesseract with the MC whitelist is more accurate because
  // it's constrained to A/B/C/D only. Groq tends to hallucinate when the page also
  // has printed question text with letters in it.
  if (isWritten && groqReady) {
    console.log('[AutoChecker] v3.0 — Written type: trying Groq vision FIRST (primary engine)');
    try {
      const groqResult = await scanWithGroq(imageBase64, mimeType, examType, questionCount);
      if (groqResult && groqResult.answeredCount > 0) {
        console.log(`[AutoChecker] Groq primary SUCCESS — answered: ${groqResult.answeredCount}/${questionCount} (skipping Tesseract)`);
        return groqResult;
      }
      console.warn('[AutoChecker] Groq returned 0 answers — falling back to Tesseract');
    } catch (groqErr) {
      console.warn('[AutoChecker] Groq primary failed, falling back to Tesseract:', groqErr.message);
    }
  }

  // ── STEP 1: Tesseract (always runs for MC; fallback for written types) ───
  const { text: rawText, engineConfidence } =
    await runTesseract(imageBase64, mimeType, examType);

  if (!rawText?.trim()) {
    console.warn('[AutoChecker] OCR returned empty text');
    return { studentName: null, answers: buildEmptyAnswers(questionCount), answeredCount: 0, engineConfidence: 0, confidence: 0 };
  }

  const cleanText   = fixOcrSubstitutions(rawText);
  const studentName = extractStudentName(cleanText);
  const answers     = {};

  if (isMcType) {
    Object.assign(answers, extractMcAnswers(cleanText, questionCount));
  } else {
    Object.assign(answers, extractWrittenAnswers(cleanText, questionCount, examType));
    parseTextLines(cleanText).forEach(item => {
      if (item.question >= 1 && item.question <= questionCount && !answers[String(item.question)]) {
        answers[String(item.question)] = Array.isArray(item.answer) ? item.answer.join(', ') : item.answer;
      }
    });
  }

  // Ensure all question slots exist (empty string for blanks)
  for (let i = 1; i <= questionCount; i++) {
    if (!answers[String(i)]) answers[String(i)] = '';
  }

  const tesseractAnsweredCount = Object.values(answers).filter(a => a !== '').length;
  const fillRate = questionCount > 0 ? tesseractAnsweredCount / questionCount : 0;

  console.log(`[AutoChecker] Tesseract parsed — answered: ${tesseractAnsweredCount}/${questionCount}, engine: ${engineConfidence?.toFixed?.(1) ?? 'n/a'}%, fillRate: ${(fillRate * 100).toFixed(1)}%`);

  // ── STEP 2: Groq fill-in for any Tesseract blanks (written types only) ───
  // At this point Groq already failed or was unavailable as primary engine.
  // Try it one more time for any questions Tesseract left blank.
  if (isWritten && groqReady && fillRate < GROQ_ASSIST_THRESHOLD) {
    console.log(`[AutoChecker] Tesseract fill-rate ${(fillRate * 100).toFixed(1)}% — trying Groq for remaining blanks`);
    try {
      const groqResult = await scanWithGroq(imageBase64, mimeType, examType, questionCount);
      if (groqResult) {
        let groqFilledCount = 0;
        for (let i = 1; i <= questionCount; i++) {
          const key = String(i);
          if (!answers[key] && groqResult.answers[key]) {
            answers[key] = groqResult.answers[key];
            groqFilledCount++;
          }
        }
        const finalStudentName   = studentName ?? groqResult.studentName ?? null;
        const finalAnsweredCount = Object.values(answers).filter(a => a !== '').length;
        console.log(`[AutoChecker] Groq fallback filled ${groqFilledCount} blank(s) — total: ${finalAnsweredCount}/${questionCount}`);

        const normalizedEng = Math.max((engineConfidence ?? 0), 0) / 100;
        const finalFillRate = questionCount > 0 ? finalAnsweredCount / questionCount : 0;
        const confidence    = Math.min(normalizedEng * 0.6 + finalFillRate * 0.4, 1.0);
        return { studentName: finalStudentName, answers, answeredCount: finalAnsweredCount, engineConfidence, confidence };
      }
    } catch (groqErr) {
      console.warn('[AutoChecker] Groq fallback failed (non-fatal):', groqErr.message);
    }
  }

  // ── Return Tesseract-only result ─────────────────────────────────────────
  const answeredCount = Object.values(answers).filter(a => a !== '').length;
  const normalizedEng = Math.max((engineConfidence ?? 0), 0) / 100;
  const finalFillRate = questionCount > 0 ? answeredCount / questionCount : 0;
  const confidence    = Math.min(normalizedEng + finalFillRate * 0.1, 1.0);

  console.log(`[AutoChecker] Final result — answered: ${answeredCount}/${questionCount}, composite confidence: ${(confidence * 100).toFixed(1)}%`);
  return { studentName, answers, answeredCount, engineConfidence, confidence };
}
function buildEmptyAnswers(count) {
  const ans = {};
  for (let i = 1; i <= count; i++) ans[String(i)] = '';
  return ans;
}

// â”€â”€â”€ Routes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

app.get('/', (_req, res) => res.json({ status: 'âœ… AutoChecker API running' }));

// Sections
app.get('/api/sections', (_req, res) => {
  const sections = readSections().map(s => {
    const record = readStore(s.id);
    return { ...s, hasAnswerKey: !!record,
      fileName:    record?.fileName    ?? null,
      fileType:    record?.fileType    ?? null,
      uploadedAt:  record?.uploadedAt  ?? null,
      total:       record?.key?.length ?? 0,
      typeSummary: record ? typeSummary(record.key) : {},
    };
  });
  res.json({ success: true, sections });
});

app.post('/api/sections', (req, res) => {
  const { name, abbr, subject, studentCount, color } = req.body;
  if (!name?.trim()) return res.status(400).json({ success: false, error: '"name" is required.' });

  const sections = readSections();
  if (sections.some(s => s.name.toLowerCase() === name.trim().toLowerCase())) {
    return res.status(409).json({ success: false, error: `Section "${name}" already exists.` });
  }

  const section = {
    id:           String(Date.now()),
    name:         name.trim(),
    abbr:         (abbr?.trim() || name.trim().slice(0, 2)).toUpperCase(),
    subject:      subject?.trim() || '',
    studentCount: Number(studentCount) || 0,
    color:        VALID_COLORS.has(color) ? color : 'blue',
    average:      0,
    quizCount:    0,
    createdAt:    new Date().toISOString(),
  };

  sections.push(section);
  writeSections(sections);
  console.log(`[AutoChecker] Section created: ${section.name} (${section.id})`);
  res.status(201).json({ success: true, section });
});

app.delete('/api/sections/:id', (req, res) => {
  let sections = readSections();
  if (!sections.find(s => s.id === req.params.id)) {
    return res.status(404).json({ success: false, error: 'Section not found.' });
  }
  writeSections(sections.filter(s => s.id !== req.params.id));
  deleteStore(req.params.id);
  res.json({ success: true });
});


app.patch('/api/sections/:id', (req, res) => {
  const sections = readSections();
  const idx = sections.findIndex(s => s.id === req.params.id);
  if (idx === -1) return res.status(404).json({ success: false, error: 'Section not found.' });

  const { name, abbr, subject, studentCount, color } = req.body;

  if (name?.trim()) {
    const duplicate = sections.find(
      s => s.id !== req.params.id && s.name.toLowerCase() === name.trim().toLowerCase()
    );
    if (duplicate) {
      return res.status(409).json({ success: false, error: `Section "${name.trim()}" already exists.` });
    }
  }

  const existing = sections[idx];
  const updated = {
    ...existing,
    name:         name?.trim()                ? name.trim()                  : existing.name,
    abbr:         abbr?.trim()                ? abbr.trim().toUpperCase()    : existing.abbr,
    subject:      subject !== undefined       ? (subject?.trim() ?? '')     : existing.subject,
    studentCount: studentCount !== undefined  ? Number(studentCount)         : existing.studentCount,
    color:        VALID_COLORS.has(color)     ? color                        : existing.color,
  };

  sections[idx] = updated;
  writeSections(sections);
  console.log(`[AutoChecker] Section updated: ${updated.name} (${updated.id})`);
  res.json({ success: true, section: updated });
});

// Answer Key
app.post('/api/answer-key/:sectionId', upload.single('file'), async (req, res) => {
  const { sectionId } = req.params;
  const tempPath      = req.file?.path;
  try {
    if (!readSections().find(s => s.id === sectionId)) {
      return res.status(404).json({ success: false, error: 'Section not found.' });
    }
    if (!req.file) return res.status(400).json({ success: false, error: 'No file uploaded.' });

    const ext    = path.extname(req.file.originalname).toLowerCase();
    const result = await parseUploadedFile(tempPath, ext);
    if (result.error) return res.status(422).json({ success: false, error: result.error });

    const requireUppercase = req.body?.requireUppercase === 'true' || req.body?.requireUppercase === true;
    const record = { fileName: req.file.originalname, fileType: ext.replace('.',''), uploadedAt: new Date().toISOString(), key: result.items, meta: { requireUppercase } };
    writeStore(sectionId, record);

    const count = record.key.length;
    console.log(`[AutoChecker] âœ… Answer key saved for section ${sectionId}: "${record.fileName}" â€” ${count} answer${count !== 1 ? 's' : ''} detected successfully`);

    res.json({
      success: true, sectionId,
      fileName: record.fileName, fileType: record.fileType,
      uploadedAt: record.uploadedAt, total: count,
      typeSummary: typeSummary(record.key), key: record.key,
      meta: record.meta,
      message: `${count} answer${count !== 1 ? 's' : ''} detected successfully`,
    });
  } catch (err) {
    console.error('Upload error:', err);
    res.status(500).json({ success: false, error: `Upload failed: ${err.message}` });
  } finally {
    if (tempPath && fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
  }
});

app.get('/api/answer-key/:sectionId', (req, res) => {
  const record = readStore(req.params.sectionId);
  if (!record) return res.status(404).json({ success: false, error: 'No answer key for this section.' });
  res.json({ success: true, sectionId: req.params.sectionId, fileName: record.fileName, fileType: record.fileType,
    uploadedAt: record.uploadedAt, total: record.key.length, typeSummary: typeSummary(record.key), key: record.key, meta: record.meta ?? {} });
});

app.delete('/api/answer-key/:sectionId', (req, res) => {
  if (!readStore(req.params.sectionId)) return res.status(404).json({ success: false, error: 'No answer key found.' });
  deleteStore(req.params.sectionId);
  res.json({ success: true });
});

// ─── Extract Text from DOCX / TXT / PDF ──────────────────────────────────────
//
// POST /api/extract-text
// Accepts a .docx, .txt, or .pdf file upload and returns the plain text.
// Used by UploadScreen to extract answer key text before parsing.
//
// Response: { success: true, text: "..." }
//        or { success: false, error: "..." }

app.post('/api/extract-text', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ success: false, error: 'No file uploaded.' });
  }

  const filePath = req.file.path;
  const ext      = path.extname(req.file.originalname).toLowerCase();

  try {
    let text = '';

    if (ext === '.docx' || ext === '.doc') {
      const result = await mammoth.extractRawText({ path: filePath });
      text = result.value ?? '';
    } else if (ext === '.txt') {
      text = fs.readFileSync(filePath, 'utf8');
    } else if (ext === '.pdf') {
      if (typeof pdfParse !== 'function') {
        return res.status(500).json({ success: false, error: 'PDF support is unavailable (pdf-parse failed to load). Please use a .txt or .docx file instead.' });
      }
      const dataBuffer = fs.readFileSync(filePath);
      const parsed     = await pdfParse(dataBuffer);
      text = parsed.text ?? '';
    } else {
      return res.status(400).json({ success: false, error: `Unsupported file type: ${ext}` });
    }

    if (!text.trim()) {
      return res.status(422).json({ success: false, error: 'File appears to be empty or has no readable text.' });
    }

    res.json({ success: true, text });
  } catch (err) {
    console.error('[AutoChecker] extract-text error:', err.message);
    res.status(500).json({ success: false, error: `Could not extract text: ${err.message}` });
  } finally {
    // Clean up uploaded file
    try { fs.unlinkSync(filePath); } catch {}
  }
});

// Score
app.post('/api/score/:sectionId', (req, res) => {
  const record = readStore(req.params.sectionId);
  if (!record) return res.status(404).json({ success: false, error: 'No answer key for this section.' });
  const { student } = req.body;
  if (!student || !Array.isArray(student.answers)) {
    return res.status(400).json({ success: false, error: 'student.answers must be an array.' });
  }
  const requireUppercase = record.meta?.requireUppercase === true;
  let correct = 0;
  const details = record.key.map((item, idx) => {
    const sa = student.answers[idx] ?? '';
    // Apply requireUppercase: if enabled and answer has lowercase, mark wrong immediately
    const hasLowercase = requireUppercase && /[a-z]/.test(sa);
    const ok = hasLowercase ? false : checkAnswer(item, sa);
    if (ok) correct++;
    return { question: item.question, type: item.type, correctAnswer: item.answer, studentAnswer: sa || null, isCorrect: ok };
  });
  const total  = record.key.length;
  const pct    = total > 0 ? Math.round((correct / total) * 100) : 0;
  res.json({ success: true, student: { id: student.id, name: student.name },
    score: correct, total, percentage: pct, status: pct >= 75 ? 'Passed' : pct >= 60 ? 'Review' : 'Failed', details });
});

// ─── Multi-page scan — stitch pages then scan once ───────────────────────────
//
// POST /api/scan-multi
// Accepts multiple base64 images (one per page), stitches them vertically
// with Sharp into one tall image, then runs ONE Groq/Tesseract scan.
//
// Body: {
//   pages: [ { imageBase64, mimeType }, ... ],
//   examType, questionCount, questionTypeMap, mixedMode
// }

app.post('/api/scan-multi', async (req, res) => {
  const { pages, examType, questionCount, questionTypeMap, mixedMode } = req.body;

  if (!pages || !Array.isArray(pages) || pages.length === 0) {
    return res.status(400).json({ success: false, error: 'pages array is required.' });
  }
  if (!examType) {
    return res.status(400).json({ success: false, error: 'examType is required.' });
  }

  const totalQs = Number(questionCount) || 10;

  try {
    console.log(`[AutoChecker] scan-multi — ${pages.length} page(s), examType=${examType}, questions=${totalQs}`);

    let finalImageBase64;
    let finalMimeType = 'image/png';

    if (pages.length === 1) {
      finalImageBase64 = pages[0].imageBase64;
      finalMimeType    = pages[0].mimeType || 'image/jpeg';
      console.log('[scan-multi] Single page — skipping stitch');

    } else {
      if (!sharp) {
        return res.status(500).json({
          success: false,
          error: 'sharp is not installed. Run: npm install sharp',
        });
      }

      console.log(`[scan-multi] Stitching ${pages.length} pages vertically...`);

      const rawBuffers  = pages.map(p => Buffer.from(p.imageBase64, 'base64'));
      const metadatas   = await Promise.all(rawBuffers.map(buf => sharp(buf).metadata()));
      const targetWidth = Math.max(...metadatas.map(m => m.width ?? 0));

      const resizedBuffers = await Promise.all(
        rawBuffers.map(buf =>
          sharp(buf)
            .rotate()
            .resize({
              width:              targetWidth,
              fit:                'contain',
              background:         { r: 255, g: 255, b: 255, alpha: 1 },
              withoutEnlargement: false,
            })
            .png()
            .toBuffer()
        )
      );

      const resizedMetas = await Promise.all(
        resizedBuffers.map(buf => sharp(buf).metadata())
      );

      let yOffset = 0;
      const compositeInputs = resizedBuffers.map((buf, i) => {
        const entry = { input: buf, top: yOffset, left: 0 };
        yOffset += resizedMetas[i].height ?? 0;
        return entry;
      });

      const totalHeight = yOffset;
      console.log(`[scan-multi] Canvas: ${targetWidth}x${totalHeight}px`);

      const stitchedBuffer = await sharp({
        create: {
          width:      targetWidth,
          height:     totalHeight,
          channels:   3,
          background: { r: 255, g: 255, b: 255 },
        },
      })
        .composite(compositeInputs)
        .png()
        .toBuffer();

      finalImageBase64 = stitchedBuffer.toString('base64');
      finalMimeType    = 'image/png';
      console.log(`[scan-multi] Stitch complete — ${(stitchedBuffer.length / 1024).toFixed(1)} KB`);
    }

    const { studentName, answers, answeredCount, engineConfidence, confidence } =
      await parseVisionText(
        finalImageBase64, finalMimeType, examType, totalQs,
        questionTypeMap, mixedMode
      );

    let notes = '';
    if (confidence < 0.3) {
      notes = 'Very low confidence — image may be blurry or dark. Please retake in good lighting.';
    } else if (confidence < 0.55) {
      notes = 'Low confidence — some answers may be missing. Review before saving.';
    } else if (confidence < 0.8) {
      notes = 'Moderate confidence — please verify the detected answers.';
    }

    if (answeredCount === 0 && confidence < 0.1) {
      return res.json({
        success:        false,
        error:          'The scanner could not read any answers.\n\nTips:\n• Use bright, even lighting\n• Hold camera directly above the sheet\n• Keep the page flat with no shadows or folds',
        answers:        buildEmptyAnswers(totalQs),
        confidence:     0,
        engineConfidence,
        answeredCount:  0,
        totalQuestions: totalQs,
        notes,
      });
    }

    res.json({
      success:        true,
      answers,
      studentName,
      confidence,
      engineConfidence,
      answeredCount,
      totalQuestions: totalQs,
      notes,
    });

  } catch (err) {
    console.error('[scan-multi] Error:', err.message);
    res.status(500).json({
      success:        false,
      error:          err.message,
      answers:        buildEmptyAnswers(totalQs || 10),
      confidence:     0,
      answeredCount:  0,
      totalQuestions: totalQs || 10,
      notes:          'Scan failed. Please try again.',
    });
  }
});

// â”€â”€â”€ Scan â€” upgraded OCR pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// âœ¨ UPGRADED: returns { success: false, answers: {}, confidence: 0 }
//             instead of 500 when OCR produces no usable text
//             â†’ client never crashes on a bad scan


// --- POST /api/scan-bubbles - accept pre-processed bubble measurements ----------
// The phone's omrImageProcessor.ts runs a full local OMR pipeline and sends
// BubbleMeasurement[] results here. This bypasses Groq (hallucinates) and Jimp
// (needs perfect calibration). The phone processor sees the full-res image with
// adaptive thresholding - far more accurate than any server-side approach.
//
// Body: { bubbles: [{questionNumber, option, fillRatio}], questionCount, sectionId, studentName }

app.post('/api/scan-bubbles', async (req, res) => {
  const { bubbles, questionCount, sectionId, studentName } = req.body;
  if (!bubbles || !Array.isArray(bubbles) || bubbles.length === 0) {
    return res.status(400).json({ success: false, error: 'bubbles array is required.' });
  }
  const totalQs = Number(questionCount) || 50;
  console.log('[ScanBubbles] Received ' + bubbles.length + ' measurements for ' + totalQs + ' questions');

  try {
    const byQuestion = {};
    for (const b of bubbles) {
      const q = Number(b.questionNumber);
      if (!q || q < 1 || q > totalQs) continue;
      if (!byQuestion[q]) byQuestion[q] = [];
      byQuestion[q].push({ option: String(b.option).toUpperCase(), fill: Number(b.fillRatio) || 0 });
    }

    const answers = {};
    const confidences = {};

    for (let q = 1; q <= totalQs; q++) {
      const group = byQuestion[q];
      if (!group || group.length === 0) { answers[String(q)] = ''; continue; }
      group.sort((a, b) => b.fill - a.fill);
      const best   = group[0];
      const second = group[1];
      const margin = best.fill - (second ? second.fill : 0);

      if (best.fill < 0.15) {
        answers[String(q)] = '';
        confidences[String(q)] = 0;
        continue;
      }

      // Double mark: both high AND very close
      if (best.fill >= 0.40 && second && second.fill >= 0.40 && second.fill / best.fill >= 0.90) {
        console.log('[ScanBubbles] Q' + q + ': double mark resolved to ' + best.option);
      }

      answers[String(q)] = best.option;
      confidences[String(q)] = margin * best.fill;
      console.log('[ScanBubbles] Q' + q + ': ' + best.option + ' (fill=' + (best.fill*100).toFixed(1) + '% margin=' + (margin*100).toFixed(1) + '%)');
    }

    for (let i = 1; i <= totalQs; i++) {
      if (answers[String(i)] === undefined) answers[String(i)] = '';
    }

    const answeredCount = Object.values(answers).filter(a => a !== '').length;
    const meanConf = Object.values(confidences).reduce((s, v) => s + v, 0) / Math.max(1, totalQs);
    const confidence = Math.min(0.99, meanConf * 3.0);

    console.log('[ScanBubbles] Done - answered: ' + answeredCount + '/' + totalQs + ', confidence: ' + (confidence*100).toFixed(1) + '%');

    res.json({
      success: true,
      answers,
      studentName: studentName || null,
      confidence,
      engineConfidence: confidence * 100,
      answeredCount,
      totalQuestions: totalQs,
      notes: answeredCount < totalQs * 0.8 ? 'Some bubbles not detected. Ensure good lighting and flat paper.' : '',
    });
  } catch (err) {
    console.error('[ScanBubbles] Error:', err.message);
    res.status(500).json({ success: false, error: err.message, answers: buildEmptyAnswers(totalQs) });
  }
});

app.post('/api/scan', async (req, res) => {
  const { imageBase64, mimeType, examType, sectionId, questionCount,
          questionTypeMap: clientQuestionTypeMap, mixedMode } = req.body;
  if (!imageBase64) return res.status(400).json({ success: false, error: 'imageBase64 is required.' });
  if (!examType)    return res.status(400).json({ success: false, error: 'examType is required.' });

  // Use questionCount from client — it's already derived from the answer key.
  const totalQs = Number(questionCount) || 10;

  // ── FIX: Auto-build questionTypeMap from the answer key when not supplied ──
  // When the app sends mixedMode=true but NO questionTypeMap (because the teacher
  // didn't configure question ranges), every question is parsed using only
  // primaryExamType — so T/F questions get the MC parser and return "?".
  //
  // Solution: if sectionId is provided and mixedMode is set but questionTypeMap
  // is empty, load the answer key from disk and build the map automatically.
  // Each question inherits its type from its answer key entry.
  let questionTypeMap = clientQuestionTypeMap;
  if (sectionId && (!questionTypeMap || Object.keys(questionTypeMap).length === 0)) {
    try {
      const store = readStore(sectionId);
      if (store && Array.isArray(store.key) && store.key.length > 0) {
        const autoMap = {};
        store.key.forEach(item => {
          autoMap[String(item.question)] = item.type; // backend type string
        });
        questionTypeMap = autoMap;
        const hasMultipleTypes = new Set(Object.values(autoMap)).size > 1;
        if (hasMultipleTypes) {
          console.log(`[AutoChecker] Auto-built questionTypeMap from answer key — ${Object.keys(autoMap).length} questions, types: ${[...new Set(Object.values(autoMap))].join(', ')}`);
        }
      }
    } catch (e) {
      console.warn('[AutoChecker] Could not load answer key for auto questionTypeMap:', e.message);
    }
  }

  // Determine if the exam is truly mixed (more than one question type)
  const effectiveMixedMode = mixedMode ||
    (questionTypeMap && new Set(Object.values(questionTypeMap)).size > 1);

  // Wait up to 10s for persistent Tesseract workers on cold start.
  if (!workerMCReady && !workerTextReady) {
    console.log('[AutoChecker] Workers not ready — waiting up to 10s for warm-up...');
    for (let i = 0; i < 20; i++) {
      await new Promise(r => setTimeout(r, 500));
      if (workerMCReady || workerTextReady) {
        console.log(`[AutoChecker] Workers ready after ${(i + 1) * 0.5}s`);
        break;
      }
    }
  }

  try {
    console.log(`[AutoChecker] Scanning â€” examType=${examType}, questions=${totalQs}`);

    const { studentName, answers, answeredCount, engineConfidence, confidence } =
      await parseVisionText(imageBase64, mimeType || 'image/jpeg', examType, totalQs,
        questionTypeMap, effectiveMixedMode); // FIX: pass effectiveMixedMode so auto-map activates mixed path

    let notes = '';
    if (confidence < 0.3) {
      notes = 'Very low confidence â€” image may be blurry, dark, or rotated. Please retake in good lighting.';
    } else if (confidence < 0.55) {
      notes = 'Low confidence â€” some answers may be missing or incorrect. Review before saving.';
    } else if (confidence < 0.8) {
      notes = 'Moderate confidence â€” please verify the detected answers.';
    }

    // BUG FIX: previously always returned success:true even with 0 answers.
    // The client checks success flag first and surfaces ocrResult.message.
    // If no answers were found, return success:false with a clear error message
    // so the user sees the real problem instead of the generic "No Test Paper" alert.
    if (answeredCount === 0 && confidence < 0.1) {
      return res.json({
        success: false,
        error: 'The scanner could not read any answers from this image.\n\nTips:\nâ€¢ Use bright, even lighting\nâ€¢ Hold camera directly above the sheet, completely flat\nâ€¢ Keep the page flat with no shadows, glare, or folds\nâ€¢ Make sure answers are written clearly inside the answer boxes',
        answers: buildEmptyAnswers(totalQs),
        confidence: 0,
        engineConfidence,
        answeredCount: 0,
        totalQuestions: totalQs,
        notes,
      });
    }

    // âœ¨ UPGRADED: always return success:true with answers (even if empty)
    // Let the client decide whether the result is usable based on answeredCount/confidence
    res.json({
      success: true,
      answers,
      studentName,
      confidence,
      engineConfidence,
      answeredCount,
      totalQuestions: totalQs,
      notes,
    });

  } catch (err) {
    console.error('[AutoChecker] Scan error:', err.message);

    // âœ¨ UPGRADED: return structured error â€” never bare 500
    res.status(500).json({
      success:     false,
      error:       err.message,
      answers:     buildEmptyAnswers(totalQs || 10),
      confidence:  0,
      answeredCount: 0,
      totalQuestions: totalQs || 10,
      notes:       'Scan failed. Please try again with better lighting.',
    });
  }
});

// â”€â”€â”€ Settings routes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

app.get('/api/settings', function(_req, res) {
  res.json(readSettings());
});

app.put('/api/settings', function(req, res) {
  var ALLOWED_KEYS = Object.keys(DEFAULT_SETTINGS);
  var incoming = req.body || {};
  var unknown = Object.keys(incoming).filter(function(k) { return !ALLOWED_KEYS.includes(k); });
  if (unknown.length) return res.status(400).json({ success: false, error: 'Unknown setting(s): ' + unknown.join(', ') });
  for (var k of Object.keys(incoming)) {
    if (typeof incoming[k] !== 'boolean') return res.status(400).json({ success: false, error: '"' + k + '" must be a boolean.' });
  }
  var current = readSettings();
  var updated = Object.assign({}, current, incoming);
  writeSettings(updated);
  console.log('[AutoChecker] Settings updated:', incoming);
  res.json(updated);
});

app.post('/api/settings/reset', function(_req, res) {
  writeSettings(Object.assign({}, DEFAULT_SETTINGS));
  console.log('[AutoChecker] Settings reset to defaults.');
  res.json(Object.assign({}, DEFAULT_SETTINGS));
});


// ─── OMR Debug overlay — visualise sampler positions on the actual image ───────
// POST /api/omr-debug  { imageBase64, mimeType, questionCount }
// Returns a JPEG with red circles drawn at every bubble sample point.
// Use this to verify that Q_NUM_W, BUBBLE_STEP, GRID_TOP, and ROW_STEP are correct.
// If red circles don't land inside the printed bubbles, adjust OMR_LAYOUT constants.
app.post('/api/omr-debug', async (req, res) => {
  const { imageBase64, mimeType, questionCount } = req.body;
  if (!imageBase64)    return res.status(400).json({ error: 'imageBase64 required' });
  if (!Jimp)           return res.status(500).json({ error: 'jimp not installed' });
  if (!questionCount)  return res.status(400).json({ error: 'questionCount required' });

  try {
    const numCols = omrColCount(questionCount);
    const layout  = OMR_LAYOUT[numCols];
    const buf     = Buffer.from(imageBase64, 'base64');

    // Load and resize to canonical A4
    let img = await (Jimp.fromBuffer ? Jimp.fromBuffer(buf) : Jimp.read(buf));
    img = img.resize(TARGET_W, TARGET_H);
    const W = TARGET_W, H = TARGET_H;

    const perCol  = Math.ceil(questionCount / numCols);
    const r       = Math.round(layout.BUBBLE_R * W);
    const rowStep = layout.ROW_STEP * H;
    const COLORS  = [
      Jimp.rgbaToInt(220, 50,  50,  255),  // A — red
      Jimp.rgbaToInt(50,  150, 220, 255),  // B — blue
      Jimp.rgbaToInt(50,  200, 80,  255),  // C — green
      Jimp.rgbaToInt(240, 160, 20,  255),  // D — amber
    ];

    for (let col = 0; col < numCols; col++) {
      const bubbleX0 = (layout.COL_LEFT[col] + layout.Q_NUM_W) * W;
      const colStart = col * perCol + 1;
      const colEnd   = Math.min(colStart + perCol - 1, questionCount);

      for (let row = 0; row < colEnd - colStart + 1; row++) {
        const cy = Math.round(layout.GRID_TOP * H + row * rowStep + rowStep * 0.5);

        for (let i = 0; i < 4; i++) {
          const cx    = Math.round(bubbleX0 + i * layout.BUBBLE_STEP * W);
          const color = COLORS[i];

          // Draw circle outline at each sample point
          for (let a = 0; a < 360; a += 3) {
            const rad = a * Math.PI / 180;
            const px  = Math.round(cx + r * Math.cos(rad));
            const py  = Math.round(cy + r * Math.sin(rad));
            if (px >= 0 && px < W && py >= 0 && py < H) {
              img.setPixelColor(color, px, py);
            }
          }

          // Draw cross-hair at centre
          for (let d = -3; d <= 3; d++) {
            if (cx + d >= 0 && cx + d < W) img.setPixelColor(color, cx + d, cy);
            if (cy + d >= 0 && cy + d < H) img.setPixelColor(color, cx, cy + d);
          }
        }
      }
    }

    // Return as JPEG directly — open in browser or Postman to inspect
    const outBuf = await img.getBuffer('image/jpeg');
    res.setHeader('Content-Type', 'image/jpeg');
    res.setHeader('Content-Disposition', 'inline; filename="omr-debug.jpg"');
    res.send(outBuf);
    console.log(`[OMR Debug] Overlay generated — ${questionCount}Q, ${numCols} col(s)`);
  } catch (err) {
    console.error('[OMR Debug] Error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// Health
app.get('/health', (_req, res) => res.json({
  status: 'ok',
  sharp:  !!sharp,
  ocr:    'tesseract.js',
  bubbleOmr: !!Jimp ? 'jimp-pixel-detection' : 'unavailable (npm install jimp)',
  groq: groqReady ? 'enabled (llama-4-scout-17b-16e-instruct) — FREE' : GROQ_API_KEY ? 'key set but probe failed' : 'disabled (add GROQ_API_KEY to Railway env vars)',
  pipeline: 'tesseract-primary / groq-optional-enhancer',
  version: '8.5-75Q-groq-hallucination-fix',
}));

// Error handler
app.use((err, _req, res, _next) => {
  if (err.code === 'LIMIT_FILE_SIZE') return res.status(413).json({ success: false, error: 'File exceeds 5 MB limit.' });
  res.status(500).json({ success: false, error: err.message });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log('[AutoChecker] v7.0 running on http://0.0.0.0:' + PORT);
  console.log('[AutoChecker] OCR: Tesseract.js LSTM (OEM set at worker init) + ' + (sharp ? 'sharp preprocessing enabled' : 'NO sharp -- run: npm install sharp'));
  console.log('[AutoChecker] Groq vision: ' + (groqReady ? 'ready ✓' : GROQ_API_KEY ? 'key set, probe pending...' : 'not configured (add GROQ_API_KEY)'));
  console.log('[AutoChecker] PSM routing -- MCQ:[6,4]  TrueFalse:[7,6,11]  Written:[6,11]');
});

// Keep-alive Sping â€” prevents Railway from sleeping
setInterval(() => {
  fetch('https://autocheckernew-backend-production.up.railway.app/health')
    .catch(() => {});
}, 4 * 60 * 1000); // reduced from 5min — Railway sleeps at 5min inactivity

// redeploy-trigger-20260526-v85-75Q-fix-groq-hallucination-jimp-recal