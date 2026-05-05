/**
 * backend/src/index.ts
 *
 * Express + SQLite (via better-sqlite3) backend.
 *
 * Install deps:
 *   npm install express better-sqlite3 multer uuid
 *   npm install -D @types/express @types/better-sqlite3 @types/multer @types/uuid ts-node typescript
 *
 * Run:
 *   npx ts-node src/index.ts
 * (or compile with tsc first, then node dist/index.js)
 *
 * ─── ROUTES ───────────────────────────────────────────────────────────────────
 *  GET    /api/sections              → list all sections
 *  POST   /api/sections              → create a section
 *  DELETE /api/sections/:id          → permanently delete a section  ← THE FIX
 *  POST   /api/answer-key/:sectionId → upload answer key (multer)
 *  GET    /api/answer-key/:sectionId → get answer key for a section
 *  DELETE /api/answer-key/:sectionId → delete answer key for a section
 *  POST   /api/score/:sectionId      → score a student submission
 */

import Database from 'better-sqlite3';
import express, { NextFunction, Request, Response } from 'express';
import multer from 'multer';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';

// ─── App & DB setup ───────────────────────────────────────────────────────────

const app  = express();
const PORT = 3000;

app.use(express.json());

// SQLite database (single file, no extra server needed)
const db = new Database(path.join(__dirname, '../../sections.db'));

// ─── Schema ───────────────────────────────────────────────────────────────────

db.exec(`
  CREATE TABLE IF NOT EXISTS sections (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    abbr         TEXT NOT NULL DEFAULT '',
    subject      TEXT NOT NULL DEFAULT '',
    studentCount INTEGER NOT NULL DEFAULT 0,
    color        TEXT NOT NULL DEFAULT 'blue',
    average      REAL NOT NULL DEFAULT 0,
    quizCount    INTEGER NOT NULL DEFAULT 0,
    createdAt    TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS answer_keys (
    sectionId  TEXT PRIMARY KEY,
    fileName   TEXT NOT NULL,
    fileType   TEXT NOT NULL,
    uploadedAt TEXT NOT NULL,
    total      INTEGER NOT NULL DEFAULT 0,
    keyData    TEXT NOT NULL,          -- JSON array of AnswerKeyItem[]
    FOREIGN KEY (sectionId) REFERENCES sections(id) ON DELETE CASCADE
  );
`);

// ─── Multer (file uploads, stored in memory for simplicity) ──────────────────

const upload = multer({ storage: multer.memoryStorage() });

// ─── Helpers ──────────────────────────────────────────────────────────────────

function now() {
  return new Date().toISOString();
}

function rowToSection(row: any) {
  return {
    id:           row.id,
    name:         row.name,
    abbr:         row.abbr,
    subject:      row.subject,
    studentCount: row.studentCount,
    color:        row.color,
    average:      row.average,
    quizCount:    row.quizCount,
    createdAt:    row.createdAt,
    // answer key fields (may be null if no key uploaded)
    hasAnswerKey: !!row.ak_sectionId,
    fileName:     row.ak_fileName   ?? null,
    fileType:     row.ak_fileType   ?? null,
    uploadedAt:   row.ak_uploadedAt ?? null,
    total:        row.ak_total      ?? null,
  };
}

// ─── GET /api/sections ────────────────────────────────────────────────────────

app.get('/api/sections', (_req: Request, res: Response) => {
  const rows = db.prepare(`
    SELECT s.*,
           ak.sectionId  AS ak_sectionId,
           ak.fileName   AS ak_fileName,
           ak.fileType   AS ak_fileType,
           ak.uploadedAt AS ak_uploadedAt,
           ak.total      AS ak_total
    FROM sections s
    LEFT JOIN answer_keys ak ON ak.sectionId = s.id
    ORDER BY s.createdAt DESC
  `).all();

  res.json({ success: true, sections: rows.map(rowToSection) });
});

// ─── POST /api/sections ───────────────────────────────────────────────────────

app.post('/api/sections', (req: Request, res: Response) => {
  const { name, abbr = '', subject = '', studentCount = 0, color = 'blue' } = req.body;

  if (!name?.trim()) {
    res.status(400).json({ success: false, error: 'Section name is required.' });
    return;
  }

  const section = {
    id:           uuidv4(),
    name:         name.trim(),
    abbr:         abbr.trim(),
    subject:      subject.trim(),
    studentCount: Number(studentCount),
    color,
    average:      0,
    quizCount:    0,
    createdAt:    now(),
  };

  db.prepare(`
    INSERT INTO sections (id, name, abbr, subject, studentCount, color, average, quizCount, createdAt)
    VALUES (@id, @name, @abbr, @subject, @studentCount, @color, @average, @quizCount, @createdAt)
  `).run(section);

  res.status(201).json({ success: true, section: { ...section, hasAnswerKey: false } });
});

// ─── DELETE /api/sections/:id  ← THE FIX ─────────────────────────────────────
//
//  WHY THIS IS THE FIX:
//  ────────────────────────────────────────────────────────────────────────────
//  Without this route the frontend had NO way to permanently remove a section.
//  setSections(prev => prev.filter(...)) only clears the section from React's
//  in-memory state, which is destroyed when the app closes. On next login,
//  fetchSections() fetches the full, unchanged database, and the "deleted"
//  section reappears.
//
//  This route physically removes the row from the `sections` table (and its
//  answer key row via the ON DELETE CASCADE foreign key).  Once gone, it can
//  never be returned by GET /api/sections.
//
//  IDEMPOTENT: returns 200 even if the section is already absent so the client
//  can safely retry on network hiccups without getting a confusing 404.

app.delete('/api/sections/:id', (req: Request, res: Response) => {
const rawId = req.params.id;
const id = Array.isArray(rawId) ? rawId[0] : rawId;

if (!(id ?? '').trim()) {
    res.status(400).json({ success: false, error: 'Section ID is required.' });
    return;
  }

  // The answer_keys row is removed automatically via ON DELETE CASCADE.
  const result = db.prepare('DELETE FROM sections WHERE id = ?').run(id);

  if (result.changes === 0) {
    // Section didn't exist — treat as success (idempotent).
    // This handles the edge-case where the client retries after a timeout.
    res.json({ success: true, message: 'Section not found (already deleted).' });
    return;
  }

  res.json({ success: true, message: 'Section permanently deleted.' });
});

// ─── POST /api/answer-key/:sectionId ─────────────────────────────────────────

app.post(
  '/api/answer-key/:sectionId',
  upload.single('file'),
  (req: Request, res: Response) => {
    const { sectionId } = req.params;
    if (!req.file) {
      res.status(400).json({ success: false, error: 'No file uploaded.' });
      return;
    }

    const section = db.prepare('SELECT id FROM sections WHERE id = ?').get(sectionId);
    if (!section) {
      res.status(404).json({ success: false, error: 'Section not found.' });
      return;
    }

    // For .json files — parse and store structured key data.
    let keyData: any[] = [];
    if (req.file.mimetype === 'application/json') {
      try {
        keyData = JSON.parse(req.file.buffer.toString('utf-8'));
      } catch {
        res.status(400).json({ success: false, error: 'Invalid JSON in answer key file.' });
        return;
      }
    }
    // For other types (pdf, docx, txt) — store raw text or handle with a parser.
    // Extend this block with pdf-parse / mammoth as needed.

    const record = {
      sectionId,
      fileName:   req.file.originalname,
      fileType:   req.file.mimetype,
      uploadedAt: now(),
      total:      keyData.length,
      keyData:    JSON.stringify(keyData),
    };

    db.prepare(`
      INSERT INTO answer_keys (sectionId, fileName, fileType, uploadedAt, total, keyData)
      VALUES (@sectionId, @fileName, @fileType, @uploadedAt, @total, @keyData)
      ON CONFLICT(sectionId) DO UPDATE SET
        fileName   = excluded.fileName,
        fileType   = excluded.fileType,
        uploadedAt = excluded.uploadedAt,
        total      = excluded.total,
        keyData    = excluded.keyData
    `).run(record);

    res.json({
      success:    true,
      sectionId,
      fileName:   record.fileName,
      fileType:   record.fileType,
      uploadedAt: record.uploadedAt,
      total:      record.total,
      key:        keyData,
    });
  },
);

// ─── GET /api/answer-key/:sectionId ──────────────────────────────────────────

app.get('/api/answer-key/:sectionId', (req: Request, res: Response) => {
  const row = db
    .prepare('SELECT * FROM answer_keys WHERE sectionId = ?')
    .get(req.params.sectionId) as any;

  if (!row) {
    res.status(404).json({ success: false, error: 'No answer key found for this section.' });
    return;
  }

  res.json({
    success:    true,
    sectionId:  row.sectionId,
    fileName:   row.fileName,
    fileType:   row.fileType,
    uploadedAt: row.uploadedAt,
    total:      row.total,
    key:        JSON.parse(row.keyData),
  });
});

// ─── DELETE /api/answer-key/:sectionId ───────────────────────────────────────

app.delete('/api/answer-key/:sectionId', (req: Request, res: Response) => {
  db.prepare('DELETE FROM answer_keys WHERE sectionId = ?').run(req.params.sectionId);
  res.json({ success: true });
});

// ─── POST /api/score/:sectionId ───────────────────────────────────────────────

app.post('/api/score/:sectionId', (req: Request, res: Response) => {
  const { sectionId } = req.params;
  const { student }   = req.body as {
    student: { id: string; name: string; answers: string[] };
  };

  const row = db
    .prepare('SELECT keyData FROM answer_keys WHERE sectionId = ?')
    .get(sectionId) as any;

  if (!row) {
    res.status(404).json({ success: false, error: 'No answer key for this section.' });
    return;
  }

  const key: Array<{ question: number; type: string; answer: string | string[] }> =
    JSON.parse(row.keyData);

  let score = 0;
  const details = key.map((item, i) => {
    const studentAnswer = student.answers[i] ?? null;
    const correct       = Array.isArray(item.answer)
      ? item.answer.map(a => a.toLowerCase()).includes((studentAnswer ?? '').toLowerCase())
      : (studentAnswer ?? '').toLowerCase() === String(item.answer).toLowerCase();

    if (correct) score++;
    return {
      question:      item.question,
      type:          item.type,
      correctAnswer: item.answer,
      studentAnswer,
      isCorrect:     correct,
    };
  });

  const total      = key.length;
  const percentage = total > 0 ? Math.round((score / total) * 100) : 0;
  const status     = percentage >= 75 ? 'Passed' : percentage >= 50 ? 'Review' : 'Failed';

  res.json({
    success: true,
    student: { id: student.id, name: student.name },
    score,
    total,
    percentage,
    status,
    details,
  });
});

// ─── Global error handler ─────────────────────────────────────────────────────

app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
  console.error('[ERROR]', err);
  res.status(500).json({ success: false, error: err?.message ?? 'Internal server error.' });
});

// ─── Start ────────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`✅  Server running on http://localhost:${PORT}`);
  console.log(`    (run: adb reverse tcp:${PORT} tcp:${PORT} for USB testing)`);
});

export { };

