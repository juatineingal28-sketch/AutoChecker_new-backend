// backend/controllers/settingsController.ts
import { Request, Response } from 'express';
import { Settings } from '../modals/Settings';

const DEFAULT_SETTINGS = {
  autoDetect: true,
  scanTips:   true,
  flagLow:    false,
};

const ALLOWED_FIELDS = new Set(Object.keys(DEFAULT_SETTINGS));

/** GET /api/settings */
export async function getSettings(req: Request, res: Response): Promise<void> {
  try {
    const userId = (req as any).user?.id;
    let doc = await Settings.findOne({ userId });

    if (!doc) {
      doc = await Settings.create({ userId, ...DEFAULT_SETTINGS });
    }

    res.json(toDTO(doc));
  } catch (err) {
    console.error('[getSettings]', err);
    res.status(500).json({ message: 'Failed to fetch settings' });
  }
}

/** PUT /api/settings – partial update */
export async function updateSettings(req: Request, res: Response): Promise<void> {
  try {
    const userId = (req as any).user?.id;

    const patch: Record<string, unknown> = {};
    for (const [key, val] of Object.entries(req.body)) {
      if (ALLOWED_FIELDS.has(key)) patch[key] = val;
    }

    if (Object.keys(patch).length === 0) {
      res.status(400).json({ message: 'No valid fields provided' });
      return;
    }

    const doc = await Settings.findOneAndUpdate(
      { userId },
      { $set: patch },
      { new: true, upsert: true, runValidators: true },
    );

    res.json(toDTO(doc!));
  } catch (err) {
    console.error('[updateSettings]', err);
    res.status(500).json({ message: 'Failed to update settings' });
  }
}

/** POST /api/settings/reset */
export async function resetSettings(req: Request, res: Response): Promise<void> {
  try {
    const userId = (req as any).user?.id;

    const doc = await Settings.findOneAndUpdate(
      { userId },
      { $set: DEFAULT_SETTINGS },
      { new: true, upsert: true },
    );

    res.json(toDTO(doc!));
  } catch (err) {
    console.error('[resetSettings]', err);
    res.status(500).json({ message: 'Failed to reset settings' });
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function toDTO(doc: InstanceType<typeof Settings>) {
  return {
    autoDetect: doc.autoDetect,
    scanTips:   doc.scanTips,
    flagLow:    doc.flagLow,
  };
}