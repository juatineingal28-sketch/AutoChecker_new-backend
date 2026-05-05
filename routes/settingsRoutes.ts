// backend/routes/settingsRoutes.ts
import { Router } from 'express';
import {
  getSettings,
  resetSettings,
  updateSettings,
} from '../controller/settingsController';
import { authenticate } from '../middleware/authenticate';

const router = Router();

// All routes require a valid JWT / session
router.use(authenticate);

router.get('/',       getSettings);
router.put('/',       updateSettings);
router.post('/reset', resetSettings);

export default router;