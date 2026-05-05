// backend/middleware/authenticate.ts
//
// Drop-in JWT authentication middleware.
// Replace the verify() stub with your real JWT library (e.g. jsonwebtoken).
//
import { NextFunction, Request, Response } from 'express';
// Extend Express Request so TypeScript knows about req.user
declare global {
  namespace Express {
    interface Request {
      user?: { id: string; email: string };
    }
  }
}

export async function authenticate(
  req: Request,
  res: Response,
  next: NextFunction,
) {
  try {
    const header = req.headers.authorization ?? '';
    if (!header.startsWith('Bearer ')) {
      return res.status(401).json({ message: 'Missing token' });
    }

    const token = header.slice(7);

    // ── Replace this block with your real JWT verification ──────────────────
    // Example using jsonwebtoken:
    //   import jwt from 'jsonwebtoken';
    //   const payload = jwt.verify(token, process.env.JWT_SECRET!) as { sub: string; email: string };
    //   req.user = { id: payload.sub, email: payload.email };
    // ────────────────────────────────────────────────────────────────────────

    if (!token) throw new Error('Empty token');

    // Placeholder: decode a base64 payload (replace with real JWT lib)
    const payloadB64 = token.split('.')[1];
    if (!payloadB64) throw new Error('Malformed token');
    const payload = JSON.parse(
      Buffer.from(payloadB64, 'base64url').toString('utf8'),
    );

    req.user = { id: payload.sub ?? payload.id, email: payload.email ?? '' };
    next();
  } catch (err) {
    console.error('[authenticate]', err);
    return res.status(401).json({ message: 'Invalid or expired token' });
  }
}