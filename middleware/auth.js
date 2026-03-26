const jwt = require('jsonwebtoken');
const User = require('../models/User');
const logger = require('../utils/logger');

// ─── authenticate ──────────────────────────────────────────────────────────────
// Verifies the Bearer JWT in the Authorization header.
// On success, attaches the full user document to req.user.
// On failure, returns 401 — never calls next() with an error so the
// global error handler doesn't accidentally leak stack traces on auth failures.
//
// Usage:
//   router.get('/protected', authenticate, handler);
//





async function authenticate(req, res, next) {
  try {
    // ── 1. Extract token from HTTP-only cookie ───────────────────────────────
    const token = req.cookies?.token;
    console.log("token",token)

    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'Authentication token missing. Please log in.',
      });
    }

    // ── 2. Verify & decode ──────────────────────────────────────────────────
    let decoded;

    try {
      decoded = jwt.verify(token, process.env.JWT_SECRET);
    } catch (jwtErr) {

      if (jwtErr.name === 'TokenExpiredError') {
        return res.status(401).json({
          success: false,
          message: 'Token expired. Please log in again.',
          expiredAt: jwtErr.expiredAt,
        });
      }

      if (jwtErr.name === 'JsonWebTokenError') {
        return res.status(401).json({
          success: false,
          message: 'Invalid token.',
        });
      }

      return res.status(401).json({
        success: false,
        message: 'Token validation failed.',
      });
    }

    // ── 3. Load user from DB ────────────────────────────────────────────────
    const user = await User.findById(decoded.id);

    if (!user) {
      return res.status(401).json({
        success: false,
        message: 'User belonging to this token no longer exists.',
      });
    }

    if (!user.isActive) {
      return res.status(403).json({
        success: false,
        message: 'Your account has been deactivated. Please contact support.',
      });
    }

    // ── 4. Attach user to request ───────────────────────────────────────────
    req.user = user;

    logger.debug(
      `Authenticated: user=${user._id} path=${req.method} ${req.originalUrl}`
    );

    next();
  } catch (err) {
    logger.error(`authenticate middleware error: ${err.message}`);
    next(err);
  }
}
// ─── optionalAuthenticate ──────────────────────────────────────────────────────
// Same as authenticate but does NOT reject unauthenticated requests.
// Sets req.user if a valid token is present, otherwise req.user stays undefined.
//
// Usage: public endpoints that behave differently for logged-in users
//   router.get('/scenes', optionalAuthenticate, handler);
//
async function optionalAuthenticate(req, res, next) {
  const authHeader = req.headers.authorization;

  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return next();   // no token — continue as guest
  }

  try {
    const token   = authHeader.split(' ')[1];
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    const user    = await User.findById(decoded.id);

    if (user && user.isActive) {
      req.user = user;
    }
  } catch {
    // Invalid/expired token on an optional route — just ignore it
  }

  next();
}

module.exports = { authenticate, optionalAuthenticate };
