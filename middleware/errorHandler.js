const logger = require('../utils/logger');

// ─── Custom application error class ───────────────────────────────────────────
// Throw this anywhere in controllers/services for clean, structured errors.
//
// Usage:
//   throw new AppError('Scene not found.', 404);
//   throw new AppError('GPU quota exceeded.', 429, 'QUOTA_EXCEEDED');
//
class AppError extends Error {
  constructor(message, statusCode = 500, code = null) {
    super(message);
    this.name       = 'AppError';
    this.statusCode = statusCode;
    this.code       = code;
    this.isOperational = true;   // distinguishes expected errors from bugs
    Error.captureStackTrace(this, this.constructor);
  }
}

// ─── Global Error Handler ──────────────────────────────────────────────────────
// Must have exactly 4 parameters for Express to treat it as an error handler.
// Mounted last in app.js: app.use(errorHandler)
//
// Handles all errors thrown or passed via next(err) across the entire app.
//
function errorHandler(err, req, res, next) { // eslint-disable-line no-unused-vars
  // ── Log with context ────────────────────────────────────────────────────────
  const logMeta = {
    method:  req.method,
    url:     req.originalUrl,
    userId:  req.user?._id?.toString() || 'unauthenticated',
    ip:      req.ip,
  };

  if (err.isOperational) {
    // Expected application error (AppError) — info level, no stack trace needed
    logger.warn(`[${err.statusCode}] ${err.message}`, logMeta);
  } else {
    // Unexpected bug — error level with full stack
    logger.error(`[${err.statusCode || 500}] ${err.message}`, {
      ...logMeta,
      stack: err.stack,
    });
  }

  // ── Mongoose: validation failed ─────────────────────────────────────────────
  if (err.name === 'ValidationError') {
    const messages = Object.values(err.errors).map((e) => e.message);
    return res.status(400).json({
      success: false,
      message: 'Validation failed.',
      errors:  messages,
    });
  }

  // ── Mongoose: duplicate key (e.g. unique email) ──────────────────────────────
  if (err.code === 11000) {
    const field = Object.keys(err.keyPattern || {})[0] || 'field';
    return res.status(409).json({
      success: false,
      message: `${capitalise(field)} already exists.`,
      field,
    });
  }

  // ── Mongoose: invalid ObjectId ───────────────────────────────────────────────
  if (err.name === 'CastError' && err.kind === 'ObjectId') {
    return res.status(400).json({
      success: false,
      message: `Invalid ID format: "${err.value}".`,
    });
  }

  // ── Multer: file too large ───────────────────────────────────────────────────
  if (err.code === 'LIMIT_FILE_SIZE') {
    return res.status(400).json({
      success: false,
      message: 'File too large. Maximum size is 100 MB per file.',
    });
  }

  // ── Multer: too many files ───────────────────────────────────────────────────
  if (err.code === 'LIMIT_FILE_COUNT') {
    return res.status(400).json({
      success: false,
      message: 'Too many files. Maximum 50 files per upload.',
    });
  }

  // ── Multer: invalid file type (thrown in fileFilter) ────────────────────────
  if (err.code === 'INVALID_FILE_TYPE' || err.message?.startsWith('Invalid file type')) {
    return res.status(400).json({
      success: false,
      message: err.message,
    });
  }

  // ── Cloudinary errors ────────────────────────────────────────────────────────
  if (err.http_code) {
    logger.error(`Cloudinary error ${err.http_code}: ${err.message}`);
    return res.status(502).json({
      success: false,
      message: 'Storage service error. Please try again.',
    });
  }

  // ── Bull / Redis errors ──────────────────────────────────────────────────────
  if (err.message?.includes('Redis') || err.message?.includes('ECONNREFUSED')) {
    return res.status(503).json({
      success: false,
      message: 'Job queue temporarily unavailable. Please try again shortly.',
    });
  }

  // ── Our own AppError ─────────────────────────────────────────────────────────
  if (err.isOperational) {
    return res.status(err.statusCode).json({
      success: false,
      message: err.message,
      ...(err.code && { code: err.code }),
    });
  }

  // ── Unhandled / unknown error ────────────────────────────────────────────────
  // In production, never reveal internal details to the client
  const isProd = process.env.NODE_ENV === 'production';
  return res.status(500).json({
    success: false,
    message: isProd ? 'An unexpected error occurred. Please try again.' : err.message,
    ...(!isProd && { stack: err.stack }),
  });
}

// ─── Helper ───────────────────────────────────────────────────────────────────
function capitalise(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

module.exports = errorHandler;
module.exports.AppError = AppError;
