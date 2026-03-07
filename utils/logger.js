const winston = require('winston');
const path = require('path');

// ─── Log levels ────────────────────────────────────────────────────────────────
// error > warn > info > http > debug
// In production only error/warn/info/http are written.
// In development all levels including debug are shown.
const LOG_LEVEL = process.env.LOG_LEVEL ||
  (process.env.NODE_ENV === 'production' ? 'http' : 'debug');

// ─── Custom format for console output ─────────────────────────────────────────
const consoleFormat = winston.format.combine(
  winston.format.colorize({ all: true }),
  winston.format.timestamp({ format: 'HH:mm:ss' }),
  winston.format.errors({ stack: true }),
  winston.format.printf(({ timestamp, level, message, stack, ...meta }) => {
    const { service, ...rest } = meta;
    const extras = Object.keys(rest).length
      ? `\n  ${JSON.stringify(rest, null, 2)}`
      : '';
    const trace = stack ? `\n${stack}` : '';
    return `${timestamp} ${level}: ${message}${extras}${trace}`;
  })
);

// ─── Format for log files (structured JSON) ────────────────────────────────────
const fileFormat = winston.format.combine(
  winston.format.timestamp(),
  winston.format.errors({ stack: true }),
  winston.format.json()
);

// ─── Transports ────────────────────────────────────────────────────────────────
const transports = [
  new winston.transports.Console({ format: consoleFormat }),
];

// In production also write to rotating log files
if (process.env.NODE_ENV === 'production') {
  const logsDir = path.join(process.cwd(), 'logs');

  transports.push(
    // All logs (info and above)
    new winston.transports.File({
      filename: path.join(logsDir, 'combined.log'),
      format:   fileFormat,
      maxsize:  10 * 1024 * 1024,   // rotate at 10 MB
      maxFiles: 5,                   // keep last 5 rotated files
    }),
    // Errors only — easy to monitor separately
    new winston.transports.File({
      filename: path.join(logsDir, 'error.log'),
      level:    'error',
      format:   fileFormat,
      maxsize:  10 * 1024 * 1024,
      maxFiles: 5,
    })
  );
}

// ─── Logger instance ───────────────────────────────────────────────────────────
const logger = winston.createLogger({
  level:       LOG_LEVEL,
  defaultMeta: { service: 'gaussian-api' },
  transports,
  // Don't crash the process on unhandled exceptions — log them instead
  exceptionHandlers: [
    new winston.transports.Console({ format: consoleFormat }),
    ...(process.env.NODE_ENV === 'production'
      ? [new winston.transports.File({
          filename: path.join(process.cwd(), 'logs', 'exceptions.log'),
          format:   fileFormat,
        })]
      : []),
  ],
  rejectionHandlers: [
    new winston.transports.Console({ format: consoleFormat }),
    ...(process.env.NODE_ENV === 'production'
      ? [new winston.transports.File({
          filename: path.join(process.cwd(), 'logs', 'rejections.log'),
          format:   fileFormat,
        })]
      : []),
  ],
});

// ─── Morgan stream ─────────────────────────────────────────────────────────────
// Used in app.js: morgan('combined', { stream: logger.stream })
logger.stream = {
  write: (message) => logger.http(message.trim()),
};

module.exports = logger;
