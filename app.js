const express = require('express');
const helmet = require('helmet');
const cors = require('cors');
const morgan = require('morgan');
const { rateLimit, ipKeyGenerator } = require('express-rate-limit');
const cookieParser =require('cookie-parser')
const logger = require('./utils/logger');

const app = express();

app.use(cookieParser()); // cookie parser

// ─── Security Middleware ───────────────────────────────────────────────────────
app.use(helmet());
// app.use(cors({
//   origin: process.env.ALLOWED_ORIGINS?.split(',') || '*',
//   methods: ['GET', 'POST', 'PUT', 'DELETE'],
//   allowedHeaders: ['Content-Type', 'Authorization'],
// }));

app.use(cors({
  origin: true, // or whitelist specific origins
  credentials: true, // REQUIRED — without this cookies won't be sent
}));

// ─── Global Rate Limit (100 req / 15 min per IP) ──────────────────────────────
const globalLimiter = rateLimit({
  windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000,
  max: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS) || 100,
  message: { success: false, message: 'Too many requests, please try again later.' },
  standardHeaders: true,
  legacyHeaders: false,
});
app.use('/api/', globalLimiter);

// ─── Stricter Limit for Job Creation (GPU is expensive!) ──────────────────────
// Uses user ID as key when authenticated, falls back to IPv6-safe IP helper
const jobCreationLimiter = rateLimit({
  windowMs: 60 * 60 * 1000,
  max: 10,
  message: { success: false, message: 'Job creation limit reached. Max 10 jobs per hour.' },
  keyGenerator: (req, res) => req.user?.id ?? ipKeyGenerator(req),
});

// ─── Body Parsing ──────────────────────────────────────────────────────────────
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// ─── HTTP Request Logging ─────────────────────────────────────────────────────
app.use(morgan('combined', {
  stream: { write: (msg) => logger.http(msg.trim()) },
}));

// ─── Health Check (no auth required) ─────────────────────────────────────────
app.get('/health', (req, res) => {
  res.json({
    success: true,
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: '1.0.0',
    storage: 'cloudinary',
  });
});

// ─── Routes ───────────────────────────────────────────────────────────────────
const authRoutes    = require('./routes/auth.routes');
const uploadRoutes  = require('./routes/upload.routes');
const jobRoutes     = require('./routes/job.routes');
const webhookRoutes = require('./routes/webhook.routes');
const storageRoutes = require('./routes/storage.routes');
const errorHandler  = require('./middleware/errorHandler');

app.use('/api/auth',     authRoutes);
app.use('/api/upload',   uploadRoutes);
app.use('/api/jobs',     jobCreationLimiter, jobRoutes);
app.use('/api/webhooks', webhookRoutes);
app.use('/api/storage',  storageRoutes);

// ─── 404 Fallback ─────────────────────────────────────────────────────────────
// Note: '/{*path}' required by path-to-regexp v8+ (used in Express 5 / Node 24)
app.use('/{*path}', (req, res) => {
  res.status(404).json({ success: false, message: `Route ${req.originalUrl} not found` });
});

// ─── Global Error Handler (must be last) ─────────────────────────────────────
app.use(errorHandler);

module.exports = app;