require('dotenv').config();
const app                  = require('./app');
const { connectDB }        = require('./config/database');
const { connectRedis }     = require('./config/redis');
const { verifyConnection } = require('./storage/cloudinaryStorage');
const { startCleanupCron } = require('./storage/cleanupCron');
const logger               = require('./utils/logger');

const PORT = process.env.PORT || 3000;

async function startServer() {
  try {
    // ── 1. Database connections ───────────────────────────────────────────────
    await connectDB();
    await connectRedis();

    // ── 2. Verify Cloudinary (Layer 3) ────────────────────────────────────────
    // Non-fatal: logs a warning but doesn't crash — uploads will fail at runtime
    await verifyConnection();

    // ── 3. Start nightly cleanup cron (Layer 3) ───────────────────────────────
    startCleanupCron();

    // ── 4. Start HTTP server ──────────────────────────────────────────────────
    app.listen(PORT, () => {
      logger.info(`🚀 Gaussian Splatting API running on port ${PORT}`);
      logger.info(`📦 Environment: ${process.env.NODE_ENV}`);
      logger.info(`🗂️  Storage: Cloudinary`);
      logger.info(`🧹 Cleanup cron: active (daily 02:00 UTC)`);
    });
  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

startServer();