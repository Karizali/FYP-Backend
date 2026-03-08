const express  = require('express');
const { authenticate } = require('../middleware/auth');
const {
  getUsageStats,
  listJobOutputs,
  deleteJobFiles,
} = require('../storage/cloudinaryStorage');
const { runCleanup } = require('../storage/cleanupCron');
const Job    = require('../models/Job');
const logger = require('../utils/logger');

const router = express.Router();

// All storage routes require authentication
// In production, add an isAdmin middleware here
router.use(authenticate);

// ─── GET /api/storage/stats ───────────────────────────────────────────────────
// Returns Cloudinary usage vs free tier limits.
// Useful for a developer/admin dashboard.
//
// Response:
// {
//   storage:   { usedGB, limitGB, usedPct },
//   bandwidth: { usedGB, limitGB, usedPct },
//   plan:      "free"
// }
router.get('/stats', async (req, res, next) => {
  try {
    const stats = await getUsageStats();
    res.json({ success: true, data: stats });
  } catch (err) {
    next(err);
  }
});

// ─── GET /api/storage/jobs/:id/files ─────────────────────────────────────────
// List all Cloudinary files stored for a specific job.
// Useful for debugging missing output files.
router.get('/jobs/:id/files', async (req, res, next) => {
  try {
    const job = await Job.findOne({ _id: req.params.id, userId: req.user._id });
    if (!job) {
      return res.status(404).json({ success: false, message: 'Job not found.' });
    }

    const files = await listJobOutputs(req.params.id);
    res.json({ success: true, jobId: req.params.id, files });
  } catch (err) {
    next(err);
  }
});

// ─── POST /api/storage/cleanup ────────────────────────────────────────────────
// Manually trigger the cleanup cron (for testing or emergency cleanup).
// Deletes Cloudinary files for all soft-deleted jobs older than 24 hours.
router.post('/cleanup', async (req, res, next) => {
  try {
    logger.info(`Manual cleanup triggered by user ${req.user._id}`);
    const result = await runCleanup();
    res.json({ success: true, data: result });
  } catch (err) {
    next(err);
  }
});

// ─── DELETE /api/storage/jobs/:id/files ──────────────────────────────────────
// Force-delete all Cloudinary files for a job immediately.
// Only works on jobs that belong to the authenticated user.
router.delete('/jobs/:id/files', async (req, res, next) => {
  try {
    const job = await Job.findOne({ _id: req.params.id, userId: req.user._id });
    if (!job) {
      return res.status(404).json({ success: false, message: 'Job not found.' });
    }

    const result = await deleteJobFiles(req.params.id);
    logger.info(`Force-deleted Cloudinary files for job ${req.params.id} by user ${req.user._id}`);
    res.json({ success: true, data: result });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
