const express = require('express');
const { param, query } = require('express-validator');
const { authenticate } = require('../middleware/auth');
const { listJobs, getJob, getJobResult, deleteJob } = require('../controllers/job.controller');

const router = express.Router();

// All job routes require a valid JWT
router.use(authenticate);

// ─── Validation ────────────────────────────────────────────────────────────────

const jobIdParam = [
  param('id')
    .isMongoId().withMessage('Invalid job ID format.'),
];

const listQueryRules = [
  query('page')
    .optional()
    .isInt({ min: 1 }).withMessage('page must be a positive integer.'),
  query('limit')
    .optional()
    .isInt({ min: 1, max: 50 }).withMessage('limit must be between 1 and 50.'),
  query('status')
    .optional()
    .isIn(['queued', 'preprocessing', 'training', 'converting', 'done', 'failed'])
    .withMessage('Invalid status filter.'),
];

// ─── Routes ────────────────────────────────────────────────────────────────────

/**
 * GET /api/jobs
 * Query params: ?page=1&limit=10&status=done
 * Returns paginated list of the user's jobs (newest first)
 */
router.get('/', listQueryRules, listJobs);

/**
 * GET /api/jobs/:id
 * Poll this every 5–10 seconds from Flutter to track processing progress.
 * Returns: { status, progressPct, estimatedMinutes, timeline, … }
 */
router.get('/:id', jobIdParam, getJob);

/**
 * GET /api/jobs/:id/result
 * Only available when job status is "done".
 * Returns time-limited signed Cloudinary URLs:
 *   - glbUrl       (24 hr expiry) — load in Flutter model_viewer
 *   - thumbnailUrl ( 2 hr expiry) — preview image
 */
router.get('/:id/result', jobIdParam, getJobResult);

/**
 * DELETE /api/jobs/:id
 * Soft-deletes the job record and triggers async Cloudinary cleanup.
 * Cannot delete jobs that are currently processing.
 */
router.delete('/:id', jobIdParam, deleteJob);

module.exports = router;
