const crypto = require('crypto');
const Job = require('../models/Job');
// GPU worker uploads GLB to Cloudinary directly — webhook just saves the IDs
const notificationService = require('../services/notifications');
const logger = require('../utils/logger');

// ─── HMAC-SHA256 signature verification ───────────────────────────────────────
// Runpod signs each webhook payload so we know it genuinely came from our worker
function verifySignature(req) {
  const signature = req.headers['x-runpod-signature'];
  if (!signature) return false;

  const expected = crypto
    .createHmac('sha256', process.env.RUNPOD_WEBHOOK_SECRET)
    .update(JSON.stringify(req.body))
    .digest('hex');

  try {
    return crypto.timingSafeEqual(
      Buffer.from(signature, 'hex'),
      Buffer.from(expected,  'hex')
    );
  } catch {
    // Buffer lengths differ → invalid signature
    return false;
  }
}

// ─── POST /api/webhooks/runpod ────────────────────────────────────────────────
//
// The GPU worker calls this endpoint to report:
//   • Progress updates during training
//   • Completion with the finished GLB file (as base64 or Cloudinary public_id)
//   • Failures with error details
//
// Expected payload:
// {
//   jobId:            string,           // matches our MongoDB Job._id
//   status:           string,           // see statusMap below
//   progressPct:      number,           // 0–100
//   runpodJobId:      string,           // Runpod's own job ID (for debugging)
//
//   // On completion — worker uploads GLB to Cloudinary and sends back these:
//   glbCloudinaryId:       string | null,
//   glbSecureUrl:          string | null,
//   thumbnailCloudinaryId: string | null,
//   thumbnailSecureUrl:    string | null,
//   glbFileSizeBytes:      number | null,
//
//   // On failure:
//   errorMessage:     string | null,
//   errorCode:        string | null,    // e.g. "COLMAP_FAILED", "GPU_OOM"
//   errorStage:       string | null,    // which stage failed
// }
//
async function handleRunpodWebhook(req, res, next) {
  try {
    // ── 1. Verify this request came from our GPU worker ─────────────────────
    if (!verifySignature(req)) {
      logger.warn(`Webhook: invalid signature from ${req.ip}`);
      return res.status(401).json({ success: false, message: 'Invalid signature.' });
    }

    const {
      jobId,
      status,
      progressPct,
      runpodJobId,
      glbCloudinaryId,
      glbSecureUrl,
      thumbnailCloudinaryId,
      thumbnailSecureUrl,
      glbFileSizeBytes,
      errorMessage,
      errorCode,
      errorStage,
    } = req.body;

    if (!jobId || !status) {
      return res.status(400).json({
        success: false,
        message: 'Payload must include "jobId" and "status".',
      });
    }

    // ── 2. Load the job ─────────────────────────────────────────────────────
    const job = await Job.findById(jobId);
    if (!job) {
      // Don't 404 — return 200 so Runpod doesn't keep retrying
      logger.warn(`Webhook: received update for unknown job "${jobId}"`);
      return res.json({ success: true, message: 'Job not found, ignored.' });
    }

    // Skip duplicate terminal-state callbacks (done/failed are immutable)
    if (job.status === 'done' || job.status === 'failed') {
      logger.info(`Webhook: job ${jobId} already in terminal state "${job.status}", skipping.`);
      return res.json({ success: true, message: 'Already in terminal state.' });
    }

    logger.info(`Webhook: job ${jobId} | ${job.status} → ${status} (${progressPct ?? '?'}%)`);

    // ── 3. Map Runpod status string to our internal status ──────────────────
    const statusMap = {
      preprocessing: 'preprocessing',
      training:      'training',
      converting:    'converting',
      completed:     'done',
      failed:        'failed',
    };

    const newStatus = statusMap[status];
    if (!newStatus) {
      logger.warn(`Webhook: unknown status "${status}" for job ${jobId}`);
      return res.status(400).json({ success: false, message: `Unknown status: "${status}".` });
    }

    // ── 4. Handle failure ───────────────────────────────────────────────────
    if (newStatus === 'failed') {
      await job.fail(
        errorMessage || 'Processing failed on GPU worker.',
        errorCode    || 'WORKER_ERROR',
        errorStage   || job.status
      );

      await notificationService.notifyJobComplete(job.userId, jobId, 'failed');
      return res.json({ success: true });
    }

    // ── 5. Build extra fields to merge into the job ─────────────────────────
    const extra = {
      progressPct: progressPct ?? job.progressPct,
    };

    if (runpodJobId) extra.runpodJobId = runpodJobId;

    // Attach output files when the worker reports completion
    if (newStatus === 'done') {
      if (!glbCloudinaryId || !glbSecureUrl) {
        logger.error(`Webhook: job ${jobId} marked "completed" but missing GLB info`);
        await job.fail('Worker reported completion but GLB output is missing.', 'MISSING_OUTPUT');
        return res.json({ success: true });
      }

      extra.output = {
        glbCloudinaryId,
        glbSecureUrl,
        thumbnailCloudinaryId: thumbnailCloudinaryId || null,
        thumbnailSecureUrl:    thumbnailSecureUrl    || null,
        fileSizeBytes:         glbFileSizeBytes      || null,
      };
      extra.progressPct = 100;
    }

    // ── 6. Advance the state machine ────────────────────────────────────────
    try {
      await job.transition(newStatus, extra);
    } catch (transitionError) {
      // e.g. invalid transition (out-of-order webhook) — log but don't crash
      logger.warn(`Webhook: ${transitionError.message} (job ${jobId})`);
      return res.json({ success: true, message: transitionError.message });
    }

    // ── 7. Push notification on completion ──────────────────────────────────
    if (newStatus === 'done') {
      await notificationService.notifyJobComplete(job.userId, jobId, 'done');
    }

    res.json({ success: true });
  } catch (error) {
    // Return 200 so Runpod doesn't retry — log the error internally
    logger.error(`Webhook handler error for job ${req.body?.jobId}: ${error.message}`, {
      stack: error.stack,
    });
    res.json({ success: false, message: 'Internal error. Logged for review.' });
  }
}

module.exports = { handleRunpodWebhook };