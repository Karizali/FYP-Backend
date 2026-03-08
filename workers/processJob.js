const Job                  = require('../models/Job');
const runpodService        = require('../services/runpod');
const notificationService  = require('../services/notifications');
const logger               = require('../utils/logger');

// ─── Stage progress checkpoints ───────────────────────────────────────────────
const PROGRESS = {
  queued:        0,
  preprocessing: 10,
  training:      30,
  converting:    85,
  done:          100,
};

/**
 * Main job pipeline.
 * Called by worker.js for every Bull job dequeued.
 *
 * Stages:
 *   1. Load job from MongoDB & mark as preprocessing
 *   2. Submit to Runpod GPU endpoint
 *   3. Poll Runpod until complete or failed
 *   4. Update MongoDB with output + mark done
 *   5. Send push notification to user
 */
async function processJob(bullJob) {
  const { jobId, userId, inputFiles, inputType, settings } = bullJob.data;

  // ── 1. Load job & transition to preprocessing ─────────────────────────────
  const job = await Job.findById(jobId);
  if (!job) {
    logger.warn(`processJob: job ${jobId} not found in DB — skipping`);
    return;
  }

  // Guard: skip if already in a terminal state (duplicate delivery)
  if (job.status === 'done' || job.status === 'failed') {
    logger.warn(`processJob: job ${jobId} already in terminal state "${job.status}" — skipping`);
    return;
  }

  await job.transition('preprocessing', { progressPct: PROGRESS.preprocessing });
  await bullJob.progress(PROGRESS.preprocessing);

  logger.info(`processJob: [${jobId}] stage=preprocessing | enhance=${settings.enhanceImages} | quality=${settings.quality}`);

  try {
    // ── 2. Submit to Runpod ─────────────────────────────────────────────────
    // Runpod receives the input file URLs (Cloudinary secure URLs),
    // runs COLMAP + Gaussian splatting, converts to GLB, then calls our webhook.
    // We also poll as a fallback in case the webhook is missed.
    const runpodJobId = await runpodService.submitJob({
      jobId,
      inputFiles:    inputFiles.map(f => ({
        url:          f.secureUrl,
        originalName: f.originalName,
        resourceType: f.resourceType,
      })),
      inputType,
      settings: {
        enhanceImages: settings.enhanceImages,
        quality:       settings.quality,
        // Map quality to concrete training iterations for the GPU worker
        iterations:    qualityToIterations(settings.quality),
      },
      webhookUrl: `${process.env.API_BASE_URL}/api/webhooks/runpod`,
    });

    // Save Runpod's job ID so the webhook can match it back to our job
    await job.transition('training', {
      runpodJobId,
      progressPct: PROGRESS.training,
    });
    await bullJob.progress(PROGRESS.training);

    logger.info(`processJob: [${jobId}] submitted to Runpod | runpodJobId=${runpodJobId}`);

    // Notify user that GPU training has started
    await notificationService.notifyJobStarted(userId, jobId, job.estimatedMinutes);

    // ── 3. Poll Runpod until done (webhook is primary, polling is fallback) ──
    const result = await runpodService.pollUntilDone(runpodJobId, jobId, async (status, pct) => {
      // Progress callback — update DB and Bull progress bar
      const mappedPct = Math.round(PROGRESS.training + (pct / 100) * (PROGRESS.converting - PROGRESS.training));
      await Job.findByIdAndUpdate(jobId, { progressPct: mappedPct });
      await bullJob.progress(mappedPct);
    });

    // ── 4. Transition to converting then done ────────────────────────────────
    await job.transition('converting', { progressPct: PROGRESS.converting });
    await bullJob.progress(PROGRESS.converting);

    await job.transition('done', {
      progressPct: PROGRESS.done,
      output: {
        glbCloudinaryId:       result.glbCloudinaryId,
        glbSecureUrl:          result.glbSecureUrl,
        thumbnailCloudinaryId: result.thumbnailCloudinaryId || null,
        thumbnailSecureUrl:    result.thumbnailSecureUrl    || null,
        fileSizeBytes:         result.fileSizeBytes         || null,
      },
    });
    await bullJob.progress(PROGRESS.done);

    logger.info(`processJob: [${jobId}] DONE | glb=${result.glbSecureUrl}`);

    // ── 5. Push notification ──────────────────────────────────────────────────
    await notificationService.notifyJobComplete(userId, jobId, 'done');

  } catch (err) {
    logger.error(`processJob: [${jobId}] FAILED at stage="${job.status}" — ${err.message}`);

    // Mark job as failed in MongoDB with structured error info
    await job.fail(
      err.userMessage || err.message,
      err.code        || 'PROCESSING_ERROR',
      err.stage       || job.status,
    ).catch(failErr => logger.error(`processJob: could not mark job as failed: ${failErr.message}`));

    // Notify user of failure
    await notificationService.notifyJobComplete(userId, jobId, 'failed').catch(() => {});

    // Re-throw so Bull records the failure and triggers retry logic
    throw err;
  }
}

// ─── Map quality setting to Gaussian splatting iteration count ─────────────────
function qualityToIterations(quality) {
  const map = {
    fast:     7_000,   //  ~5 min
    balanced: 30_000,  // ~15 min
    high:     100_000, // ~30 min
  };
  return map[quality] || 30_000;
}

module.exports = processJob;
