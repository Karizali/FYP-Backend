const Bull = require('bull');
const logger = require('../utils/logger');

// ─── Queue Definition ──────────────────────────────────────────────────────────
// One queue handles all Gaussian splatting jobs.
// Bull persists jobs in Redis so they survive server restarts.
const jobQueue = new Bull('gaussian-processing', {
  redis: process.env.REDIS_URL || 'redis://localhost:6379',

  defaultJobOptions: {
    attempts:  2,                              // try once, retry once on failure
    backoff: {
      type:  'fixed',
      delay: 30_000,                           // wait 30s before retry
    },
    timeout:          30 * 60 * 1000,          // 30 min max — GPU jobs are slow
    removeOnComplete: 100,                     // keep last 100 completed in Redis
    removeOnFail:     200,                     // keep last 200 failed for debugging
  },
});

// ─── Queue Event Logging ───────────────────────────────────────────────────────

jobQueue.on('waiting', (jobId) => {
  logger.debug(`Queue: job ${jobId} is waiting`);
});

jobQueue.on('active', (job) => {
  logger.info(`Queue: job ${job.data.jobId} started processing`);
});

jobQueue.on('progress', (job, progress) => {
  logger.debug(`Queue: job ${job.data.jobId} progress ${progress}%`);
});

jobQueue.on('completed', (job) => {
  logger.info(`Queue: job ${job.data.jobId} completed`);
});

jobQueue.on('failed', (job, err) => {
  logger.error(`Queue: job ${job.data.jobId} failed — ${err.message}`, {
    attempt:  job.attemptsMade,
    maxTries: job.opts.attempts,
  });
});

jobQueue.on('stalled', (job) => {
  // Happens when a worker crashes mid-job — Bull will auto-retry
  logger.warn(`Queue: job ${job.data.jobId} stalled, will be retried`);
});

jobQueue.on('error', (err) => {
  logger.error(`Queue error: ${err.message}`);
});

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * Add a new Gaussian splatting job to the queue.
 *
 * @param {Object} payload
 * @param {string}   payload.jobId       - MongoDB Job._id
 * @param {string}   payload.userId      - MongoDB User._id
 * @param {Array}    payload.inputFiles  - array of { cloudinaryId, secureUrl, … }
 * @param {string}   payload.inputType   - "images" | "video"
 * @param {Object}   payload.settings    - { enhanceImages, quality }
 *
 * @returns {Promise<Bull.Job>}
 */
async function addJob(payload) {
  const { jobId } = payload;

  const bullJob = await jobQueue.add(payload, {
    jobId,            // makes Bull job ID match our MongoDB Job._id
    priority: _getPriority(payload.settings?.quality),
  });

  logger.info(`Queue: enqueued job ${jobId} (Bull id: ${bullJob.id})`);
  return bullJob;
}

/**
 * Fetch the current queue statistics.
 * Useful for an admin dashboard or health check endpoint.
 *
 * @returns {Promise<Object>}
 */
async function getQueueStats() {
  const [waiting, active, completed, failed, delayed] = await Promise.all([
    jobQueue.getWaitingCount(),
    jobQueue.getActiveCount(),
    jobQueue.getCompletedCount(),
    jobQueue.getFailedCount(),
    jobQueue.getDelayedCount(),
  ]);

  return { waiting, active, completed, failed, delayed };
}

/**
 * Retrieve a specific Bull job by its ID (same as MongoDB Job._id).
 * Returns null if not found.
 *
 * @param {string} jobId
 * @returns {Promise<Bull.Job|null>}
 */
async function getBullJob(jobId) {
  return jobQueue.getJob(jobId);
}

/**
 * Retry a failed Bull job by its ID.
 * Used by admin tooling — not exposed to regular users.
 *
 * @param {string} jobId
 */
async function retryJob(jobId) {
  const job = await getBullJob(jobId);
  if (!job) throw new Error(`Bull job ${jobId} not found`);
  await job.retry();
  logger.info(`Queue: manually retried job ${jobId}`);
}

/**
 * Gracefully close the queue connection.
 * Called on SIGINT/SIGTERM to let in-flight jobs finish.
 */
async function closeQueue() {
  await jobQueue.close();
  logger.info('Queue: connection closed gracefully');
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

// Higher priority number = processed first in Bull
// fast jobs skip ahead of slow high-quality ones
function _getPriority(quality) {
  const map = { fast: 1, balanced: 2, high: 3 };
  return map[quality] || 2;
}

// ─── Graceful shutdown ────────────────────────────────────────────────────────
process.on('SIGTERM', closeQueue);
process.on('SIGINT',  closeQueue);

// ─── Exports ──────────────────────────────────────────────────────────────────
// Export the raw queue instance (needed by Bull Board dashboard if you add it)
// AND the helper functions for clean usage in controllers
jobQueue.addJob      = addJob;
jobQueue.getStats    = getQueueStats;
jobQueue.getBullJob  = getBullJob;
jobQueue.retryJob    = retryJob;
jobQueue.closeQueue  = closeQueue;

module.exports = jobQueue;
