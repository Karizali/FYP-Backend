const Job                  = require('../models/Job');
const { deleteJobFiles }   = require('../storage/cloudinaryStorage');
const logger               = require('../utils/logger');

// ─── Cleanup Cron ─────────────────────────────────────────────────────────────
// Runs every night at 02:00. Finds jobs that were soft-deleted (deletedAt is set)
// but still have Cloudinary files, purges them, then hard-deletes the DB record.
//
// Why cron instead of immediate delete?
//   • deleteJob() does a best-effort immediate cleanup but it can fail
//   • This cron is the safety net that guarantees files are eventually removed
//   • Keeps the DELETE /jobs/:id response fast (no blocking on Cloudinary)

const BATCH_SIZE         = 20;    // jobs per run
const HARD_DELETE_AFTER  = 24;    // hours after soft-delete before DB record is removed

// ─── Main cleanup task ────────────────────────────────────────────────────────
async function runCleanup() {
  logger.info('Cleanup cron: starting run');

  const cutoff = new Date(Date.now() - HARD_DELETE_AFTER * 60 * 60 * 1000);

  // Find jobs that were soft-deleted more than HARD_DELETE_AFTER hours ago
  const jobs = await Job.find({
    deletedAt: { $ne: null, $lte: cutoff },
  })
    .select('_id userId deletedAt')
    .limit(BATCH_SIZE)
    .lean();

  if (jobs.length === 0) {
    logger.info('Cleanup cron: nothing to clean up');
    return { cleaned: 0, failed: 0 };
  }

  logger.info(`Cleanup cron: found ${jobs.length} job(s) to purge`);

  let cleaned = 0;
  let failed  = 0;

  for (const job of jobs) {
    try {
      // 1. Delete all Cloudinary files (input + output)
      await deleteJobFiles(job._id.toString());

      // 2. Hard-delete the MongoDB record
      await Job.findByIdAndDelete(job._id);

      cleaned++;
      logger.info(`Cleanup cron: purged job ${job._id}`);
    } catch (err) {
      failed++;
      logger.error(`Cleanup cron: failed to purge job ${job._id}: ${err.message}`);
      // Leave it for the next run — don't hard-delete if Cloudinary cleanup failed
    }
  }

  logger.info(`Cleanup cron: done | cleaned=${cleaned} failed=${failed}`);
  return { cleaned, failed };
}

// ─── Schedule: run every night at 02:00 ──────────────────────────────────────
function startCleanupCron() {
  // Simple interval-based scheduler — no external cron library needed
  // 02:00 UTC every night
  scheduleDaily(2, 0, () => {
    runCleanup().catch(err =>
      logger.error(`Cleanup cron: unhandled error — ${err.message}`)
    );
  });

  logger.info('Cleanup cron: scheduled (daily at 02:00 UTC)');
}

// ─── Helper: schedule a task at a specific hour/minute daily ──────────────────
function scheduleDaily(hour, minute, fn) {
  function msUntilNext() {
    const now    = new Date();
    const target = new Date();
    target.setUTCHours(hour, minute, 0, 0);
    if (target <= now) target.setUTCDate(target.getUTCDate() + 1);
    return target - now;
  }

  function schedule() {
    setTimeout(() => {
      fn();
      // After first run, repeat exactly every 24 hours
      setInterval(fn, 24 * 60 * 60 * 1000);
    }, msUntilNext());
  }

  schedule();
}

module.exports = { startCleanupCron, runCleanup };
