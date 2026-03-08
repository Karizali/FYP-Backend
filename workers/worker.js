require('dotenv').config();

const { connectDB }    = require('../config/database');
const { connectRedis } = require('../config/redis');
const jobQueue         = require('../services/jobQueue');
const processJob       = require('./processJob');
const logger           = require('../utils/logger');

// ─── Concurrency: process 1 GPU job at a time ─────────────────────────────────
// Runpod endpoints have limited concurrency on free tier.
// Increase to 2-3 if you upgrade your Runpod plan.
const CONCURRENCY = parseInt(process.env.WORKER_CONCURRENCY) || 1;

async function startWorker() {
  try {
    await connectDB();
    await connectRedis();

    logger.info(`⚙️  Worker started | concurrency=${CONCURRENCY}`);

    // Bull calls this function for every job it dequeues
    jobQueue.process(CONCURRENCY, async (bullJob) => {
      const { jobId } = bullJob.data;
      logger.info(`Worker: picked up job ${jobId}`);

      try {
        await processJob(bullJob);
      } catch (err) {
        // Log and re-throw so Bull marks the job as failed and retries
        logger.error(`Worker: job ${jobId} threw unhandled error — ${err.message}`, {
          stack: err.stack,
        });
        throw err;
      }
    });

    // ── Worker-level event hooks ──────────────────────────────────────────
    jobQueue.on('active',    (job)      => logger.info(`Worker: active  job=${job.data.jobId}`));
    jobQueue.on('completed', (job)      => logger.info(`Worker: done    job=${job.data.jobId}`));
    jobQueue.on('failed',    (job, err) => logger.error(`Worker: failed  job=${job.data.jobId} — ${err.message} (attempt ${job.attemptsMade})`));

    logger.info('Worker is listening for jobs...');
  } catch (err) {
    logger.error(`Worker failed to start: ${err.message}`);
    process.exit(1);
  }
}

startWorker();
