const axios  = require('axios');
const logger = require('../utils/logger');

// ─── Runpod Serverless API ────────────────────────────────────────────────────
// Docs: https://docs.runpod.io/serverless/endpoints/job-operations
//
// Flow:
//   POST /run        → submits job, returns { id }
//   GET  /status/:id → returns { status, output, error }
//
// Statuses from Runpod:
//   IN_QUEUE → IN_PROGRESS → COMPLETED | FAILED | CANCELLED | TIMED_OUT

const RUNPOD_BASE = 'https://api.runpod.ai/v2';

// Polling config
const POLL_INTERVAL_MS  = parseInt(process.env.RUNPOD_POLL_INTERVAL_MS)  || 15_000; // 15s
const POLL_TIMEOUT_MS   = parseInt(process.env.RUNPOD_POLL_TIMEOUT_MS)   || 40 * 60 * 1000; // 40 min max

function getHeaders() {
  return {
    'Authorization': `Bearer ${process.env.RUNPOD_API_KEY}`,
    'Content-Type':  'application/json',
  };
}

// ─── Submit a job to Runpod Serverless endpoint ───────────────────────────────
/**
 * @param {Object} payload
 * @param {string}   payload.jobId        - Our MongoDB job ID (sent back in webhook)
 * @param {Array}    payload.inputFiles   - [{ url, originalName, resourceType }]
 * @param {string}   payload.inputType    - "images" | "video"
 * @param {Object}   payload.settings     - { enhanceImages, quality, iterations }
 * @param {string}   payload.webhookUrl   - URL Runpod will POST results to
 * @returns {Promise<string>} Runpod job ID
 */
async function submitJob(payload) {
  const endpointId = process.env.RUNPOD_ENDPOINT_ID;
  if (!endpointId) throw new Error('RUNPOD_ENDPOINT_ID is not set in environment');

  const url = `${RUNPOD_BASE}/${endpointId}/run`;

  // This is the payload your GPU worker script receives as `job["input"]`
  const body = {
    input: {
      job_id:       payload.jobId,
      input_files:  payload.inputFiles,
      input_type:   payload.inputType,
      enhance:      payload.settings.enhanceImages,
      quality:      payload.settings.quality,
      iterations:   payload.settings.iterations,
      webhook_url:  payload.webhookUrl,
      webhook_secret: process.env.RUNPOD_WEBHOOK_SECRET,
    },
    webhook: payload.webhookUrl, // Runpod's native webhook (backup notification)
  };

  try {
    const res = await axios.post(url, body, { headers: getHeaders(), timeout: 30_000 });

    if (!res.data?.id) {
      throw new Error(`Runpod did not return a job ID. Response: ${JSON.stringify(res.data)}`);
    }

    logger.info(`Runpod: submitted job | runpodId=${res.data.id} | ourJobId=${payload.jobId}`);
    return res.data.id;

  } catch (err) {
    const message = err.response?.data?.error || err.message;
    const code    = err.response?.status;

    logger.error(`Runpod submit failed [${code}]: ${message}`);

    const appErr      = new Error(`Failed to submit job to GPU: ${message}`);
    appErr.code       = 'RUNPOD_SUBMIT_FAILED';
    appErr.stage      = 'preprocessing';
    appErr.userMessage = 'Could not start processing. Please try again.';
    throw appErr;
  }
}

// ─── Get the current status of a Runpod job ───────────────────────────────────
/**
 * @param {string} runpodJobId
 * @returns {Promise<Object>} { status, output, error }
 */
async function getJobStatus(runpodJobId) {
  const endpointId = process.env.RUNPOD_ENDPOINT_ID;
  const url        = `${RUNPOD_BASE}/${endpointId}/status/${runpodJobId}`;

  const res = await axios.get(url, { headers: getHeaders(), timeout: 15_000 });
  return res.data;
}

// ─── Poll until Runpod job completes (fallback to webhook) ────────────────────
/**
 * Polls Runpod every POLL_INTERVAL_MS until the job reaches a terminal state.
 * The webhook is the primary notification path — this is the safety net.
 *
 * @param {string}   runpodJobId  - Runpod job ID to poll
 * @param {string}   ourJobId     - Our MongoDB job ID (for logging)
 * @param {Function} onProgress   - Callback(status, progressPct) for intermediate updates
 * @returns {Promise<Object>} The output from the GPU worker
 */
async function pollUntilDone(runpodJobId, ourJobId, onProgress) {
  const deadline = Date.now() + POLL_TIMEOUT_MS;
  let   attempt  = 0;

  // Map Runpod statuses to rough progress percentages
  const progressMap = {
    IN_QUEUE:    5,
    IN_PROGRESS: 50,
    COMPLETED:   100,
  };

  while (Date.now() < deadline) {
    attempt++;
    await sleep(POLL_INTERVAL_MS);

    let statusData;
    try {
      statusData = await getJobStatus(runpodJobId);
    } catch (pollErr) {
      logger.warn(`Runpod poll attempt ${attempt} failed: ${pollErr.message} — retrying`);
      continue;  // transient network error, keep trying
    }

    const { status, output, error } = statusData;
    logger.debug(`Runpod poll [attempt=${attempt}] runpodId=${runpodJobId} status=${status}`);

    // Report intermediate progress
    const pct = progressMap[status] || 50;
    if (onProgress) await onProgress(status, pct).catch(() => {});

    // ── Terminal states ─────────────────────────────────────────────────────
    if (status === 'COMPLETED') {
      if (!output) {
        const err      = new Error('Runpod job completed but returned no output');
        err.code       = 'MISSING_OUTPUT';
        err.stage      = 'converting';
        err.userMessage = 'Processing completed but the output file is missing.';
        throw err;
      }
      logger.info(`Runpod: job ${runpodJobId} COMPLETED after ${attempt} polls`);
      return parseRunpodOutput(output);
    }

    if (status === 'FAILED') {
      const message   = error || 'GPU worker reported failure';
      const err       = new Error(`Runpod FAILED: ${message}`);
      err.code        = parseErrorCode(message);
      err.stage       = 'training';
      err.userMessage = humanizeError(message);
      throw err;
    }

    if (status === 'CANCELLED') {
      const err       = new Error('Runpod job was cancelled');
      err.code        = 'JOB_CANCELLED';
      err.stage       = 'training';
      err.userMessage = 'Processing was cancelled. Please try again.';
      throw err;
    }

    if (status === 'TIMED_OUT') {
      const err       = new Error('Runpod job timed out on the GPU worker');
      err.code        = 'GPU_TIMEOUT';
      err.stage       = 'training';
      err.userMessage = 'Processing took too long. Try using "fast" quality for quicker results.';
      throw err;
    }

    // IN_QUEUE or IN_PROGRESS — keep polling
  }

  // Outer timeout: job exceeded our maximum wait time
  const err       = new Error(`Polling timed out after ${POLL_TIMEOUT_MS / 60000} minutes`);
  err.code        = 'POLL_TIMEOUT';
  err.stage       = 'training';
  err.userMessage = 'Processing is taking too long. We\'ll notify you when it completes.';
  throw err;
}

// ─── Cancel a running Runpod job ──────────────────────────────────────────────
async function cancelJob(runpodJobId) {
  try {
    const endpointId = process.env.RUNPOD_ENDPOINT_ID;
    const url        = `${RUNPOD_BASE}/${endpointId}/cancel/${runpodJobId}`;
    await axios.post(url, {}, { headers: getHeaders(), timeout: 10_000 });
    logger.info(`Runpod: cancelled job ${runpodJobId}`);
  } catch (err) {
    logger.warn(`Runpod: could not cancel job ${runpodJobId}: ${err.message}`);
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Parse and validate the output from the GPU worker.
 * Your Runpod worker script should return this structure:
 * {
 *   glb_cloudinary_id:       string,
 *   glb_secure_url:          string,
 *   thumbnail_cloudinary_id: string | null,
 *   thumbnail_secure_url:    string | null,
 *   file_size_bytes:         number | null,
 * }
 */
function parseRunpodOutput(output) {
  if (!output.glb_cloudinary_id || !output.glb_secure_url) {
    const err       = new Error(`Invalid output from GPU worker: missing GLB fields. Got: ${JSON.stringify(output)}`);
    err.code        = 'INVALID_OUTPUT';
    err.stage       = 'converting';
    err.userMessage = 'Processing finished but the 3D file could not be saved.';
    throw err;
  }

  return {
    glbCloudinaryId:       output.glb_cloudinary_id,
    glbSecureUrl:          output.glb_secure_url,
    thumbnailCloudinaryId: output.thumbnail_cloudinary_id || null,
    thumbnailSecureUrl:    output.thumbnail_secure_url    || null,
    fileSizeBytes:         output.file_size_bytes         || null,
  };
}

// Map common GPU error strings to structured error codes
function parseErrorCode(message = '') {
  const msg = message.toLowerCase();
  if (msg.includes('colmap') || msg.includes('sfm'))   return 'COLMAP_FAILED';
  if (msg.includes('out of memory') || msg.includes('oom')) return 'GPU_OOM';
  if (msg.includes('too few') || msg.includes('not enough images')) return 'TOO_FEW_IMAGES';
  if (msg.includes('cuda'))                            return 'CUDA_ERROR';
  if (msg.includes('timeout'))                         return 'GPU_TIMEOUT';
  return 'WORKER_ERROR';
}

// Map error codes to user-friendly messages
function humanizeError(message = '') {
  const code = parseErrorCode(message);
  const map  = {
    COLMAP_FAILED:    'Could not reconstruct 3D geometry. Try uploading images with more overlap and different angles.',
    GPU_OOM:          'The GPU ran out of memory. Try using "fast" quality or uploading fewer images.',
    TOO_FEW_IMAGES:   'Not enough usable images. Upload at least 20 images from different angles.',
    CUDA_ERROR:       'A GPU error occurred. Please try again.',
    GPU_TIMEOUT:      'Processing took too long. Try "fast" quality for quicker results.',
    WORKER_ERROR:     'An error occurred during processing. Please try again.',
  };
  return map[code] || 'Processing failed. Please try again.';
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

module.exports = { submitJob, getJobStatus, pollUntilDone, cancelJob };
