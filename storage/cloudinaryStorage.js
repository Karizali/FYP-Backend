const cloudinary = require('cloudinary').v2;
const logger     = require('../utils/logger');

// ─── Layer 3: Processed Output Storage ───────────────────────────────────────
// All Cloudinary operations for GPU output files (GLB, thumbnails).
// Input uploads (images/video) are handled by config/storage.js (multer layer).
// This module owns everything that happens AFTER the GPU worker finishes.

// ─── Upload a GLB file ────────────────────────────────────────────────────────
/**
 * Upload the finished .glb file to Cloudinary.
 * Called by the Node.js worker (processJob.js) after Runpod completes,
 * OR the result can come directly from the Python worker via the webhook.
 *
 * @param {string|Buffer} source   - file path, URL, or Buffer
 * @param {string}        jobId    - MongoDB Job._id
 * @param {string}        userId   - MongoDB User._id
 * @returns {Promise<{ cloudinaryId, secureUrl, fileSizeBytes }>}
 */
async function uploadGlb(source, jobId, userId) {
  try {
    logger.info(`Storage: uploading GLB for job ${jobId}`);

    const result = await cloudinary.uploader.upload(source, {
      resource_type: 'raw',                         // GLB is binary, not image/video
      folder:        `gaussian-outputs/${jobId}`,
      public_id:     'scene',
      tags:          [`job_${jobId}`, `user_${userId}`, 'glb', 'output'],
      overwrite:     true,
    });

    logger.info(`Storage: GLB uploaded | id=${result.public_id} | size=${result.bytes}B`);

    return {
      cloudinaryId:  result.public_id,
      secureUrl:     result.secure_url,
      fileSizeBytes: result.bytes,
    };
  } catch (err) {
    logger.error(`Storage: GLB upload failed for job ${jobId}: ${err.message}`);
    const e        = new Error(`Failed to save GLB output: ${err.message}`);
    e.code         = 'STORAGE_GLB_UPLOAD_FAILED';
    e.userMessage  = 'Could not save the 3D scene file. Please try again.';
    throw e;
  }
}

// ─── Upload a thumbnail image ─────────────────────────────────────────────────
/**
 * Stores a JPEG preview thumbnail for the Flutter jobs list screen.
 * Non-fatal — if this fails the job still completes, just without a preview.
 *
 * @param {string|Buffer} source
 * @param {string}        jobId
 * @param {string}        userId
 * @returns {Promise<{ cloudinaryId, secureUrl } | null>}
 */
async function uploadThumbnail(source, jobId, userId) {
  try {
    const result = await cloudinary.uploader.upload(source, {
      resource_type:  'image',
      folder:         `gaussian-outputs/${jobId}`,
      public_id:      'thumbnail',
      tags:           [`job_${jobId}`, `user_${userId}`, 'thumbnail', 'output'],
      overwrite:      true,
      transformation: [
        { width: 512, height: 512, crop: 'fill', gravity: 'center' },
        { quality: 'auto:good', fetch_format: 'auto' },
      ],
    });

    logger.info(`Storage: thumbnail uploaded | id=${result.public_id}`);
    return { cloudinaryId: result.public_id, secureUrl: result.secure_url };
  } catch (err) {
    logger.warn(`Storage: thumbnail upload failed for job ${jobId} (non-fatal): ${err.message}`);
    return null;
  }
}

// ─── Upload from a Node.js stream ────────────────────────────────────────────
/**
 * Used when the GPU worker streams a GLB directly without saving to disk.
 *
 * @param {ReadableStream} stream
 * @param {{ jobId, userId, resourceType?, publicId? }} options
 * @returns {Promise<Object>} Cloudinary upload result
 */
async function uploadStream(stream, options = {}) {
  const { jobId, userId, resourceType = 'raw', publicId = 'scene' } = options;

  return new Promise((resolve, reject) => {
    const up = cloudinary.uploader.upload_stream(
      {
        resource_type: resourceType,
        folder:        `gaussian-outputs/${jobId}`,
        public_id:     publicId,
        tags:          [`job_${jobId}`, `user_${userId}`, 'output'],
        overwrite:     true,
      },
      (err, result) => {
        if (err) {
          logger.error(`Storage: stream upload failed for job ${jobId}: ${err.message}`);
          return reject(err);
        }
        resolve(result);
      }
    );
    stream.pipe(up);
  });
}

// ─── Generate a signed download URL ──────────────────────────────────────────
/**
 * Creates a time-limited private URL for the Flutter app.
 * GLB expires in 24h, thumbnails in 2h.
 *
 * @param {string} cloudinaryId
 * @param {string} resourceType  - 'raw' | 'image'
 * @param {number} expiresIn     - seconds (default 24h)
 * @returns {string} signed URL
 */
function generateDownloadUrl(cloudinaryId, resourceType = 'raw', expiresIn = 86400) {
  const expiresAt = Math.floor(Date.now() / 1000) + expiresIn;

  const url = cloudinary.utils.private_download_url(cloudinaryId, '', {
    resource_type: resourceType,
    expires_at:    expiresAt,
    attachment:    false,
  });

  logger.debug(`Storage: signed URL generated for ${cloudinaryId} (expires in ${expiresIn}s)`);
  return url;
}

// ─── Delete a single file ─────────────────────────────────────────────────────
async function deleteFile(cloudinaryId, resourceType = 'raw') {
  try {
    const result = await cloudinary.uploader.destroy(cloudinaryId, {
      resource_type: resourceType,
      invalidate:    true,   // purge from CDN cache
    });
    logger.info(`Storage: deleted ${cloudinaryId} → ${result.result}`);
    return result.result === 'ok';
  } catch (err) {
    logger.warn(`Storage: delete failed for ${cloudinaryId}: ${err.message}`);
    return false;
  }
}

// ─── Delete all files for a job ───────────────────────────────────────────────
/**
 * Deletes ALL Cloudinary resources tagged with job_<jobId>.
 * Covers input images, input video, GLB output, and thumbnail.
 * Called by the cleanup cron job for soft-deleted jobs.
 *
 * @param {string} jobId
 */
async function deleteJobFiles(jobId) {
  const tag     = `job_${jobId}`;
  const deleted = { image: 0, video: 0, raw: 0 };

  for (const resourceType of ['image', 'video', 'raw']) {
    try {
      const result = await cloudinary.api.delete_resources_by_tag(tag, { resource_type: resourceType });
      deleted[resourceType] = Object.keys(result.deleted || {}).length;
      logger.debug(`Storage: deleted ${deleted[resourceType]} ${resourceType} file(s) for job ${jobId}`);
    } catch (err) {
      logger.warn(`Storage: could not delete ${resourceType} files for job ${jobId}: ${err.message}`);
    }
  }

  logger.info(`Storage: cleanup complete for job ${jobId} | deleted=${JSON.stringify(deleted)}`);
  return deleted;
}

// ─── Get Cloudinary usage stats ───────────────────────────────────────────────
/**
 * Free tier limits: 25 GB storage, 25 GB bandwidth/month.
 * Used by the admin /storage/stats route.
 *
 * @returns {Promise<Object>}
 */
async function getUsageStats() {
  const usage = await cloudinary.api.usage();

  const pct = (used, limit) =>
    limit > 0 ? ((used / limit) * 100).toFixed(1) : '0.0';

  return {
    storage: {
      usedBytes:  usage.storage.usage,
      usedGB:     (usage.storage.usage  / 1e9).toFixed(2),
      limitGB:    (usage.storage.limit  / 1e9).toFixed(2),
      usedPct:    pct(usage.storage.usage, usage.storage.limit),
    },
    bandwidth: {
      usedBytes:  usage.bandwidth.usage,
      usedGB:     (usage.bandwidth.usage / 1e9).toFixed(2),
      limitGB:    (usage.bandwidth.limit / 1e9).toFixed(2),
      usedPct:    pct(usage.bandwidth.usage, usage.bandwidth.limit),
    },
    resources:  usage.resources,
    plan:       usage.plan,
    retrievedAt: new Date().toISOString(),
  };
}

// ─── Verify Cloudinary credentials at startup ─────────────────────────────────
async function verifyConnection() {
  try {
    await cloudinary.api.ping();
    logger.info('✅ Cloudinary connected');
    return true;
  } catch (err) {
    logger.error(`❌ Cloudinary connection failed: ${err.message}`);
    return false;
  }
}

// ─── List all output files for a job (admin/debug) ───────────────────────────
async function listJobOutputs(jobId) {
  try {
    const folder = `gaussian-outputs/${jobId}`;
    const [rawRes, imgRes] = await Promise.all([
      cloudinary.api.resources({ type: 'upload', resource_type: 'raw',   prefix: folder, max_results: 10 }),
      cloudinary.api.resources({ type: 'upload', resource_type: 'image', prefix: folder, max_results: 10 }),
    ]);
    return [
      ...rawRes.resources.map(r  => ({ ...r, category: 'glb'       })),
      ...imgRes.resources.map(r  => ({ ...r, category: 'thumbnail' })),
    ];
  } catch (err) {
    logger.warn(`Storage: could not list outputs for job ${jobId}: ${err.message}`);
    return [];
  }
}

module.exports = {
  uploadGlb,
  uploadThumbnail,
  uploadStream,
  generateDownloadUrl,
  deleteFile,
  deleteJobFiles,
  getUsageStats,
  verifyConnection,
  listJobOutputs,
};
