const cloudinary = require('cloudinary').v2;
const { CloudinaryStorage } = require('multer-storage-cloudinary');
const multer = require('multer');
const path = require('path');
const { v4: uuidv4 } = require('uuid');
const logger = require('../utils/logger');

// ─── Configure Cloudinary ─────────────────────────────────────────────────────
// Free tier: 25 GB storage, 25 GB bandwidth/month — plenty for FYP
cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key:    process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
  secure: true,
});

// ─── Allowed MIME types ────────────────────────────────────────────────────────
const ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const ALLOWED_VIDEO_TYPES = ['video/mp4', 'video/quicktime', 'video/x-msvideo'];
const ALLOWED_TYPES       = [...ALLOWED_IMAGE_TYPES, ...ALLOWED_VIDEO_TYPES];

// ─── File filter ──────────────────────────────────────────────────────────────
const fileFilter = (req, file, cb) => {
  if (ALLOWED_TYPES.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(
      new Error(
        `Invalid file type: ${file.mimetype}. ` +
        `Allowed: JPEG, PNG, WEBP images or MP4, MOV, AVI videos.`
      ),
      false
    );
  }
};

// ─── Cloudinary Storage Engine ────────────────────────────────────────────────
// Images → stored in  gaussian-uploads/<userId>/<jobId>/
// Videos → stored in  gaussian-videos/<userId>/<jobId>/
const storage = new CloudinaryStorage({
  cloudinary,
  params: async (req, file) => {
    const isVideo   = file.mimetype.startsWith('video/');
    const userId    = req.user?.id || 'unknown';
    const jobId     = req.jobId;                    // set by assignJobId middleware
    const publicId  = `${uuidv4()}`;

    return {
      resource_type: isVideo ? 'video' : 'image',
      folder:        isVideo
        ? `gaussian-videos/${userId}/${jobId}`
        : `gaussian-uploads/${userId}/${jobId}`,
      public_id:     publicId,

      // Images: keep original quality, no transformation at upload time
      // (enhancement happens later in the preprocessing Python worker)
      format: isVideo ? 'mp4' : undefined,

      // Tag files for easy bulk operations (e.g. cleanup old jobs)
      tags: [`user_${userId}`, `job_${jobId}`],
    };
  },
});

// ─── Multer instance ──────────────────────────────────────────────────────────
const upload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 100 * 1024 * 1024, // 100 MB per file
    files:    50,                // max 50 files per request
  },
});

// ─── Helper: generate a short-lived signed URL for a private resource ─────────
// Use this when serving GLB output files securely to authenticated users
async function getSignedUrl(publicId, resourceType = 'raw', expiresInSeconds = 86400) {
  try {
    const timestamp = Math.floor(Date.now() / 1000) + expiresInSeconds;
    const signedUrl = cloudinary.utils.private_download_url(publicId, '', {
      resource_type: resourceType,
      expires_at:    timestamp,
      attachment:    false,
    });
    return signedUrl;
  } catch (error) {
    logger.error(`Failed to generate signed URL for ${publicId}: ${error.message}`);
    throw error;
  }
}

// ─── Helper: upload a file buffer directly (used by GPU worker result handler) ─
// The Runpod worker sends the .glb back as a raw upload, not via multipart form
async function uploadBuffer(buffer, options = {}) {
  return new Promise((resolve, reject) => {
    const uploadStream = cloudinary.uploader.upload_stream(
      {
        resource_type: 'raw',           // GLB files must use 'raw'
        folder:        options.folder || 'gaussian-outputs',
        public_id:     options.publicId || uuidv4(),
        tags:          options.tags || [],
      },
      (error, result) => {
        if (error) return reject(error);
        resolve(result);
      }
    );
    uploadStream.end(buffer);
  });
}

// ─── Helper: delete a file from Cloudinary (cleanup on job delete) ────────────
async function deleteFile(publicId, resourceType = 'image') {
  try {
    const result = await cloudinary.uploader.destroy(publicId, { resource_type: resourceType });
    logger.info(`Cloudinary: deleted ${publicId} → ${result.result}`);
    return result;
  } catch (error) {
    logger.error(`Cloudinary delete failed for ${publicId}: ${error.message}`);
    // Non-fatal — log and continue
  }
}

// ─── Helper: delete all files tagged with a jobId (bulk cleanup) ──────────────
async function deleteJobFiles(jobId) {
  try {
    // Delete images
    await cloudinary.api.delete_resources_by_tag(`job_${jobId}`, { resource_type: 'image' });
    // Delete videos
    await cloudinary.api.delete_resources_by_tag(`job_${jobId}`, { resource_type: 'video' });
    // Delete raw (GLB outputs)
    await cloudinary.api.delete_resources_by_tag(`job_${jobId}`, { resource_type: 'raw' });
    logger.info(`Cloudinary: cleaned up all files for job ${jobId}`);
  } catch (error) {
    logger.error(`Cloudinary bulk delete failed for job ${jobId}: ${error.message}`);
  }
}

module.exports = {
  cloudinary,
  upload,
  getSignedUrl,
  uploadBuffer,
  deleteFile,
  deleteJobFiles,
};
