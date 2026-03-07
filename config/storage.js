const cloudinary = require('cloudinary').v2;
const { CloudinaryStorage } = require('multer-storage-cloudinary');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');
const logger = require('../utils/logger');

// ─── Configure Cloudinary ─────────────────────────────────────────────────────
cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key:    process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
  secure:     true,
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
const storage = new CloudinaryStorage({
  cloudinary,
  params: async (req, file) => {
    const isVideo  = file.mimetype.startsWith('video/');
    const userId   = req.user?.id || 'unknown';
    const jobId    = req.jobId;
    const publicId = uuidv4();

    if (isVideo) {
      return {
        resource_type: 'video',
        folder:        `gaussian-videos/${userId}/${jobId}`,
        public_id:     publicId,
        // ── Fix: async upload so Cloudinary doesn't time out on large videos ──
        // Without this, videos over ~100MB fail with "too large to process synchronously"
        eager_async:   true,
        // Upload in 6MB chunks — prevents timeouts on slow connections
        chunk_size:    6_000_000,
        tags:          [`user_${userId}`, `job_${jobId}`],
      };
    }

    return {
      resource_type: 'image',
      folder:        `gaussian-uploads/${userId}/${jobId}`,
      public_id:     publicId,
      // No transformations at upload — enhancement runs later in the Python worker
      tags:          [`user_${userId}`, `job_${jobId}`],
    };
  },
});

// ─── Multer instance ──────────────────────────────────────────────────────────
const upload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 500 * 1024 * 1024, // 500 MB per file (mobile videos can be large)
    files:    50,
  },
});

// ─── Helper: generate a short-lived signed URL ────────────────────────────────
async function getSignedUrl(publicId, resourceType = 'raw', expiresInSeconds = 86400) {
  try {
    const timestamp = Math.floor(Date.now() / 1000) + expiresInSeconds;
    return cloudinary.utils.private_download_url(publicId, '', {
      resource_type: resourceType,
      expires_at:    timestamp,
      attachment:    false,
    });
  } catch (error) {
    logger.error(`Failed to generate signed URL for ${publicId}: ${error.message}`);
    throw error;
  }
}

// ─── Helper: upload a buffer directly (for GLB output from GPU worker) ────────
async function uploadBuffer(buffer, options = {}) {
  return new Promise((resolve, reject) => {
    const uploadStream = cloudinary.uploader.upload_stream(
      {
        resource_type: 'raw',
        folder:        options.folder    || 'gaussian-outputs',
        public_id:     options.publicId  || uuidv4(),
        tags:          options.tags      || [],
      },
      (error, result) => {
        if (error) return reject(error);
        resolve(result);
      }
    );
    uploadStream.end(buffer);
  });
}

// ─── Helper: delete a single file ─────────────────────────────────────────────
async function deleteFile(publicId, resourceType = 'image') {
  try {
    const result = await cloudinary.uploader.destroy(publicId, { resource_type: resourceType });
    logger.info(`Cloudinary: deleted ${publicId} → ${result.result}`);
    return result;
  } catch (error) {
    logger.error(`Cloudinary delete failed for ${publicId}: ${error.message}`);
  }
}

// ─── Helper: delete all files for a job ──────────────────────────────────────
async function deleteJobFiles(jobId) {
  try {
    await cloudinary.api.delete_resources_by_tag(`job_${jobId}`, { resource_type: 'image' });
    await cloudinary.api.delete_resources_by_tag(`job_${jobId}`, { resource_type: 'video' });
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