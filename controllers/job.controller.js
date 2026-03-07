const Job = require('../models/Job');
const { getSignedUrl, deleteJobFiles } = require('../config/storage');
const logger = require('../utils/logger');

// ─── GET /api/jobs ────────────────────────────────────────────────────────────
// List all jobs for the authenticated user, newest first, paginated
async function listJobs(req, res, next) {
  try {
    const page  = Math.max(1, parseInt(req.query.page)  || 1);
    const limit = Math.min(50, parseInt(req.query.limit) || 10);
    const skip  = (page - 1) * limit;

    // Optional status filter: GET /api/jobs?status=done
    const filter = {
      userId:    req.user._id,
      deletedAt: null,                              // exclude soft-deleted
    };
    if (req.query.status) filter.status = req.query.status;

    const [jobs, total] = await Promise.all([
      Job.find(filter)
        .sort({ createdAt: -1 })
        .skip(skip)
        .limit(limit)
        .select('-inputFiles -output.glbCloudinaryId -output.thumbnailCloudinaryId'),
      Job.countDocuments(filter),
    ]);

    res.json({
      success: true,
      data: jobs.map((j) => j.toSummary()),
      pagination: {
        page,
        limit,
        total,
        pages: Math.ceil(total / limit),
        hasNext: page * limit < total,
      },
    });
  } catch (error) {
    next(error);
  }
}

// ─── GET /api/jobs/:id ────────────────────────────────────────────────────────
// Poll this endpoint every 5–10 seconds from Flutter to track progress
async function getJob(req, res, next) {
  try {
    const job = await Job.findOne({
      _id:       req.params.id,
      userId:    req.user._id,
      deletedAt: null,
    });

    if (!job) {
      return res.status(404).json({ success: false, message: 'Job not found.' });
    }

    res.json({
      success: true,
      data:    job.toSummary(),
    });
  } catch (error) {
    next(error);
  }
}

// ─── GET /api/jobs/:id/result ─────────────────────────────────────────────────
// Returns a time-limited signed URL to download the GLB file
// Only works when job.status === 'done'
async function getJobResult(req, res, next) {
  try {
    const job = await Job.findOne({
      _id:       req.params.id,
      userId:    req.user._id,
      deletedAt: null,
    });

    if (!job) {
      return res.status(404).json({ success: false, message: 'Job not found.' });
    }

    if (job.status !== 'done') {
      return res.status(400).json({
        success:     false,
        message:     `Job is not complete yet. Current status: "${job.status}".`,
        status:      job.status,
        progressPct: job.progressPct,
      });
    }

    if (!job.output?.glbCloudinaryId) {
      logger.error(`Job ${job._id} is "done" but has no GLB cloudinaryId`);
      return res.status(500).json({
        success: false,
        message: 'Output file missing. Please contact support.',
      });
    }

    // Generate signed Cloudinary URLs (GLB: 24hr, thumbnail: 2hr)
    const [glbUrl, thumbnailUrl] = await Promise.all([
      getSignedUrl(job.output.glbCloudinaryId, 'raw',   86400),
      job.output.thumbnailCloudinaryId
        ? getSignedUrl(job.output.thumbnailCloudinaryId, 'image', 7200)
        : null,
    ]);

    res.json({
      success: true,
      data: {
        jobId:           job._id,
        title:           job.title,
        glbUrl,                                    // load in Flutter model_viewer
        thumbnailUrl,
        glbExpiresAt:    new Date(Date.now() + 86400 * 1000).toISOString(),
        fileSizeBytes:   job.output.fileSizeBytes,
        durationSeconds: job.durationSeconds,
        completedAt:     job.timeline.completedAt,
      },
    });
  } catch (error) {
    next(error);
  }
}

// ─── DELETE /api/jobs/:id ─────────────────────────────────────────────────────
// Soft-deletes the DB record; actual Cloudinary files are purged by a cron job
async function deleteJob(req, res, next) {
  try {
    const job = await Job.findOne({
      _id:       req.params.id,
      userId:    req.user._id,
      deletedAt: null,
    });

    if (!job) {
      return res.status(404).json({ success: false, message: 'Job not found.' });
    }

    // Block deletion of in-flight jobs — could cause GPU worker corruption
    if (job.isProcessing) {
      return res.status(400).json({
        success: false,
        message: `Cannot delete a job with status "${job.status}". Wait for it to finish or fail.`,
      });
    }

    // Soft-delete: stamp deletedAt — cron job will cleanup Cloudinary files later
    job.deletedAt = new Date();
    await job.save();

    // Best-effort immediate Cloudinary cleanup (non-blocking, won't fail the request)
    deleteJobFiles(job._id.toString()).catch((err) =>
      logger.warn(`Cloudinary cleanup failed for job ${job._id}: ${err.message}`)
    );

    logger.info(`Job ${job._id} soft-deleted by user ${req.user._id}`);

    res.json({ success: true, message: 'Job deleted.' });
  } catch (error) {
    next(error);
  }
}

module.exports = { listJobs, getJob, getJobResult, deleteJob };
