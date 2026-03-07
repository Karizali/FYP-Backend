const { v4: uuidv4 } = require('uuid');
const { upload } = require('../config/storage');
const Job = require('../models/Job');
const jobQueue = require('../services/jobQueue');
const logger = require('../utils/logger');

// ─── Middleware: assign jobId BEFORE multer runs ───────────────────────────────
// This is needed so storage.js can embed the jobId in the Cloudinary folder path
function assignJobId(req, res, next) {
  req.jobId = uuidv4();
  next();
}

// ─── Middleware: check plan limits before accepting the upload ─────────────────
// Reject early so we don't waste Cloudinary bandwidth on over-limit users
function checkPlanLimit(req, res, next) {
  if (!req.user.canCreateJob()) {
    const remaining = req.user.jobsRemaining;   // virtual from User model
    return res.status(403).json({
      success: false,
      message: `Monthly job limit reached for your "${req.user.plan}" plan. Jobs remaining: ${remaining}.`,
      jobsRemaining: remaining,
      plan: req.user.plan,
    });
  }
  next();
}

// ─── Middleware: validate quality setting against plan ─────────────────────────
function checkQualitySetting(req, res, next) {
  const { quality } = req.body;
  if (quality === 'high' && req.user.plan === 'free') {
    return res.status(403).json({
      success: false,
      message: 'High quality is only available on the Pro plan. Use "fast" or "balanced".',
    });
  }
  next();
}

// ─── Main upload handler ──────────────────────────────────────────────────────
// Composed as an array so Express runs them in sequence as middleware
const handleUpload = [
  checkPlanLimit,
  checkQualitySetting,
  assignJobId,

  // Multer-Cloudinary streams files directly to Cloudinary (no disk touch)
  upload.array('files', 50),

  async (req, res, next) => {
    try {
      if (!req.files || req.files.length === 0) {
        return res.status(400).json({
          success: false,
          message: 'No files uploaded. Include files under the "files" field.',
        });
      }

      // ── Determine input type ────────────────────────────────────────────────
      const mimeTypes = req.files.map((f) => f.mimetype);
      const hasVideo  = mimeTypes.some((m) => m.startsWith('video/'));
      const inputType = hasVideo ? 'video' : 'images';

      if (hasVideo && req.files.length > 1) {
        return res.status(400).json({
          success: false,
          message: 'Only one video file is allowed per job. For images, upload multiple files.',
        });
      }

      // Gaussian splatting needs at least ~20 images for a good result
      if (!hasVideo && req.files.length < 5) {
        return res.status(400).json({
          success: false,
          message: `Too few images (${req.files.length}). Upload at least 5 images for a basic result; 20–50 recommended.`,
        });
      }

      // ── Map Cloudinary upload results to our inputFile sub-schema ───────────
      // multer-storage-cloudinary attaches Cloudinary metadata to each req.file
      const inputFiles = req.files.map((file) => ({
        originalName:  file.originalname,
        cloudinaryId:  file.filename,           // public_id set by CloudinaryStorage
        secureUrl:     file.path,               // secure https URL from Cloudinary
        resourceType:  hasVideo ? 'video' : 'image',
        mimeType:      file.mimetype,
        sizeBytes:     file.size,
        folder:        file.folder || null,
      }));

      // ── Parse job settings ──────────────────────────────────────────────────
      const settings = {
        enhanceImages: req.body.enhanceImages !== 'false',   // default true
        quality:       ['fast', 'balanced', 'high'].includes(req.body.quality)
          ? req.body.quality
          : 'balanced',
      };

      // ── Create job record in MongoDB ────────────────────────────────────────
      const job = await Job.create({
        _id:       req.jobId,
        userId:    req.user._id,
        title:     req.body.title?.trim() || `Scene – ${new Date().toLocaleDateString()}`,
        inputType,
        inputFiles,
        settings,
      });

      // ── Increment user's monthly usage counter ──────────────────────────────
      await req.user.updateOne({ $inc: { jobsThisMonth: 1 } });

      // ── Push job payload onto Bull queue ────────────────────────────────────
      await jobQueue.add(
        {
          jobId:      job._id.toString(),
          userId:     req.user._id.toString(),
          inputFiles,
          inputType,
          settings,
        },
        {
          jobId: job._id.toString(),   // makes Bull job ID match our DB job ID
        }
      );

      logger.info(`Job ${job._id} queued | user=${req.user._id} | files=${req.files.length} | type=${inputType}`);

      res.status(201).json({
        success:          true,
        message:          'Files uploaded successfully. Your 3D scene is being processed.',
        jobId:            job._id,
        status:           job.status,
        filesUploaded:    req.files.length,
        inputType,
        estimatedMinutes: job.estimatedMinutes,   // virtual from Job model
        jobsRemaining:    req.user.jobsRemaining - 1,
      });
    } catch (error) {
      next(error);
    }
  },
];

module.exports = { handleUpload };
