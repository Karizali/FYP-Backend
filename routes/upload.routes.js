const express = require('express');
const { authenticate } = require('../middleware/auth');
const { handleUpload } = require('../controllers/upload.controller');

const router = express.Router();

// All upload routes require a valid JWT
router.use(authenticate);

/**
 * POST /api/upload
 *
 * Accepts multipart/form-data. Streams files directly to Cloudinary.
 *
 * Form fields:
 * ┌─────────────────┬──────────────────────────────────────────────────────────┐
 * │ Field           │ Description                                              │
 * ├─────────────────┼──────────────────────────────────────────────────────────┤
 * │ files[]         │ REQUIRED. Image files (JPEG/PNG/WEBP) or a single video  │
 * │                 │ (MP4/MOV). Min 5 images recommended for good results.    │
 * │ title           │ Optional scene name (default: "Scene – <date>")          │
 * │ quality         │ "fast" (~5 min) | "balanced" (~15 min) | "high" (~30 min)│
 * │                 │ Default: "balanced". Pro plan only for "high".           │
 * │ enhanceImages   │ "true" | "false". Run Real-ESRGAN upscaling before GPU.  │
 * │                 │ Default: "true". Set to "false" for already sharp images.│
 * └─────────────────┴──────────────────────────────────────────────────────────┘
 *
 * Flutter example:
 *   var req = MultipartRequest('POST', Uri.parse('$base/api/upload'));
 *   req.headers['Authorization'] = 'Bearer $token';
 *   req.fields['title']   = 'Living Room';
 *   req.fields['quality'] = 'balanced';
 *   req.files.addAll(images.map((f) => MultipartFile.fromPath('files', f.path)));
 *
 * Response: { jobId, status, filesUploaded, estimatedMinutes, jobsRemaining }
 */
router.post('/', handleUpload);

module.exports = router;
