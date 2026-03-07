const express = require('express');
const { handleRunpodWebhook } = require('../controllers/webhook.controller');

const router = express.Router();

// ─── No JWT auth here ──────────────────────────────────────────────────────────
// Webhook authenticity is verified inside the controller using HMAC-SHA256.
// The GPU worker signs every payload with RUNPOD_WEBHOOK_SECRET.

/**
 * POST /api/webhooks/runpod
 *
 * Called by the Runpod GPU worker to report job progress and completion.
 * Must include header: x-runpod-signature: <HMAC-SHA256 of JSON body>
 *
 * Expected payload:
 * {
 *   jobId:                 string,   // matches MongoDB Job._id
 *   status:                string,   // "preprocessing" | "training" | "converting" | "completed" | "failed"
 *   progressPct:           number,   // 0–100
 *   runpodJobId:           string,   // Runpod's internal job ID (for debugging)
 *
 *   // Sent only when status === "completed":
 *   glbCloudinaryId:       string,   // Cloudinary public_id of the .glb file
 *   glbSecureUrl:          string,   // direct https URL
 *   thumbnailCloudinaryId: string,   // optional preview image public_id
 *   thumbnailSecureUrl:    string,   // optional preview https URL
 *   glbFileSizeBytes:      number,
 *
 *   // Sent only when status === "failed":
 *   errorMessage:          string,
 *   errorCode:             string,   // e.g. "COLMAP_FAILED", "GPU_OOM", "TOO_FEW_IMAGES"
 *   errorStage:            string,   // stage where failure occurred
 * }
 */
router.post('/runpod', handleRunpodWebhook);

module.exports = router;
