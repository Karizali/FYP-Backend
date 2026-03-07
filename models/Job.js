const mongoose = require('mongoose');

// ─── Status machine ───────────────────────────────────────────────────────────
//
//   queued ──► preprocessing ──► training ──► converting ──► done
//      │             │               │              │
//      └─────────────┴───────────────┴──────────────┴──────► failed
//
const JOB_STATUSES = [
  'queued',         // uploaded, waiting in Bull queue
  'preprocessing',  // image enhancement (Real-ESRGAN) running
  'training',       // Gaussian splatting training on GPU (Runpod)
  'converting',     // .ply → .glb conversion
  'done',           // GLB ready for download
  'failed',         // unrecoverable error
];

// Valid forward transitions — prevents accidental backwards moves
const ALLOWED_TRANSITIONS = {
  queued:         ['preprocessing', 'failed'],
  preprocessing:  ['training',      'failed'],
  training:       ['converting',    'failed'],
  converting:     ['done',          'failed'],
  done:           [],
  failed:         [],
};

// ─── Sub-schema: one uploaded input file ─────────────────────────────────────
const inputFileSchema = new mongoose.Schema(
  {
    originalName:  { type: String, required: true },
    cloudinaryId:  { type: String, required: true },  // public_id in Cloudinary
    secureUrl:     { type: String, required: true },  // full https URL
    resourceType:  { type: String, enum: ['image', 'video'], required: true },
    mimeType:      { type: String, required: true },
    sizeBytes:     { type: Number, required: true },
    folder:        { type: String },                  // Cloudinary folder path
  },
  { _id: false }
);

// ─── Sub-schema: processing output ───────────────────────────────────────────
const outputSchema = new mongoose.Schema(
  {
    glbCloudinaryId:       { type: String, default: null },  // raw GLB file
    glbSecureUrl:          { type: String, default: null },
    thumbnailCloudinaryId: { type: String, default: null },  // preview image
    thumbnailSecureUrl:    { type: String, default: null },
    fileSizeBytes:         { type: Number, default: null },
  },
  { _id: false }
);

// ─── Sub-schema: job settings chosen by user ─────────────────────────────────
const settingsSchema = new mongoose.Schema(
  {
    enhanceImages: {
      type:    Boolean,
      default: true,      // run Real-ESRGAN before sending to GPU
    },
    quality: {
      type:    String,
      enum:    ['fast', 'balanced', 'high'],
      default: 'balanced',
      // fast      ~5 min  — fewer training iterations
      // balanced  ~15 min — good trade-off (default)
      // high      ~30 min — maximum quality
    },
  },
  { _id: false }
);

// ─── Sub-schema: timeline timestamps ─────────────────────────────────────────
const timelineSchema = new mongoose.Schema(
  {
    queuedAt:       { type: Date, default: Date.now },
    preprocessedAt: { type: Date, default: null },
    trainingAt:     { type: Date, default: null },
    convertingAt:   { type: Date, default: null },
    completedAt:    { type: Date, default: null },
    failedAt:       { type: Date, default: null },
  },
  { _id: false }
);

// ─── Main Job Schema ──────────────────────────────────────────────────────────
const jobSchema = new mongoose.Schema(
  {
    userId: {
      type:     mongoose.Schema.Types.ObjectId,
      ref:      'User',
      required: true,
      index:    true,
    },

    title: {
      type:      String,
      default:   'Untitled Scene',
      trim:      true,
      maxlength: [200, 'Title cannot exceed 200 characters'],
    },

    // ─── Status & Progress ─────────────────────────────────────────────────
    status: {
      type:    String,
      enum:    JOB_STATUSES,
      default: 'queued',
      index:   true,
    },
    progressPct: {
      type:    Number,
      default: 0,
      min:     0,
      max:     100,
    },

    // ─── Input ────────────────────────────────────────────────────────────
    inputType: {
      type:     String,
      enum:     ['images', 'video'],
      required: true,
    },
    inputFiles: {
      type:    [inputFileSchema],
      default: [],
    },

    // ─── Output ───────────────────────────────────────────────────────────
    output: {
      type:    outputSchema,
      default: () => ({}),
    },

    // ─── Settings ─────────────────────────────────────────────────────────
    settings: {
      type:    settingsSchema,
      default: () => ({}),
    },

    // ─── Timeline ─────────────────────────────────────────────────────────
    timeline: {
      type:    timelineSchema,
      default: () => ({}),
    },

    // ─── Runpod integration ───────────────────────────────────────────────
    runpodJobId: {
      type:    String,
      default: null,
      index:   true,       // looked up frequently in webhook callbacks
    },

    // ─── Error info ───────────────────────────────────────────────────────
    error: {
      message: { type: String, default: null },
      code:    { type: String, default: null },  // e.g. 'COLMAP_FAILED', 'GPU_OOM'
      stage:   { type: String, default: null },  // which stage it failed at
    },

    // ─── Soft-delete flag (files cleaned up by a cron job) ────────────────
    deletedAt: {
      type:    Date,
      default: null,
    },
  },
  {
    timestamps: true,
    toJSON:     { virtuals: true },
    toObject:   { virtuals: true },
  }
);

// ─── Indexes ──────────────────────────────────────────────────────────────────
jobSchema.index({ userId: 1, createdAt: -1 });   // list jobs sorted by newest
jobSchema.index({ userId: 1, status: 1 });        // filter by status
jobSchema.index({ deletedAt: 1 });                // cleanup cron query

// ─── Virtuals ─────────────────────────────────────────────────────────────────
// Total processing time in seconds (from first GPU touch to completion)
jobSchema.virtual('durationSeconds').get(function () {
  const start = this.timeline?.trainingAt || this.timeline?.preprocessedAt;
  const end   = this.timeline?.completedAt || this.timeline?.failedAt;
  if (start && end) return Math.round((end - start) / 1000);
  return null;
});

// Human-readable estimated wait based on quality setting
jobSchema.virtual('estimatedMinutes').get(function () {
  const map = { fast: 5, balanced: 15, high: 30 };
  return map[this.settings?.quality] || 15;
});

// Is the job currently being processed (i.e. in-flight)
jobSchema.virtual('isProcessing').get(function () {
  return ['queued', 'preprocessing', 'training', 'converting'].includes(this.status);
});

// ─── Instance Methods ─────────────────────────────────────────────────────────

/**
 * Advance job to a new status with optional field updates.
 * Validates the transition is allowed before saving.
 *
 * Usage:
 *   await job.transition('training', { progressPct: 10, runpodJobId: 'rp_xyz' });
 */
jobSchema.methods.transition = async function (newStatus, extra = {}) {
  const allowed = ALLOWED_TRANSITIONS[this.status];

  if (!allowed.includes(newStatus)) {
    throw new Error(
      `Invalid status transition: ${this.status} → ${newStatus}. ` +
      `Allowed: [${allowed.join(', ')}]`
    );
  }

  this.status = newStatus;

  // Stamp the matching timeline field
  const timelineMap = {
    preprocessing: 'preprocessedAt',
    training:      'trainingAt',
    converting:    'convertingAt',
    done:          'completedAt',
    failed:        'failedAt',
  };
  if (timelineMap[newStatus]) {
    this.timeline[timelineMap[newStatus]] = new Date();
  }

  // Apply any extra fields (progressPct, output, error, runpodJobId…)
  Object.assign(this, extra);

  return this.save();
};

/**
 * Mark the job as failed with structured error info.
 */
jobSchema.methods.fail = async function (message, code = null, stage = null) {
  return this.transition('failed', {
    error: { message, code, stage: stage || this.status },
    progressPct: this.progressPct,   // keep last known progress
  });
};

/**
 * Return a safe summary for the Flutter app (no internal Cloudinary IDs).
 */
jobSchema.methods.toSummary = function () {
  return {
    id:                this._id,
    title:             this.title,
    status:            this.status,
    progressPct:       this.progressPct,
    inputType:         this.inputType,
    fileCount:         this.inputFiles?.length ?? 0,
    settings:          this.settings,
    timeline:          this.timeline,
    durationSeconds:   this.durationSeconds,
    estimatedMinutes:  this.estimatedMinutes,
    hasResult:         this.status === 'done',
    error:             this.status === 'failed' ? this.error : null,
    createdAt:         this.createdAt,
  };
};

module.exports = mongoose.model('Job', jobSchema);
module.exports.JOB_STATUSES         = JOB_STATUSES;
module.exports.ALLOWED_TRANSITIONS  = ALLOWED_TRANSITIONS;