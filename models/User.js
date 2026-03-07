const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

// ─── Plan limits ──────────────────────────────────────────────────────────────
const PLAN_LIMITS = {
  free: {
    jobsPerMonth: 5,
    maxFilesPerJob: 20,
    maxQuality: 'balanced',   // cannot use 'high' quality
  },
  pro: {
    jobsPerMonth: 100,
    maxFilesPerJob: 50,
    maxQuality: 'high',
  },
};

// ─── Schema ───────────────────────────────────────────────────────────────────
const userSchema = new mongoose.Schema(
  {
    name: {
      type:      String,
      required:  [true, 'Name is required'],
      trim:      true,
      maxlength: [100, 'Name cannot exceed 100 characters'],
    },

    email: {
      type:     String,
      required: [true, 'Email is required'],
      unique:   true,
      lowercase: true,
      trim:     true,
      match:    [/^\S+@\S+\.\S+$/, 'Please provide a valid email address'],
    },

    password: {
      type:      String,
      required:  [true, 'Password is required'],
      minlength: [8, 'Password must be at least 8 characters'],
      select:    false,   // never returned in queries unless explicitly asked
    },

    // ─── Push Notifications ────────────────────────────────────────────────
    // Flutter app registers its FCM token after login
    fcmToken: {
      type:    String,
      default: null,
      select:  false,     // only fetched when sending a notification
    },

    // ─── Subscription plan ────────────────────────────────────────────────
    plan: {
      type:    String,
      enum:    ['free', 'pro'],
      default: 'free',
    },

    // ─── Usage tracking (reset monthly by a cron job) ─────────────────────
    jobsThisMonth: {
      type:    Number,
      default: 0,
      min:     0,
    },
    usageResetAt: {
      type:    Date,
      default: () => nextMonthStart(),
    },

    // ─── Account state ────────────────────────────────────────────────────
    isActive: {
      type:    Boolean,
      default: true,
    },
    lastLoginAt: {
      type:    Date,
      default: null,
    },
  },
  {
    timestamps: true,   // adds createdAt + updatedAt
    toJSON:     { virtuals: true },
    toObject:   { virtuals: true },
  }
);

// ─── Indexes ──────────────────────────────────────────────────────────────────
userSchema.index({ email: 1 });

// ─── Virtuals ─────────────────────────────────────────────────────────────────
userSchema.virtual('planLimits').get(function () {
  return PLAN_LIMITS[this.plan];
});

userSchema.virtual('jobsRemaining').get(function () {
  return Math.max(0, PLAN_LIMITS[this.plan].jobsPerMonth - this.jobsThisMonth);
});

// ─── Hooks ────────────────────────────────────────────────────────────────────
// Hash password before save (only when modified)
// Note: async hooks in Mongoose must NOT use the next() parameter —
// Mongoose detects the async function and waits for the promise to resolve.
userSchema.pre('save', async function () {
  if (!this.isModified('password')) return;
  this.password = await bcrypt.hash(this.password, 12);
});

// Auto-reset monthly usage counter when the reset date passes
userSchema.pre('save', function () {
  if (this.usageResetAt && new Date() >= this.usageResetAt) {
    this.jobsThisMonth = 0;
    this.usageResetAt  = nextMonthStart();
  }
});

// ─── Instance Methods ─────────────────────────────────────────────────────────

// Verify a plain-text password against the stored hash
userSchema.methods.comparePassword = async function (candidatePassword) {
  return bcrypt.compare(candidatePassword, this.password);
};

// Check whether the user is allowed to create another job this month
userSchema.methods.canCreateJob = function () {
  const limit = PLAN_LIMITS[this.plan].jobsPerMonth;
  return this.jobsThisMonth < limit;
};

// Return a safe public object (no password, no fcmToken)
userSchema.methods.toPublic = function () {
  return {
    id:             this._id,
    name:           this.name,
    email:          this.email,
    plan:           this.plan,
    jobsThisMonth:  this.jobsThisMonth,
    jobsRemaining:  this.jobsRemaining,
    planLimits:     this.planLimits,
    lastLoginAt:    this.lastLoginAt,
    createdAt:      this.createdAt,
  };
};

// ─── Helpers ──────────────────────────────────────────────────────────────────
function nextMonthStart() {
  const d = new Date();
  return new Date(d.getFullYear(), d.getMonth() + 1, 1);
}

module.exports = mongoose.model('User', userSchema);
module.exports.PLAN_LIMITS = PLAN_LIMITS;