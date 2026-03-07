const admin = require('firebase-admin');
const User = require('../models/User');
const logger = require('../utils/logger');

// ─── Firebase Initialization ───────────────────────────────────────────────────
// Initialized lazily on first use so the server boots even if the
// service account file is missing (dev environments without Firebase)
let _app = null;

function getApp() {
  if (_app) return _app;

  try {
    // Support both a file path and a JSON string in the env var
    const raw = process.env.FIREBASE_SERVICE_ACCOUNT;
    let credential;

    if (raw) {
      // Inline JSON string (recommended for cloud deployments — no file needed)
      const serviceAccount = JSON.parse(raw);
      credential = admin.credential.cert(serviceAccount);
    } else if (process.env.FIREBASE_SERVICE_ACCOUNT_PATH) {
      // Local file path (for development)
      const serviceAccount = require(process.env.FIREBASE_SERVICE_ACCOUNT_PATH);
      credential = admin.credential.cert(serviceAccount);
    } else {
      throw new Error(
        'Firebase not configured. Set FIREBASE_SERVICE_ACCOUNT (JSON string) ' +
        'or FIREBASE_SERVICE_ACCOUNT_PATH (file path) in your .env'
      );
    }

    _app = admin.initializeApp({ credential });
    logger.info('✅ Firebase Admin initialized');
  } catch (err) {
    logger.warn(`⚠️  Firebase init failed: ${err.message} — push notifications disabled`);
    _app = null;
  }

  return _app;
}

// ─── Core send function ────────────────────────────────────────────────────────

/**
 * Send a push notification to a single FCM device token.
 * Failures are logged but never thrown — a failed push must never break
 * the main request/response flow.
 *
 * @param {string} fcmToken  - Device token registered by Flutter app
 * @param {string} title     - Notification title
 * @param {string} body      - Notification body text
 * @param {Object} data      - Optional key-value payload sent to the app
 * @returns {Promise<boolean>} true if sent, false if failed/skipped
 */
async function sendPush(fcmToken, title, body, data = {}) {
  const app = getApp();
  if (!app) return false;            // Firebase not configured — skip silently

  if (!fcmToken) {
    logger.debug('sendPush: no FCM token provided, skipping');
    return false;
  }

  try {
    const messageId = await app.messaging().send({
      token: fcmToken,

      // Visible notification (shown in system tray)
      notification: { title, body },

      // Invisible data payload — Flutter app reads this in onMessage handler
      data: {
        ...Object.fromEntries(
          Object.entries(data).map(([k, v]) => [k, String(v)]) // FCM requires string values
        ),
        timestamp: Date.now().toString(),
      },

      // Android: high priority wakes up the device
      android: {
        priority: 'high',
        notification: {
          sound:       'default',
          channelId:   'gaussian_jobs',   // must be created in Flutter app
          clickAction: 'FLUTTER_NOTIFICATION_CLICK',
        },
      },

      // iOS: content-available wakes the app in background
      apns: {
        payload: {
          aps: {
            sound:             'default',
            'content-available': 1,
          },
        },
      },
    });

    logger.info(`Push sent | messageId=${messageId} | token=${fcmToken.slice(0, 20)}...`);
    return true;
  } catch (err) {
    // Token is invalid or app was uninstalled — remove stale token from DB
    if (
      err.code === 'messaging/invalid-registration-token' ||
      err.code === 'messaging/registration-token-not-registered'
    ) {
      logger.warn(`Stale FCM token detected, clearing from DB`);
      await User.findOneAndUpdate({ fcmToken }, { fcmToken: null }).catch(() => {});
    } else {
      logger.error(`Push notification failed: ${err.message}`);
    }
    return false;
  }
}

// ─── Domain-specific notification helpers ─────────────────────────────────────

/**
 * Notify the job owner when their 3D scene finishes (or fails).
 * Looks up the user's FCM token from DB automatically.
 *
 * @param {string|ObjectId} userId
 * @param {string}          jobId
 * @param {'done'|'failed'} status
 */
async function notifyJobComplete(userId, jobId, status) {
  try {
    // Fetch fcmToken — it has select:false in the schema so we must be explicit
    const user = await User.findById(userId).select('+fcmToken');
    if (!user?.fcmToken) {
      logger.debug(`notifyJobComplete: user ${userId} has no FCM token`);
      return;
    }

    const isDone = status === 'done';

    await sendPush(
      user.fcmToken,
      isDone ? '✅ Your 3D scene is ready!' : '❌ Processing failed',
      isDone
        ? 'Tap to view your 3D scene in the app.'
        : 'Something went wrong. Tap to see details.',
      {
        type:   'JOB_COMPLETE',
        jobId:  jobId.toString(),
        status,
      }
    );
  } catch (err) {
    // Non-fatal — log and continue
    logger.error(`notifyJobComplete error for user ${userId}: ${err.message}`);
  }
}

/**
 * Notify the user that their job has started processing on the GPU.
 * Good UX touch — reassures user the job didn't get stuck in the queue.
 *
 * @param {string|ObjectId} userId
 * @param {string}          jobId
 * @param {number}          estimatedMinutes
 */
async function notifyJobStarted(userId, jobId, estimatedMinutes) {
  try {
    const user = await User.findById(userId).select('+fcmToken');
    if (!user?.fcmToken) return;

    await sendPush(
      user.fcmToken,
      '⚙️ Processing started',
      `Your 3D scene is being built. Estimated time: ~${estimatedMinutes} minutes.`,
      {
        type:              'JOB_STARTED',
        jobId:             jobId.toString(),
        estimatedMinutes:  String(estimatedMinutes),
      }
    );
  } catch (err) {
    logger.error(`notifyJobStarted error for user ${userId}: ${err.message}`);
  }
}

module.exports = {
  sendPush,
  notifyJobComplete,
  notifyJobStarted,
};
