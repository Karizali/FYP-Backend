const express = require('express');
const { body } = require('express-validator');
const {
  register,
  login,
  getProfile,
  updateProfile,
  changePassword,
  updateFcmToken,
} = require('../controllers/auth.controller');
const { authenticate } = require('../middleware/auth');

const router = express.Router();

// ─── Validation rule sets ──────────────────────────────────────────────────────

const registerRules = [
  body('name')
    .trim()
    .notEmpty().withMessage('Name is required.')
    .isLength({ max: 100 }).withMessage('Name cannot exceed 100 characters.'),
  body('email')
    .isEmail().withMessage('A valid email address is required.')
    .normalizeEmail(),
  body('password')
    .isLength({ min: 8 }).withMessage('Password must be at least 8 characters.')
    .matches(/\d/).withMessage('Password must contain at least one number.'),
];

const loginRules = [
  body('email')
    .isEmail().withMessage('A valid email address is required.')
    .normalizeEmail(),
  body('password')
    .notEmpty().withMessage('Password is required.'),
];

const changePasswordRules = [
  body('currentPassword')
    .notEmpty().withMessage('Current password is required.'),
  body('newPassword')
    .isLength({ min: 8 }).withMessage('New password must be at least 8 characters.')
    .matches(/\d/).withMessage('New password must contain at least one number.'),
];

// ─── Public routes (no token required) ────────────────────────────────────────

/**
 * POST /api/auth/register
 * Body: { name, email, password }
 * Returns: { token, user }
 */
router.post('/register', registerRules, register);

/**
 * POST /api/auth/login
 * Body: { email, password }
 * Returns: { token, user }
 */
router.post('/login', loginRules, login);

// ─── Protected routes (JWT required) ──────────────────────────────────────────

/**
 * GET /api/auth/me
 * Returns the authenticated user's profile
 */
router.get('/me', authenticate, getProfile);

/**
 * PUT /api/auth/me
 * Body: { name }
 * Updates display name
 */
router.put('/me', authenticate, updateProfile);

/**
 * PUT /api/auth/password
 * Body: { currentPassword, newPassword }
 */
router.put('/password', authenticate, changePasswordRules, changePassword);

/**
 * PUT /api/auth/fcm-token
 * Body: { fcmToken }
 * Called by Flutter after login to register push notification token
 */
router.put('/fcm-token', authenticate, updateFcmToken);

module.exports = router;
