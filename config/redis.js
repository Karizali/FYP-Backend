const { createClient } = require('redis');
const logger = require('../utils/logger');

let redisClient = null;

async function connectRedis() {
  redisClient = createClient({
    url: process.env.REDIS_URL || 'redis://localhost:6379',
    socket: {
      reconnectStrategy: (retries) => {
        if (retries > 10) {
          logger.error('Redis: max reconnection attempts reached');
          return new Error('Too many reconnect attempts');
        }
        const delay = Math.min(retries * 500, 5000); // backoff up to 5s
        logger.warn(`Redis: reconnecting in ${delay}ms (attempt ${retries})`);
        return delay;
      },
    },
  });

  redisClient.on('connect',      () => logger.info('✅ Redis connected'));
  redisClient.on('ready',        () => logger.info('✅ Redis ready'));
  redisClient.on('error',  (err) => logger.error(`Redis error: ${err.message}`));
  redisClient.on('end',          () => logger.warn('⚠️  Redis connection closed'));

  await redisClient.connect();
}

function getRedisClient() {
  if (!redisClient || !redisClient.isOpen) {
    throw new Error('Redis client is not connected. Ensure connectRedis() was called on startup.');
  }
  return redisClient;
}

// Graceful shutdown
process.on('SIGINT', async () => {
  if (redisClient?.isOpen) {
    await redisClient.quit();
    logger.info('Redis connection closed due to app termination');
  }
});

module.exports = { connectRedis, getRedisClient };
