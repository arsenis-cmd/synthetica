"""
Database management for user tracking and subscriptions.
"""
import secrets
import sqlite3
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Check if we should use PostgreSQL
USE_POSTGRES = bool(os.getenv("DATABASE_URL"))

if USE_POSTGRES:
    import psycopg2
    from psycopg2.extras import RealDictCursor


class SubscriptionTier(str, Enum):
    """Subscription tier options."""
    FREE = "free"
    STARTER = "starter"
    GROWTH = "growth"


TIER_LIMITS = {
    SubscriptionTier.FREE: 100,
    SubscriptionTier.STARTER: 1000,
    SubscriptionTier.GROWTH: 10000,
}

TIER_PRICES = {
    SubscriptionTier.STARTER: 9900,  # $99 in cents
    SubscriptionTier.GROWTH: 45000,  # $450 in cents
}


class DatabaseManager:
    """Manages database operations for user and subscription tracking."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection."""
        self.use_postgres = USE_POSTGRES
        self.database_url = os.getenv("DATABASE_URL")

        if not self.use_postgres:
            if db_path is None:
                db_path = str(Path(__file__).parent.parent.parent / "synthetica.db")
            self.db_path = db_path

        self._init_database()

    def get_connection(self):
        """Get database connection based on environment."""
        if self.use_postgres:
            return psycopg2.connect(self.database_url)
        else:
            return sqlite3.connect(self.db_path)

    def _init_database(self):
        """Initialize database tables."""
        if self.use_postgres:
            # PostgreSQL syntax
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Users table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        api_key TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE,
                        subscription_tier TEXT NOT NULL DEFAULT 'free',
                        usage_count INTEGER NOT NULL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Payments table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS payments (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        stripe_session_id TEXT UNIQUE,
                        stripe_payment_intent_id TEXT,
                        amount INTEGER NOT NULL,
                        currency TEXT DEFAULT 'usd',
                        subscription_tier TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        completed_at TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)

                conn.commit()
                logger.info("PostgreSQL database initialized successfully")
        else:
            # SQLite syntax
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Users table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        api_key TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE,
                        subscription_tier TEXT NOT NULL DEFAULT 'free',
                        usage_count INTEGER NOT NULL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Payments table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS payments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        stripe_session_id TEXT UNIQUE,
                        stripe_payment_intent_id TEXT,
                        amount INTEGER NOT NULL,
                        currency TEXT DEFAULT 'usd',
                        subscription_tier TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        completed_at TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)

                conn.commit()
                logger.info("SQLite database initialized successfully")

    def create_user(self, email: Optional[str] = None) -> Tuple[str, dict]:
        """
        Create a new user with a free tier subscription.

        Args:
            email: Optional email address

        Returns:
            Tuple of (api_key, user_dict)
        """
        api_key = f"sk-synth-{secrets.token_urlsafe(32)}"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO users (api_key, email, subscription_tier, usage_count)
                VALUES (%s, %s, %s, 0)
                """ if self.use_postgres else """
                INSERT INTO users (api_key, email, subscription_tier, usage_count)
                VALUES (?, ?, ?, 0)
                """,
                (api_key, email, SubscriptionTier.FREE)
            )
            user_id = cursor.lastrowid if not self.use_postgres else cursor.fetchone()
            conn.commit()

        user = self.get_user_by_api_key(api_key)
        logger.info(f"Created new user")
        return api_key, user

    def get_user_by_api_key(self, api_key: str) -> Optional[dict]:
        """Get user by API key."""
        with self.get_connection() as conn:
            if self.use_postgres:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
            else:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM users WHERE api_key = %s" if self.use_postgres else "SELECT * FROM users WHERE api_key = ?",
                (api_key,)
            )
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

    def get_user_by_email(self, email: str) -> Optional[dict]:
        """Get user by email."""
        with self.get_connection() as conn:
            if self.use_postgres:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
            else:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM users WHERE email = %s" if self.use_postgres else "SELECT * FROM users WHERE email = ?",
                (email,)
            )
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

    def increment_usage(self, api_key: str) -> bool:
        """
        Increment usage count for a user.

        Args:
            api_key: User's API key

        Returns:
            True if successful, False if user not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE users
                SET usage_count = usage_count + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE api_key = %s
                """ if self.use_postgres else """
                UPDATE users
                SET usage_count = usage_count + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE api_key = ?
                """,
                (api_key,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def check_usage_limit(self, api_key: str) -> Tuple[bool, dict]:
        """
        Check if user is within their usage limit.

        Args:
            api_key: User's API key

        Returns:
            Tuple of (within_limit, usage_info)
        """
        user = self.get_user_by_api_key(api_key)

        if not user:
            return False, {"error": "Invalid API key"}

        tier = user['subscription_tier']
        limit = TIER_LIMITS.get(tier, 0)
        usage = user['usage_count']

        usage_info = {
            "tier": tier,
            "usage": usage,
            "limit": limit,
            "remaining": max(0, limit - usage),
            "within_limit": usage < limit
        }

        return usage < limit, usage_info

    def upgrade_subscription(
        self,
        api_key: str,
        new_tier: SubscriptionTier,
        stripe_session_id: Optional[str] = None
    ) -> bool:
        """
        Upgrade user's subscription tier.

        Args:
            api_key: User's API key
            new_tier: New subscription tier
            stripe_session_id: Stripe checkout session ID

        Returns:
            True if successful
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE users
                SET subscription_tier = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE api_key = %s
                """ if self.use_postgres else """
                UPDATE users
                SET subscription_tier = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE api_key = ?
                """,
                (new_tier, api_key)
            )
            conn.commit()

            if cursor.rowcount > 0:
                logger.info(f"Upgraded user to {new_tier}")
                return True
            return False

    def create_payment(
        self,
        api_key: str,
        stripe_session_id: str,
        amount: int,
        subscription_tier: str
    ) -> bool:
        """
        Create a payment record.

        Args:
            api_key: User's API key
            stripe_session_id: Stripe checkout session ID
            amount: Amount in cents
            subscription_tier: Target subscription tier

        Returns:
            True if successful
        """
        user = self.get_user_by_api_key(api_key)
        if not user:
            return False

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO payments
                (user_id, stripe_session_id, amount, subscription_tier, status)
                VALUES (%s, %s, %s, %s, 'pending')
                """ if self.use_postgres else """
                INSERT INTO payments
                (user_id, stripe_session_id, amount, subscription_tier, status)
                VALUES (?, ?, ?, ?, 'pending')
                """,
                (user['id'], stripe_session_id, amount, subscription_tier)
            )
            conn.commit()
            return True

    def complete_payment(
        self,
        stripe_session_id: str,
        payment_intent_id: str
    ) -> bool:
        """
        Mark payment as completed and upgrade user.

        Args:
            stripe_session_id: Stripe checkout session ID
            payment_intent_id: Stripe payment intent ID

        Returns:
            True if successful
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Get payment info
            cursor.execute(
                """
                SELECT p.*, u.api_key
                FROM payments p
                JOIN users u ON p.user_id = u.id
                WHERE p.stripe_session_id = %s
                """ if self.use_postgres else """
                SELECT p.*, u.api_key
                FROM payments p
                JOIN users u ON p.user_id = u.id
                WHERE p.stripe_session_id = ?
                """,
                (stripe_session_id,)
            )
            payment = cursor.fetchone()

            if not payment:
                return False

            user_id, api_key, tier = payment[1], payment[-1], payment[5]

            # Update payment status
            cursor.execute(
                """
                UPDATE payments
                SET status = 'completed',
                    stripe_payment_intent_id = %s,
                    completed_at = CURRENT_TIMESTAMP
                WHERE stripe_session_id = %s
                """ if self.use_postgres else """
                UPDATE payments
                SET status = 'completed',
                    stripe_payment_intent_id = ?,
                    completed_at = CURRENT_TIMESTAMP
                WHERE stripe_session_id = ?
                """,
                (payment_intent_id, stripe_session_id)
            )

            # Upgrade user
            cursor.execute(
                """
                UPDATE users
                SET subscription_tier = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """ if self.use_postgres else """
                UPDATE users
                SET subscription_tier = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (tier, user_id)
            )

            conn.commit()
            logger.info(f"Completed payment and upgraded user to {tier}")
            return True

    def get_user_stats(self, api_key: str) -> Optional[dict]:
        """Get comprehensive user statistics."""
        user = self.get_user_by_api_key(api_key)
        if not user:
            return None

        tier = user['subscription_tier']
        limit = TIER_LIMITS.get(tier, 0)
        usage = user['usage_count']

        return {
            "email": user['email'],
            "tier": tier,
            "usage": usage,
            "limit": limit,
            "remaining": max(0, limit - usage),
            "percentage_used": round((usage / limit * 100) if limit > 0 else 0, 1),
            "created_at": user['created_at']
        }
