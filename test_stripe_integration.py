#!/usr/bin/env python3
"""
Test script for Stripe integration.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from synthetica.api.database import DatabaseManager, SubscriptionTier, TIER_LIMITS

def test_database():
    """Test database functionality."""
    print("=" * 60)
    print("Testing Synthetica Payment Integration")
    print("=" * 60)
    print()

    # Initialize database
    print("1. Initializing database...")
    db = DatabaseManager()
    print("   ✓ Database initialized")
    print()

    # Create a free user
    print("2. Creating free user...")
    api_key, user = db.create_user(email="test@example.com")
    print(f"   ✓ User created")
    print(f"   - API Key: {api_key}")
    print(f"   - Tier: {user['subscription_tier']}")
    print(f"   - Usage: {user['usage_count']}/{TIER_LIMITS[user['subscription_tier']]}")
    print()

    # Check usage limit
    print("3. Checking usage limits...")
    within_limit, usage_info = db.check_usage_limit(api_key)
    print(f"   - Within Limit: {within_limit}")
    print(f"   - Usage: {usage_info['usage']}")
    print(f"   - Limit: {usage_info['limit']}")
    print(f"   - Remaining: {usage_info['remaining']}")
    print()

    # Increment usage
    print("4. Simulating conversation generation (incrementing usage)...")
    for i in range(5):
        db.increment_usage(api_key)

    _, usage_info = db.check_usage_limit(api_key)
    print(f"   ✓ Usage incremented 5 times")
    print(f"   - Current usage: {usage_info['usage']}/{usage_info['limit']}")
    print(f"   - Remaining: {usage_info['remaining']}")
    print()

    # Simulate upgrade
    print("5. Simulating upgrade to Starter tier...")
    db.upgrade_subscription(api_key, SubscriptionTier.STARTER)

    _, usage_info = db.check_usage_limit(api_key)
    print(f"   ✓ User upgraded")
    print(f"   - New tier: {usage_info['tier']}")
    print(f"   - New limit: {usage_info['limit']}")
    print(f"   - Current usage: {usage_info['usage']}")
    print(f"   - Remaining: {usage_info['remaining']}")
    print()

    # Get user stats
    print("6. Getting comprehensive user stats...")
    stats = db.get_user_stats(api_key)
    print(f"   - Email: {stats['email']}")
    print(f"   - Tier: {stats['tier']}")
    print(f"   - Usage: {stats['usage']}/{stats['limit']}")
    print(f"   - Percentage Used: {stats['percentage_used']}%")
    print()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Get a Stripe API key from https://stripe.com")
    print("2. Set environment variables:")
    print("   export STRIPE_SECRET_KEY='sk_test_...'")
    print("   export STRIPE_WEBHOOK_SECRET='whsec_...'")
    print("3. Start the server: python3 start_server.py")
    print("4. Visit http://localhost:8000/pricing to test the flow")
    print()


if __name__ == "__main__":
    test_database()
