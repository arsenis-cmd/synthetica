# Payment Integration Guide

This guide explains how to set up and use the Stripe payment integration in Synthetica.

## Overview

Synthetica now includes a tiered pricing system with usage tracking and Stripe integration:

- **Free Tier**: 100 conversations (no payment required)
- **Starter Tier**: 1,000 conversations for $99 (one-time payment)
- **Growth Tier**: 10,000 conversations for $450 (one-time payment)

## Architecture

### Database

SQLite database (`synthetica.db`) tracks:
- **Users**: API keys, subscription tiers, usage counts
- **Payments**: Stripe sessions, payment status, transaction history

### API Endpoints

#### User Management
- `POST /api/users` - Create a new free-tier user
- `GET /api/usage` - Check current usage and limits (requires X-API-Key header)

#### Conversation Generation
- `POST /api/generate` - Generate conversations (enforces usage limits, requires X-API-Key header)

#### Payments
- `POST /api/checkout` - Create Stripe checkout session (requires X-API-Key header)
- `POST /api/webhooks/stripe` - Handle Stripe webhook events

#### UI Routes
- `GET /` - Main conversation generator interface
- `GET /pricing` - Pricing and account management page

## Setup Instructions

### 1. Get Stripe API Keys

1. Sign up at https://stripe.com
2. Go to Developers > API keys
3. Copy your test keys:
   - Secret key (starts with `sk_test_`)
   - Publishable key (starts with `pk_test_`)

### 2. Set Environment Variables

```bash
export ANTHROPIC_API_KEY='your-anthropic-key'
export STRIPE_SECRET_KEY='sk_test_...'
export STRIPE_WEBHOOK_SECRET='whsec_...'  # Optional for local testing
```

### 3. Initialize Database

The database is automatically created on first run. To test:

```bash
python3 test_stripe_integration.py
```

### 4. Configure Stripe Webhooks (Production Only)

For production, configure a webhook in Stripe Dashboard:

1. Go to Developers > Webhooks
2. Add endpoint: `https://yourdomain.com/api/webhooks/stripe`
3. Select event: `checkout.session.completed`
4. Copy the webhook secret and add to environment variables

### 5. Start the Server

```bash
python3 start_server.py
```

## Usage Flow

### For End Users

1. **Visit Pricing Page**: Go to `/pricing`
2. **Create Free Account**: Click "Get Started" to generate an API key
3. **Copy API Key**: Store the generated API key securely
4. **Generate Conversations**: Use the API key in the generator interface or via API
5. **Monitor Usage**: View usage stats on the pricing page
6. **Upgrade Plan**: When approaching limit, click upgrade button
7. **Complete Payment**: Redirected to Stripe Checkout, complete payment
8. **Automatic Upgrade**: Account automatically upgraded after payment

### For Developers

#### Create a User

```bash
curl -X POST http://localhost:8000/api/users \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com"}'
```

Response:
```json
{
  "api_key": "sk-synth-...",
  "subscription_tier": "free",
  "usage_limit": 100
}
```

#### Check Usage

```bash
curl http://localhost:8000/api/usage \
  -H "X-API-Key: sk-synth-..."
```

Response:
```json
{
  "tier": "free",
  "usage": 5,
  "limit": 100,
  "remaining": 95,
  "percentage_used": 5.0
}
```

#### Generate Conversations

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-synth-..." \
  -d '{
    "industry": "technology",
    "topics": ["software bug"],
    "count": 3,
    "customer_tone": "professional",
    "agent_style": "professional",
    "message_count": 6
  }'
```

Response includes usage_info:
```json
{
  "success": true,
  "count": 3,
  "conversations": [...],
  "usage_info": {
    "tier": "free",
    "usage": 8,
    "limit": 100,
    "remaining": 92
  }
}
```

#### Create Checkout Session

```bash
curl -X POST http://localhost:8000/api/checkout \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-synth-..." \
  -d '{
    "tier": "starter",
    "success_url": "http://localhost:8000/pricing?success=true",
    "cancel_url": "http://localhost:8000/pricing?canceled=true"
  }'
```

Response:
```json
{
  "session_id": "cs_test_...",
  "checkout_url": "https://checkout.stripe.com/..."
}
```

## Rate Limiting

The API enforces usage limits based on subscription tier:

- Returns `429 Too Many Requests` when limit exceeded
- Error response includes upgrade URL
- Response body shows current usage and remaining quota

Example error response:
```json
{
  "error": "Usage limit exceeded",
  "tier": "free",
  "limit": 100,
  "usage": 100,
  "upgrade_url": "/pricing"
}
```

## Webhook Handling

The webhook endpoint (`/api/webhooks/stripe`) handles:

1. **Signature Verification**: Validates webhook authenticity
2. **Payment Completion**: Processes `checkout.session.completed` events
3. **Automatic Upgrade**: Updates user's subscription tier
4. **Transaction Logging**: Records payment in database

## Testing

### Local Testing (Without Stripe)

1. Create users and test usage tracking:
```bash
python3 test_stripe_integration.py
```

2. Generate conversations with usage limits:
```bash
# Generate conversations until limit reached
for i in {1..105}; do
  curl -X POST http://localhost:8000/api/generate \
    -H "X-API-Key: sk-synth-..." \
    -H "Content-Type: application/json" \
    -d '{"industry":"tech","topics":["bug"],"count":1}'
done
```

### Testing with Stripe

1. Use Stripe test card numbers:
   - Success: `4242 4242 4242 4242`
   - Decline: `4000 0000 0000 0002`
   - Any future expiry date and CVC

2. Test the complete flow:
   - Create free account
   - Generate some conversations
   - Attempt upgrade
   - Complete payment with test card
   - Verify account upgraded

3. Test webhook locally with Stripe CLI:
```bash
stripe listen --forward-to localhost:8000/api/webhooks/stripe
```

## Database Schema

### Users Table
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    api_key TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE,
    subscription_tier TEXT DEFAULT 'free',
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Payments Table
```sql
CREATE TABLE payments (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    stripe_session_id TEXT UNIQUE,
    stripe_payment_intent_id TEXT,
    amount INTEGER NOT NULL,
    currency TEXT DEFAULT 'usd',
    subscription_tier TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);
```

## Security Considerations

1. **API Key Storage**: API keys are stored in plain text in the database. For production, consider encryption.
2. **Webhook Verification**: Always verify webhook signatures in production.
3. **Rate Limiting**: Usage limits are enforced at the application level.
4. **HTTPS**: Use HTTPS in production for all API requests and webhooks.

## Troubleshooting

### Webhook Not Received

- Check Stripe Dashboard > Developers > Webhooks for delivery logs
- Verify webhook endpoint is accessible from internet
- Check webhook secret is correct

### Payment Completed But Account Not Upgraded

- Check server logs for webhook processing errors
- Verify `checkout.session.completed` event is enabled
- Check payments table for status

### Usage Count Not Incrementing

- Verify `db.increment_usage()` is called after each generation
- Check database write permissions
- Review application logs

## Production Checklist

- [ ] Use production Stripe keys (not test keys)
- [ ] Set up webhook endpoint with HTTPS
- [ ] Configure webhook secret
- [ ] Enable webhook signature verification
- [ ] Implement API key encryption
- [ ] Set up database backups
- [ ] Monitor webhook delivery
- [ ] Implement proper error handling
- [ ] Add logging and monitoring
- [ ] Test payment flow end-to-end

## Support

For issues or questions:
1. Check server logs for errors
2. Review Stripe Dashboard for payment status
3. Test with Stripe test mode first
4. Consult Stripe documentation: https://stripe.com/docs
