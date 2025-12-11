# Quick Deployment Commands

## Step 1: Initialize Git and Push to GitHub

```bash
cd /Users/arsenispapachristos/synthetica

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Ready for Railway deployment"

# Create a new repository on GitHub (https://github.com/new)
# Then run:
git remote add origin https://github.com/YOUR_USERNAME/synthetica.git
git branch -M main
git push -u origin main
```

## Step 2: Deploy to Railway (Web Interface)

1. Go to https://railway.app/new
2. Click "Deploy from GitHub repo"
3. Select your `synthetica` repository
4. Click "Deploy Now"

## Step 3: Add PostgreSQL Database

1. In Railway dashboard, click "+ New"
2. Select "Database" → "Add PostgreSQL"
3. Done! `DATABASE_URL` is automatically linked

## Step 4: Add Environment Variables

In Railway → Your Service → Variables tab, add these:

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
STRIPE_SECRET_KEY=your_stripe_secret_key_here
STRIPE_WEBHOOK_SECRET=whsec_YOUR_WEBHOOK_SECRET_HERE
```

**Note:** The `DATABASE_URL` should already be there from Step 3.

## Step 5: Get Your Deployment URL

1. Go to Settings → Networking
2. Click "Generate Domain"
3. Copy your URL (e.g., `https://synthetica-production.up.railway.app`)

## Step 6: Configure Stripe Webhook

1. Go to https://dashboard.stripe.com/webhooks
2. Click "Add endpoint"
3. URL: `https://YOUR-RAILWAY-URL.up.railway.app/api/webhooks/stripe`
4. Events to send: `checkout.session.completed`
5. Copy the signing secret and update it in Railway env vars

## Step 7: Test Your Deployment

Visit these URLs (replace with your Railway domain):

```
https://your-app.up.railway.app/             # Homepage
https://your-app.up.railway.app/generate     # Generator
https://your-app.up.railway.app/marketplace  # Marketplace
https://your-app.up.railway.app/pricing      # Pricing
https://your-app.up.railway.app/docs         # API Docs
https://your-app.up.railway.app/health       # Health Check
```

## Future Updates

Every time you want to deploy changes:

```bash
git add .
git commit -m "Your update description"
git push origin main
```

Railway will automatically redeploy! 🚀

## Troubleshooting

### View Logs
Railway Dashboard → Deployments → Click deployment → View logs

### Database Issues
Check that `DATABASE_URL` exists in environment variables

### App Won't Start
Check deployment logs for Python errors

---

## Cost Estimate
- **Railway Hobby**: $5/month (includes $5 credit)
- **PostgreSQL**: ~$2-5/month
- **Total**: ~$5-10/month for moderate traffic

## Files Created for Deployment

✅ `requirements.txt` - Updated with PostgreSQL dependencies
✅ `Procfile` - Tells Railway how to start the app
✅ `railway.json` - Railway configuration
✅ `.railwayignore` - Files to exclude from deployment
✅ `.gitignore` - Prevents committing sensitive data
✅ `synthetica/api/database.py` - Now supports PostgreSQL!
✅ `RAILWAY_DEPLOYMENT.md` - Full deployment guide
✅ `DEPLOY_COMMANDS.md` - This file!

## Ready to Deploy! 🎉

Everything is configured. Just follow steps 1-7 above!
